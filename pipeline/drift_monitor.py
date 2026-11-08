"""KS-based feature drift monitoring + surrogate alignment monitoring.

Feature drift: KS two-sample test (p < 0.01 AND D > 0.2) against round-0
    baseline — diagnostic only, NEVER stops the loop.

Surrogate alignment: checks whether the frozen Sparse Group Lasso surrogate
    remains aligned with the operational detector. Alignment degradation is
    the actual drift-related stop criterion.

Report structure::

    {
      "feature_drift": {
        "status": "OK" | "WARNING" | "INSUFFICIENT_SAMPLE_TAIL",
        "stop": false,       # always false — feature drift never stops
        "n_current": ...,
        "tests": [...]
      },
      "surrogate_alignment": {
        "status": "OK" | "WARNING" | "STOP"
                | "INSUFFICIENT_SAMPLE_TAIL" | "UNAVAILABLE",
        "stop": true/false,
        "n": ...,
        "agreement": ...,
        "corr": ...,
        "mae": ...,
        "baseline_agreement": ...,
        "baseline_corr": ...,
        "baseline_mae": ...
      },
      "overall": {
        "stop": true/false,
        "reason": "..."
      }
    }

Cached drift_report.json files in the old format (no "overall" key) are
regenerated automatically on next run.
"""
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from .constants import MIN_ACTIVE_FAKE


# ── Feature drift helpers ─────────────────────────────────────────────────────

def _load_col(path: str, col: str, min_n: int) -> Optional[np.ndarray]:
    """Return a finite float array for *col* from *path*, or None if unusable."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8")
        if col not in df.columns:
            return None
        arr = pd.to_numeric(df[col], errors="coerce").dropna().values.astype(float)
        return arr if len(arr) >= min_n else None
    except Exception:
        return None


def _count_rows(path: str) -> int:
    """Return the number of data rows in a CSV, or 0 if missing/unreadable."""
    if not os.path.exists(path):
        return 0
    try:
        df = pd.read_csv(path, encoding="utf-8")
        return len(df)
    except Exception:
        return 0


def _resolve_round0_shap_fallback(primary_path: str, detector: str) -> str:
    # Round-0 SHAP is written to Output/meta_frozen/<det>/ by the baseline pipeline; the
    # per-round path Output/rounds/round_000/shap/<det>/ stays empty in resume-from-baseline
    # flows, which would make KS tests skip every feature as insufficient_data.
    if os.path.exists(primary_path):
        return primary_path
    norm = os.path.abspath(primary_path).replace("\\", "/").split("/")
    try:
        idx = norm.index("rounds")
    except ValueError:
        return primary_path
    fallback = os.path.normpath(
        "/".join(norm[:idx] + ["meta_frozen", detector, "shap_attributions_per_id.csv"])
    )
    return fallback if os.path.exists(fallback) else primary_path


def _ks_result(arr0: np.ndarray, arr_t: np.ndarray, alpha: float, min_d: float) -> dict:
    stat, p_val = stats.ks_2samp(arr0, arr_t)
    return {
        "D": float(stat),
        "p": float(p_val),
        "drifted": bool(p_val < alpha and stat > min_d),
    }


# ── Surrogate alignment helpers ───────────────────────────────────────────────

def _load_frozen_surrogate(pipe_path: str):
    """Load the frozen Sparse Group Lasso pipeline. Returns (pipe, error_str)."""
    if not os.path.exists(pipe_path):
        return None, f"pipe not found: {pipe_path}"
    try:
        pipe = joblib.load(pipe_path)
        if not all(k in pipe.named_steps for k in ("impute", "scale", "clf")):
            return None, "pipe missing steps: impute, scale, clf"
        return pipe, None
    except Exception as e:
        return None, f"failed to load pipe: {e}"


def _load_pipe_feature_cols(pipe_path: str) -> Optional[List[str]]:
    """Load feature_cols from feature_meta.json next to the pipe."""
    meta_path = os.path.join(
        os.path.dirname(os.path.abspath(pipe_path)), "feature_meta.json"
    )
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cols = meta.get("feature_cols")
        if isinstance(cols, list) and len(cols) > 0:
            return cols
    except Exception:
        pass
    return None


def _detect_id_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("ID", "id", "Id"):
        if c in df.columns:
            return c
    return None


def _detect_detector_label_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("pred_label", "label", "prediction", "pred", "class"):
        if c in df.columns:
            return c
    return None


def _detect_detector_score_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("confidence", "score", "probability", "prob", "fake_prob", "real_prob"):
        if c in df.columns:
            return c
    return None


def _binarize_labels(label_series: pd.Series) -> Optional[np.ndarray]:
    """Convert detector label strings to 0/1 (FAKE=0, REAL=1). Returns None on failure."""
    s = label_series.astype(str).str.strip().str.upper()
    real_set = {"REAL", "TRUE", "POSITIVE", "LEGIT", "AUTHENTIC"}
    fake_set = {"FAKE", "FALSE", "NEGATIVE", "HOAX", "SYNTHETIC"}
    y = np.full(len(s), -1, dtype=int)
    y[s.isin(real_set)] = 1
    y[s.isin(fake_set)] = 0
    y[s.eq("LABEL_1")] = 1
    y[s.eq("LABEL_0")] = 0
    unk = y == -1
    if unk.any():
        num = pd.to_numeric(s[unk], errors="coerce")
        valid = num.isin([0.0, 1.0])
        idx = np.where(unk)[0]
        if valid.any():
            y[idx[valid.values]] = num[valid].astype(int).values
    if (y == -1).any():
        return None
    return y


def _predict_surrogate(
    pipe, X_df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run frozen pipe on X_df. Returns (labels, probs_real or None)."""
    X = X_df[feature_cols].copy()
    X_imp = pipe.named_steps["impute"].transform(X)
    X_std = pipe.named_steps["scale"].transform(X_imp)
    clf = pipe.named_steps["clf"]

    probs_real = None
    if hasattr(clf, "predict_proba"):
        try:
            p = clf.predict_proba(X_std)
            if p.ndim == 2 and p.shape[1] == 2:
                probs_real = p[:, 1]   # P(class=1) = P(REAL)
        except Exception:
            pass

    if probs_real is not None:
        labels = (probs_real >= 0.5).astype(int)
    elif hasattr(clf, "decision_function"):
        labels = (clf.decision_function(X_std) >= 0.0).astype(int)
    else:
        labels = clf.predict(X_std).astype(int)

    return labels, probs_real


def _compute_alignment(
    metrics_df: pd.DataFrame,
    detector_df: pd.DataFrame,
    pipe,
    feature_cols: List[str],
) -> Tuple[Optional[dict], Optional[str]]:
    """Compute alignment metrics between frozen surrogate and detector.

    Returns (result_dict, error_str). On error result_dict is None.
    """
    id_col_m = _detect_id_column(metrics_df)
    id_col_d = _detect_id_column(detector_df)
    if id_col_m is None:
        return None, "no ID column in metrics CSV"
    if id_col_d is None:
        return None, "no ID column in detector CSV"

    label_col = _detect_detector_label_column(detector_df)
    if label_col is None:
        return None, "no label column in detector CSV"

    score_col = _detect_detector_score_column(detector_df)

    # NaN-fill any missing feature columns so the frozen imputer can handle them
    metrics_df = metrics_df.copy()
    for f in feature_cols:
        if f not in metrics_df.columns:
            metrics_df[f] = np.nan

    metrics_df["__ID__"] = metrics_df[id_col_m].astype(str)
    detector_df = detector_df.copy()
    detector_df["__ID__"] = detector_df[id_col_d].astype(str)

    det_cols = ["__ID__", label_col] + ([score_col] if score_col else [])
    merged = metrics_df[["__ID__"] + feature_cols].merge(
        detector_df[det_cols], on="__ID__", how="inner"
    )
    n = len(merged)
    if n == 0:
        return None, "no rows after ID intersection"

    # Surrogate predictions
    try:
        surr_labels, surr_probs = _predict_surrogate(pipe, merged, feature_cols)
    except Exception as e:
        return None, f"surrogate prediction failed: {e}"

    # Binarize detector labels
    det_labels = _binarize_labels(merged[label_col])
    if det_labels is None:
        return None, "could not binarize detector labels"

    agreement = float(np.mean(surr_labels == det_labels))
    result: dict = {"n": n, "agreement": round(agreement, 4), "corr": None, "mae": None}

    # Score-based metrics (need both surrogate probs and detector scores)
    if score_col and score_col in merged.columns and surr_probs is not None:
        det_scores_raw = pd.to_numeric(merged[score_col], errors="coerce")
        if det_scores_raw.notna().sum() >= 5:
            det_scores = det_scores_raw.fillna(0.5).values

            # Heuristic: if the detector's score is higher for FAKEs than REALs,
            # the score column is P(FAKE) — invert to get P(REAL) for a fair comparison
            # with surr_probs which is P(REAL).
            real_mask = det_labels == 1
            fake_mask = det_labels == 0
            if real_mask.sum() > 0 and fake_mask.sum() > 0:
                if det_scores[fake_mask].mean() > det_scores[real_mask].mean():
                    det_scores = 1.0 - det_scores

            try:
                corr_val, _ = stats.pearsonr(surr_probs, det_scores)
                result["corr"] = round(float(corr_val), 4)
            except Exception:
                pass
            try:
                result["mae"] = round(float(np.mean(np.abs(surr_probs - det_scores))), 4)
            except Exception:
                pass

    return result, None


def _assess_alignment_status(
    align: dict,
    baseline_align: Optional[dict],
    min_n: int,
) -> Tuple[str, bool, List[str]]:
    """Determine status and stop flag from alignment metrics.

    Returns (status_str, stop_bool, reasons_list).
    """
    n = align.get("n", 0)
    if n < min_n:
        return "INSUFFICIENT_SAMPLE_TAIL", False, [f"n={n} < min_n={min_n}"]

    stop = False
    warning = False
    reasons: List[str] = []

    agreement = align.get("agreement")
    corr = align.get("corr")
    mae = align.get("mae")
    baseline_mae = baseline_align.get("mae") if baseline_align else None

    if agreement is not None:
        if agreement < 0.60:
            stop = True
            reasons.append(f"agreement={agreement:.3f} < 0.60 (STOP)")
        elif agreement < 0.70:
            warning = True
            reasons.append(f"agreement={agreement:.3f} < 0.70 (WARNING)")

    if corr is not None:
        if corr < 0.30:
            stop = True
            reasons.append(f"corr={corr:.3f} < 0.30 (STOP)")
        elif corr < 0.50:
            warning = True
            reasons.append(f"corr={corr:.3f} < 0.50 (WARNING)")

    if mae is not None and baseline_mae is not None:
        delta = mae - baseline_mae
        if delta > 0.25:
            stop = True
            reasons.append(f"mae_delta={delta:.3f} > 0.25 (STOP)")
        elif delta > 0.15:
            warning = True
            reasons.append(f"mae_delta={delta:.3f} > 0.15 (WARNING)")

    if stop:
        return "STOP", True, reasons
    if warning:
        return "WARNING", False, reasons
    return "OK", False, reasons


# ── §11 DDM-style fidelity monitoring ────────────────────────────────────────

# Minimum n_r for meaningful DDM inference. Sourced from pipeline.constants
# (single source of truth, shared with meta_model / run_loop_rounds /
# retraining_buffer).
DEFAULT_MIN_ACTIVE_FAKE = MIN_ACTIVE_FAKE

# KS feature-drift thresholds: classify based on # of features in drift
KS_DRIFT_FEATURE_COUNT_WARNING = 3   # ≥3 features drifted → WARNING
KS_DRIFT_FEATURE_COUNT_STOP = 6      # ≥6 features drifted → STOP (≈46% of 13 total)


def compute_fidelity_ddm(
    active_fake_ids_r: List[str],
    sgl_predictions: Dict[str, str],
    detector_labels: Dict[str, str],
    e_min: float,
    sigma_min: float,
    min_active_fake: int = DEFAULT_MIN_ACTIVE_FAKE,
) -> dict:
    """Batch DDM-style fidelity check (Gama et al., 2004; Zhou, 2004).

    Returns a dict with status, fidelity_r, e_r, sigma_r, e_min (updated),
    sigma_min (updated), n_r, stop, warning.
    """
    n_r = len(active_fake_ids_r)
    if n_r < min_active_fake:
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "n_r": n_r,
            "stop": False,
            "warning": False,
        }

    n_agree = sum(
        1 for i in active_fake_ids_r
        if sgl_predictions.get(str(i)) == detector_labels.get(str(i))
    )
    fidelity_r = n_agree / n_r
    e_r = 1.0 - fidelity_r
    sigma_r = math.sqrt(max(e_r * (1.0 - e_r) / n_r, 1e-9))

    e_min_new, sigma_min_new = e_min, sigma_min
    if e_min <= 0.0 or (e_r + sigma_r < e_min + sigma_min):
        e_min_new = e_r
        sigma_min_new = sigma_r

    warning = (e_r + sigma_r) >= (e_min_new + 2 * sigma_min_new)
    stop = (e_r + sigma_r) >= (e_min_new + 3 * sigma_min_new)

    if stop:
        status = "STOP_FIDELITY_DEGRADED"
    elif warning:
        status = "WARNING_FIDELITY_DEGRADED"
    else:
        status = "OK"

    return {
        "status": status,
        "fidelity_r": round(fidelity_r, 4),
        "e_r": round(e_r, 4),
        "sigma_r": round(sigma_r, 4),
        "e_min": round(e_min_new, 4),
        "sigma_min": round(sigma_min_new, 4),
        "n_r": n_r,
        "stop": bool(stop),
        "warning": bool(warning),
    }


# ── Main public function ──────────────────────────────────────────────────────

def compute_drift_report(
    metrics0_path: str,
    metrics_t_path: str,
    shap0_path: str,
    shap_t_path: str,
    feature_cols: List[str],
    round_idx: int,
    detector: str,
    out_path: str,
    alpha: float = 0.01,
    min_d: float = 0.2,
    min_n: int = 20,
    min_n_alignment: int = MIN_ACTIVE_FAKE,
    resume: bool = True,
    pipe_path: Optional[str] = None,
    det0_path: Optional[str] = None,
    det_t_path: Optional[str] = None,
    e_min: float = 0.0,
    sigma_min: float = 0.0,
    min_active_fake: int = DEFAULT_MIN_ACTIVE_FAKE,
) -> Tuple[bool, dict]:
    """Compute drift report and write to *out_path*.

    Returns (overall_stop, report_dict).

    Feature drift (KS tests) is diagnostic only — never stops the loop.
    Surrogate alignment degradation is the actual stop criterion.

    If *resume* is True and *out_path* already exists in the **new** format
    (has an "overall" key), the cached report is reloaded.  Old-format files
    (no "overall" key) are silently regenerated.
    """
    shap0_path = _resolve_round0_shap_fallback(shap0_path, detector)

    if resume and os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if "overall" in cached:
                # New format uses `action` (OK/WARNING/RETRAIN). Old format only `stop`.
                action = cached["overall"].get("action")
                if action is None:
                    # Old cached report — regenerate to ensure 3-state semantics.
                    raise ValueError("legacy report missing action")
                retrain_required = bool(action == "RETRAIN")
                sa_status = cached.get("surrogate_fidelity", cached.get("surrogate_alignment", {})).get("status", "?")
                fd_status = cached.get("feature_drift", {}).get("status", "?")
                print(
                    f"[DRIFT] Cached round={round_idx} "
                    f"feat_drift={fd_status} surrogate_fidelity={sa_status} "
                    f"action={action} -> {out_path}"
                )
                return retrain_required, cached
        except Exception:
            pass  # regenerate on any parse error

    # ── Section 1: Feature drift (KS tests) — DIAGNOSTIC ONLY ────────────────
    n_current = _count_rows(metrics_t_path)
    feat_tests: List[dict] = []

    if n_current < min_n:
        feat_status = "INSUFFICIENT_SAMPLE_TAIL"
        n_drifted = 0
    else:
        for feat in feature_cols:
            arr0_x = _load_col(metrics0_path, feat, min_n)
            arr_t_x = _load_col(metrics_t_path, feat, min_n)
            arr0_c = _load_col(shap0_path, f"shap_{feat}", min_n)
            arr_t_c = _load_col(shap_t_path, f"shap_{feat}", min_n)

            if any(a is None for a in (arr0_x, arr_t_x, arr0_c, arr_t_c)):
                feat_tests.append({
                    "feature": feat,
                    "feature_dist": None,
                    "shap_dist": None,
                    "drifted": False,
                    "skipped": True,
                    "skip_reason": "insufficient_data",
                })
                continue

            fks = _ks_result(arr0_x, arr_t_x, alpha, min_d)
            sks = _ks_result(arr0_c, arr_t_c, alpha, min_d)
            drifted = fks["drifted"] or sks["drifted"]
            feat_tests.append({
                "feature": feat,
                "feature_dist": fks,
                "shap_dist": sks,
                "drifted": drifted,
                "skipped": False,
            })

        n_drifted = sum(1 for t in feat_tests if t.get("drifted"))
        if n_drifted >= KS_DRIFT_FEATURE_COUNT_STOP:
            feat_status = "STOP"
        elif n_drifted >= KS_DRIFT_FEATURE_COUNT_WARNING:
            feat_status = "WARNING"
        elif n_drifted > 0:
            feat_status = "WARNING"  # any drift → at least WARNING
        else:
            feat_status = "OK"

    feature_drift_section = {
        "status": feat_status,
        "stop": bool(feat_status == "STOP"),
        "n_drifted": int(n_drifted),
        "n_current": n_current,
        "alpha": alpha,
        "min_d": min_d,
        "min_n": min_n,
        "ks_warning_threshold": KS_DRIFT_FEATURE_COUNT_WARNING,
        "ks_stop_threshold": KS_DRIFT_FEATURE_COUNT_STOP,
        "tests": feat_tests,
    }

    # ── Section 2: Surrogate fidelity (DDM-style) ────────────────────────────
    surr_stop = False
    fidelity_section: dict

    if pipe_path is None or det_t_path is None:
        fidelity_section = {
            "status": "UNAVAILABLE",
            "stop": False,
            "reason": "pipe_path or det_t_path not provided",
        }
    else:
        pipe, pipe_err = _load_frozen_surrogate(pipe_path)
        if pipe is None:
            fidelity_section = {
                "status": "UNAVAILABLE",
                "stop": False,
                "reason": pipe_err,
            }
        elif not os.path.exists(metrics_t_path):
            fidelity_section = {
                "status": "UNAVAILABLE",
                "stop": False,
                "reason": f"metrics_t not found: {metrics_t_path}",
            }
        elif not os.path.exists(det_t_path):
            fidelity_section = {
                "status": "UNAVAILABLE",
                "stop": False,
                "reason": f"det_t not found: {det_t_path}",
            }
        else:
            try:
                metrics_t_df = pd.read_csv(metrics_t_path, encoding="utf-8")
                det_t_df = pd.read_csv(det_t_path, encoding="utf-8")
            except Exception as e:
                fidelity_section = {
                    "status": "UNAVAILABLE",
                    "stop": False,
                    "reason": f"failed to read current-round CSVs: {e}",
                }
            else:
                pipe_feat_cols = _load_pipe_feature_cols(pipe_path) or feature_cols
                id_col_m = _detect_id_column(metrics_t_df)
                id_col_d = _detect_id_column(det_t_df)
                label_col = _detect_detector_label_column(det_t_df)

                if id_col_m is None or id_col_d is None or label_col is None:
                    fidelity_section = {
                        "status": "UNAVAILABLE",
                        "stop": False,
                        "reason": "missing ID or label column in CSVs",
                    }
                else:
                    for f in pipe_feat_cols:
                        if f not in metrics_t_df.columns:
                            metrics_t_df[f] = np.nan
                    metrics_t_df["__ID__"] = metrics_t_df[id_col_m].astype(str)
                    det_t_df["__ID__"] = det_t_df[id_col_d].astype(str)

                    merged = metrics_t_df[["__ID__"] + pipe_feat_cols].merge(
                        det_t_df[["__ID__", label_col]], on="__ID__", how="inner"
                    )

                    try:
                        surr_labels, _ = _predict_surrogate(pipe, merged, pipe_feat_cols)
                    except Exception as e:
                        fidelity_section = {
                            "status": "UNAVAILABLE",
                            "stop": False,
                            "reason": f"surrogate prediction failed: {e}",
                        }
                        surr_labels = None

                    if surr_labels is not None:
                        det_labels_raw = _binarize_labels(merged[label_col])
                        if det_labels_raw is None:
                            fidelity_section = {
                                "status": "UNAVAILABLE",
                                "stop": False,
                                "reason": "could not binarize detector labels",
                            }
                        else:
                            # Build dicts: ACTIVE_FAKE = those detector classifies FAKE (label=0)
                            merged = merged.reset_index(drop=True)
                            fake_mask = det_labels_raw == 0
                            active_ids = [
                                str(merged.at[i, "__ID__"])
                                for i in range(len(merged)) if fake_mask[i]
                            ]
                            sgl_pred_dict = {
                                str(merged.at[i, "__ID__"]): ("FAKE" if surr_labels[i] == 0 else "REAL")
                                for i in range(len(merged))
                            }
                            det_label_dict = {
                                str(merged.at[i, "__ID__"]): ("FAKE" if det_labels_raw[i] == 0 else "REAL")
                                for i in range(len(merged))
                            }

                            fidelity_section = compute_fidelity_ddm(
                                active_ids, sgl_pred_dict, det_label_dict,
                                e_min, sigma_min,
                                min_active_fake=min_active_fake,
                            )
                            surr_stop = bool(fidelity_section.get("stop", False))

    # ── Section 3: Overall ────────────────────────────────────────────────────
    # Three-state model:
    #   OK       — KS=OK AND DDM=OK
    #   WARNING  — KS XOR DDM in WARNING/STOP
    #   RETRAIN  — KS=STOP AND DDM=STOP (convergent evidence → trigger retrain)
    # The loop NEVER stops on drift; only max_round / no_FAKE → STOP_HARD upstream.
    ks_status = feat_status
    fid_status = fidelity_section.get("status", "UNAVAILABLE")
    ks_stop = bool(ks_status == "STOP")
    ddm_stop = bool(surr_stop)
    ks_warn = ks_status in ("WARNING", "STOP")
    ddm_warn = fid_status in ("WARNING_FIDELITY_DEGRADED", "STOP_FIDELITY_DEGRADED")

    if ks_stop and ddm_stop:
        overall_action = "RETRAIN"
        overall_reason = (
            f"KS+DDM convergent: feat_drift=STOP (n_drifted={n_drifted}) "
            f"AND surrogate_fidelity={fid_status}"
        )
    elif ks_warn or ddm_warn:
        overall_action = "WARNING"
        overall_reason = (
            f"feat_drift={ks_status} (n_drifted={n_drifted}), "
            f"surrogate_fidelity={fid_status}"
        )
    else:
        overall_action = "OK"
        overall_reason = "OK"

    # `stop` always False — backward-compat field, loop no longer stops on drift.
    overall_stop = False

    report = {
        "round": round_idx,
        "detector": detector,
        "feature_drift": feature_drift_section,
        "surrogate_fidelity": fidelity_section,
        "overall": {
            "action": overall_action,         # NEW: OK | WARNING | RETRAIN
            "stop": overall_stop,             # kept False for backward compat
            "ks_stop": bool(ks_stop),
            "ddm_stop": bool(ddm_stop),
            "reason": overall_reason,
        },
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    retrain_required = bool(overall_action == "RETRAIN")
    print(
        f"[DRIFT] round={round_idx} feat_drift={feat_status} (n_drifted={n_drifted}) "
        f"surrogate_fidelity={fid_status} "
        f"action={overall_action} "
        f"n_current={n_current} -> {out_path}"
    )
    if retrain_required:
        print(f"[DRIFT][RETRAIN] {overall_reason}")
    elif overall_action == "WARNING":
        print(f"[DRIFT][WARN] {overall_reason}")

    # First return value semantics changed: now `retrain_required` (not `overall_stop`).
    return retrain_required, report
