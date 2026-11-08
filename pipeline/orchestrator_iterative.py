import os
import json
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .metrics_runner import METRICS_REGISTRY
from .objectives import get_objective_text

# ============================================================
# Iterative Orchestrator (SYNCED to METRICS_REGISTRY)
# - Uses the SAME feature space produced by metrics_runner.py
# - Selects next unused feature per ID based on SHAP attributions
# - Writes decisions CSV for pipeline (pivot_agent + objective only for agents)
# - Writes audit log to track fallback vs "sensible" decision quality
#
# Global rules enforced:
# - Agents must NEVER see: metric, direction, agent name, reason, shap, beta, detector
# - Only field passed to agents is INSTRUCTIONS == objective (text)
# ============================================================

FEATURE_COLS: List[str] = list(METRICS_REGISTRY.keys())
FEATURE_TO_AGENT: Dict[str, str] = {k: v.agent_id for k, v in METRICS_REGISTRY.items()}

# Human-readable metric names (INTERNAL ONLY; never shown to agents)
FEATURE_TO_METRIC_NAME: Dict[str, str] = {
    # SA (4)
    "SA_semantic_coverage": "Semantic Coverage",
    "SA_text_compression_ratio": "Text Compression Ratio",
    "SA_numeric_coverage": "Numeric Coverage",
    "SA_numeric_compression_ratio": "Numeric Compression Ratio",
    # SSE (3)
    "SSE_semantic_drift": "Semantic Drift",
    "SSE_lexical_stability": "Lexical Stability",
    "SSE_fluency_ppl": "Fluency (Perplexity)",
    # NM (2)
    "NM_change_magnitude": "Change Magnitude",
    "NM_edit_success_rate": "Edit Success Rate",
    # NaM (2)
    "NaM_rewrite_distance": "Global Rewrite Distance",
    "NaM_contradiction_prob": "Contradiction Probability",
    # TE (2)
    "TE_title_entailment": "Title Entailment",
    "TE_title_contradiction_prob": "Title Contradiction Probability",
    # TE style vector (5)
    "TE_style_titlecase": "Title Case Ratio",
    "TE_style_allcaps": "All-Caps Rate",
    "TE_style_exclquest": "Exclamation/Question Rate",
    "TE_style_title_len": "Title Length",
    "TE_style_nnp": "Proper-Noun Rate",
    # NaM style vector (5)
    "NaM_style_fk_grade": "Reading Grade Level",
    "NaM_style_ttr": "Lexical Diversity",
    "NaM_style_quote_rate": "Quotation Rate",
    "NaM_style_punct_rate": "Punctuation Rate",
    "NaM_style_adjadv": "Adjective/Adverb Density",
}

# Subtask identifiers (tracking)
FEATURE_TO_SUBTASK: Dict[str, str] = {
    # SA
    "SA_semantic_coverage": "SA_REFRESH_SENTENCE_COVERAGE",
    "SA_text_compression_ratio": "SA_REFRESH_SENTENCE_COMPRESSION",
    "SA_numeric_coverage": "SA_REFRESH_NUMBER_COVERAGE",
    "SA_numeric_compression_ratio": "SA_REFRESH_NUMBER_COMPRESSION",
    # SSE
    "SSE_semantic_drift": "SSE_EDIT_SEMANTIC_DRIFT",
    "SSE_lexical_stability": "SSE_EDIT_LEXICAL_STABILITY",
    "SSE_fluency_ppl": "SSE_EDIT_FLUENCY",
    # NM
    "NM_edit_success_rate": "NM_IMPROVE_EDIT_SUCCESS",
    "NM_change_magnitude": "NM_EDIT_MAGNITUDE",
    # NaM
    "NaM_rewrite_distance": "NaM_CONTROL_REWRITE_DISTANCE",
    "NaM_contradiction_prob": "NaM_REDUCE_CONTRADICTIONS",
    # TE
    "TE_title_entailment": "TE_IMPROVE_ENTAILMENT",
    "TE_title_contradiction_prob": "TE_REDUCE_CONTRADICTION",
    # TE style vector
    "TE_style_titlecase": "TE_STYLE_TITLECASE",
    "TE_style_allcaps": "TE_STYLE_ALLCAPS",
    "TE_style_exclquest": "TE_STYLE_EXCLQUEST",
    "TE_style_title_len": "TE_STYLE_TITLE_LEN",
    "TE_style_nnp": "TE_STYLE_NNP",
    # NaM style vector
    "NaM_style_fk_grade": "NaM_STYLE_READING_GRADE",
    "NaM_style_ttr": "NaM_STYLE_LEXICAL_DIVERSITY",
    "NaM_style_quote_rate": "NaM_STYLE_QUOTE_RATE",
    "NaM_style_punct_rate": "NaM_STYLE_PUNCT_RATE",
    "NaM_style_adjadv": "NaM_STYLE_ADJADV",
}

# ============================================================
# Helpers
# ============================================================

def _safe_float_or_none(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None

def _load_trajectory(trajectory_path: Optional[str]) -> dict:
    if not trajectory_path or not os.path.exists(trajectory_path):
        return {}
    try:
        with open(trajectory_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[ORCH][TRAJ] WARN: could not load trajectory file {trajectory_path}: {e}")
        return {}

def _derive_trajectory_feedback(traj: dict, article_id: str, feature: str, no_effect_epsilon: float = 1e-6) -> dict:
    by_id = traj.get("by_id", {}) if isinstance(traj, dict) else {}
    rec_id = by_id.get(str(article_id), {}) if isinstance(by_id, dict) else {}
    features = rec_id.get("features", {}) if isinstance(rec_id, dict) else {}
    rec = features.get(str(feature), {}) if isinstance(features, dict) else {}
    rounds = rec.get("rounds", {}) if isinstance(rec, dict) else {}

    pairs: List[Tuple[int, float]] = []
    if isinstance(rounds, dict):
        for k, v in rounds.items():
            try:
                rk = int(k)
            except Exception:
                continue
            fv = _safe_float_or_none(v)
            if fv is not None:
                pairs.append((rk, fv))
    pairs = sorted(pairs, key=lambda x: x[0])

    current_value = pairs[-1][1] if pairs else None
    previous_value = pairs[-2][1] if len(pairs) >= 2 else None
    round0_value = next((v for r, v in pairs if r == 0), pairs[0][1] if pairs else None)
    baseline_median = _safe_float_or_none(rec.get("baseline_median")) if isinstance(rec, dict) else None
    previous_delta = (current_value - previous_value) if current_value is not None and previous_value is not None else None
    cumulative_delta = (current_value - round0_value) if current_value is not None and round0_value is not None else None

    return {
        "feature": str(feature),
        "current_value": current_value,
        "baseline_median": baseline_median,
        "previous_delta": previous_delta,
        "cumulative_delta": cumulative_delta,
        "no_effect_flag": bool(previous_delta is not None and abs(previous_delta) <= float(no_effect_epsilon)),
        "rounds_observed": [int(r) for r, _ in pairs],
    }

def _json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _ensure_dir(p: str):
    # Idempotente e tollerante ai FileExistsError spuri di makedirs(exist_ok=True)
    # sul filesystem parallelo Lustre sotto carico concorrente.
    try:
        os.makedirs(p, exist_ok=True)
    except FileExistsError:
        if not os.path.isdir(p):
            raise

def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _direction_from_beta(beta: float) -> str:
    return "INCREASE" if beta > 0 else "DECREASE"

def _load_coefficients(coeff_path: str) -> Dict[str, float]:
    df = pd.read_csv(coeff_path, encoding="utf-8")
    if "feature" not in df.columns or "beta" not in df.columns:
        raise ValueError(f"meta_model_coefficients.csv must have columns: feature, beta. Found: {list(df.columns)}")
    df["feature"] = df["feature"].astype(str)
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    return dict(zip(df["feature"], df["beta"]))

def _load_feature_meta(meta_path: Optional[str]) -> dict:
    if not meta_path or not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_beta_stability(stability_path: Optional[str]) -> Dict[str, bool]:
    """Load beta_stability.csv and return {feature: reliable} dict.
    Returns empty dict (all features trusted) if file is missing."""
    if not stability_path or not os.path.exists(stability_path):
        return {}
    try:
        df = pd.read_csv(stability_path, encoding="utf-8")
        if "feature" not in df.columns or "reliable" not in df.columns:
            return {}
        df["feature"] = df["feature"].astype(str)
        return {str(r["feature"]): bool(r["reliable"]) for _, r in df.iterrows()}
    except Exception:
        return {}

def _rank_features_for_id(shap_vals: Dict[str, float],
                           reliability: Optional[Dict[str, bool]] = None,
                           eligible: Optional[Set[str]] = None) -> List[str]:
    """Rank features by SHAP sign then magnitude. Features with SHAP < 0
    come first (sorted by |SHAP| descending), then features with SHAP >= 0
    (sorted by |SHAP| descending). Negative SHAP on a FAKE-classified
    article means the feature is actively pushing the prediction away from
    REAL and has the most room for improvement. Unreliable features
    (reliability=False) are zeroed before ranking and effectively pushed
    to the end."""
    effective: Dict[str, float] = {}
    items = shap_vals.items() if eligible is None else [(k, v) for k, v in shap_vals.items() if k in eligible]
    for k, v in items:
        if reliability and not reliability.get(k, True):
            effective[k] = 0.0   # unreliable beta → never selected
        else:
            effective[k] = v
    neg = [(k, v) for k, v in effective.items() if v < 0]
    pos = [(k, v) for k, v in effective.items() if v >= 0]
    neg_sorted = sorted(neg, key=lambda kv: abs(kv[1]), reverse=True)
    pos_sorted = sorted(pos, key=lambda kv: abs(kv[1]), reverse=True)
    return [k for (k, _) in neg_sorted] + [k for (k, _) in pos_sorted]

def _select_next_feature(shap_vals: Dict[str, float],
                         reliability: Optional[Dict[str, bool]] = None,
                         eligible: Optional[Set[str]] = None) -> Optional[Tuple[str, float]]:
    for f in _rank_features_for_id(shap_vals, reliability, eligible=eligible):
        if f in FEATURE_COLS:
            return f, float(shap_vals.get(f, 0.0))
    return None


def _select_least_feature(shap_vals: Dict[str, float],
                          eligible: Optional[Set[str]] = None) -> Optional[Tuple[str, float]]:
    """ANTI-FEATURE ablation (exp3): pick the LEAST important eligible feature
    (smallest |SHAP|) instead of the most important. `eligible` already excludes
    unreliable / invalid-beta features upstream, so any of these is a legitimate
    (just minimally-attributed) choice. Direction is left untouched (normal beta sign)."""
    pool = eligible if eligible is not None else set(shap_vals.keys())
    cands = [(f, abs(float(shap_vals.get(f, 0.0)))) for f in pool if f in FEATURE_COLS]
    if not cands:
        return None
    f = min(cands, key=lambda kv: kv[1])[0]
    return f, float(shap_vals.get(f, 0.0))

def _run_flags(active: str) -> Dict[str, int]:
    order = ["SA", "SSE", "NM", "NaM", "TE"]
    flags = {f"run_{a}": 0 for a in order}
    if active not in order:
        return flags
    for a in order[order.index(active):]:
        flags[f"run_{a}"] = 1
    return flags

def _state_flags(active: str) -> Dict[str, str | int]:
    order = ["SA", "SSE", "NM", "NaM", "TE"]
    states = {f"state_{a}": "KEEP" for a in order}
    if active in order:
        states[f"state_{active}"] = "MODIFY"
    return {**states, **_run_flags(active)}

def _audit_row(
    _id: str,
    pivot_agent: str,
    chosen_feature: str,
    chosen_subtask: str,
    fallback_used: bool,
    fallback_reason: str,
    objective: str,
    shap_path: str,
    coeffs_path: str,
) -> Dict[str, object]:
    return {
        "timestamp": _now_iso(),
        "ID": str(_id),
        "pivot_agent": str(pivot_agent),
        "chosen_feature": str(chosen_feature),
        "chosen_subtask": str(chosen_subtask),
        "fallback_used": bool(fallback_used),
        "fallback_reason": str(fallback_reason or ""),
        "objective_len": int(len(objective or "")),
        "objective_empty": bool(not (objective and str(objective).strip())),
        "shap_path": str(shap_path),
        "coeffs_path": str(coeffs_path),
    }

def _load_agent_snapshot_orch(
    snaps_base_dir: Optional[str],
    agent: str,
    article_id: str,
    search_before_round: int,
) -> Optional[dict]:
    """Search backward from search_before_round-1 to 0 for the latest agent snapshot."""
    if not snaps_base_dir:
        return None
    for r in range(search_before_round - 1, -1, -1):
        path = os.path.join(
            snaps_base_dir, f"round_{r:03d}", f"after_{agent.lower()}_{article_id}.json"
        )
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as _f:
                    return json.load(_f)
            except Exception:
                continue
    return None


def _build_previous_attempt(snapshot: Optional[dict], pivot_agent: str) -> Optional[dict]:
    """Build previous_attempt dict from snapshot for NaM pivot only.

    2026-05-20 revision — SSE and NM receive no `previous_attempt` snapshot:
    SSE is anchored to the round-0 REAL sentence (the metric's own anchor) via
    cosine match, and NM carries a fixed `target_number` (round-0 SA value)
    plus `current_number` = lineage[-1] (last successful NM edit on this
    target). Both pivots therefore return None here.
    """
    if not snapshot or not isinstance(snapshot.get("edits"), list):
        return None
    edits = snapshot["edits"]
    if pivot_agent in ("SSE", "NM"):
        return None
    if pivot_agent == "NaM":
        first_edit = next((e for e in edits if e.get("modified_article")), None)
        if first_edit:
            return {
                "my_last_modified_article": first_edit.get("modified_article", ""),
            }
    return None


def _find_output_root(round_dir: str) -> Optional[str]:
    """
    Robustly find the project Output/ folder by walking upwards.
    Returns the absolute path to the folder named 'Output' if found.
    """
    cur = os.path.abspath(round_dir)
    for _ in range(0, 8):
        base = os.path.basename(cur)
        if base.lower() == "output":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


# ============================================================
# MAIN
# ============================================================
def run_orchestrator_round_iterative(
    detector_name: str,
    round_dir: str,
    detector_csv_prefix: str = "detector_",
    filter_fake: bool = True,

    # SHAP:
    shap_dir: Optional[str] = None,
    shap_filename: str = "shap_attributions_per_id.csv",

    # Frozen baseline coefficients (preferred).
    baseline_coeffs_path: Optional[str] = None,

    trajectory_path: Optional[str] = None,
    agent_snapshots_base_dir: Optional[str] = None,

    # --- Framework-validity ablations (experiments 2 & 3) ---
    # invert_direction (exp2): keep the SAME selected feature but flip the
    #   edit direction handed to the agent (INCREASE<->DECREASE). Counterfactual
    #   to test whether SHAP-guided direction is causal.
    # anti_feature (exp3): select the LEAST important eligible feature
    #   (smallest |SHAP|) instead of the most important; direction stays normal.
    invert_direction: bool = False,
    anti_feature: bool = False,
):
    """
    Iterative orchestration:
      - reads SHAP per-ID attributions
      - reads frozen baseline betas (direction from beta sign)
      - optionally filters IDs to those predicted FAKE for this round
      - picks next unused feature (prefer negative SHAP)
      - writes decisions to:
          <round_dir>/orchestrator/<detector>/orch_decisions_with_reasons.csv
          <round_dir>/orchestrator/<detector>/orch_guides.jsonl   (objective-only)
          <round_dir>/orchestrator/<detector>/orch_audit_log.csv  (fallback logging)
    """

    if invert_direction or anti_feature:
        print(f"[ORCH][ABLATION] detector={detector_name} "
              f"invert_direction={invert_direction} anti_feature={anti_feature}")

    # ---------- SHAP path (FIX: robust fallbacks) ----------
    shap_candidates: List[str] = []

    # 1) user-provided shap_dir (if any)
    if shap_dir is not None:
        shap_candidates.append(os.path.join(shap_dir, shap_filename))

    # 2) common locations inside round_dir
    shap_candidates.extend([
        os.path.join(round_dir, "shap", detector_name, shap_filename),
        os.path.join(round_dir, "meta", detector_name, shap_filename),
        os.path.join(round_dir, "meta", detector_name, "shap_attributions_per_id.csv"),
        os.path.join(round_dir, "meta_frozen", detector_name, shap_filename),
    ])

    # 3) if round_dir is nested under Output/, try Output-level meta_frozen
    out_root = _find_output_root(round_dir)
    if out_root is not None:
        shap_candidates.extend([
            os.path.join(out_root, "meta_frozen", detector_name, shap_filename),
            os.path.join(out_root, "meta_frozen", detector_name, "shap_attributions_per_id.csv"),
        ])

    shap_path = None
    for p in shap_candidates:
        if p and os.path.exists(p):
            shap_path = p
            break

    if shap_path is None:
        raise FileNotFoundError(
            "Missing SHAP per-ID attributions. Tried:\n- " + "\n- ".join(shap_candidates)
        )

    # ---------- COEFFS path (frozen baseline) ----------
    if baseline_coeffs_path is None:
        out_root = _find_output_root(round_dir)
        if out_root is not None:
            cand = os.path.join(out_root, "meta_frozen", detector_name, "meta_model_coefficients.csv")
            if os.path.exists(cand):
                baseline_coeffs_path = cand

        if baseline_coeffs_path is None:
            # legacy fallback location (still within Output/)
            baseline_coeffs_path = os.path.join("Output", detector_name, "Baseline", "meta_model_coefficients.csv")

    if not os.path.exists(baseline_coeffs_path):
        raise FileNotFoundError(f"Missing frozen baseline meta-model coefficients: {baseline_coeffs_path}")

    # ---------- output dirs ----------
    orch_dir_new = os.path.join(round_dir, "orchestrator", detector_name)
    _ensure_dir(orch_dir_new)

    out_csv_new = os.path.join(orch_dir_new, "orch_decisions_with_reasons.csv")
    guides_path_new = os.path.join(orch_dir_new, "orch_guides.jsonl")
    audit_path_new = os.path.join(orch_dir_new, "orch_audit_log.csv")

    # --- Load SHAP per-ID ---
    df_shap = pd.read_csv(shap_path, encoding="utf-8")
    if "ID" not in df_shap.columns:
        raise ValueError(f"SHAP file missing 'ID' column: {shap_path}")
    df_shap["ID"] = df_shap["ID"].astype(str)

    # --- Load detector output (for filtering) ---
    det_candidates = [
        os.path.join(round_dir, "generation", f"{detector_csv_prefix}{detector_name}.csv"),
        os.path.join(round_dir, f"{detector_csv_prefix}{detector_name}.csv"),
        os.path.join(round_dir, "detectors", detector_name, f"{detector_csv_prefix}{detector_name}.csv"),
    ]
    det_path = None
    for p in det_candidates:
        if os.path.exists(p):
            det_path = p
            break
    if det_path is None:
        raise FileNotFoundError(f"Missing detector output. Tried: {det_candidates}")

    df_det = pd.read_csv(det_path, encoding="utf-8")
    if not {"ID", "pred_label", "confidence"}.issubset(df_det.columns):
        raise ValueError(f"Detector CSV must contain: ID, pred_label, confidence. Found: {list(df_det.columns)}")
    df_det["ID"] = df_det["ID"].astype(str)
    df_det["pred_label"] = df_det["pred_label"].astype(str).str.upper()
    df_det["confidence"] = pd.to_numeric(df_det["confidence"], errors="coerce")

    df = df_shap.merge(df_det[["ID", "pred_label", "confidence"]], on="ID", how="inner")
    if filter_fake:
        df = df[df["pred_label"] == "FAKE"].copy()

    # --- Load betas ---
    betas = _load_coefficients(baseline_coeffs_path)

    # --- Load beta stability + selected feature set from frozen baseline ---
    meta_path = os.path.join(os.path.dirname(baseline_coeffs_path), "feature_meta.json")
    feature_meta = _load_feature_meta(meta_path)
    eligible_features = set(feature_meta.get("feature_cols", []) or [])

    _meta_model_type = feature_meta.get("meta_model_type", "sparse_group_lasso")
    _BLOCK_MODEL_TYPES = {"ridge_fallback", "insufficient_fake"}
    if _meta_model_type in _BLOCK_MODEL_TYPES:
        print(f"[ORCH][WARN] Detector '{detector_name}' meta_model_type='{_meta_model_type}': "
              f"SHAP attributions are not interpretable. Skipping orchestration.")
        open(guides_path_new, "w", encoding="utf-8").close()
        df_out = pd.DataFrame(columns=[
            "ID", "chosen_feature", "chosen_subtask", "pivot_agent", "objective",
            "agent", "metric", "direction", "chosen_shap",
            "trajectory_current_value", "trajectory_baseline_median",
            "trajectory_previous_delta", "trajectory_cumulative_delta",
            "trajectory_no_effect_flag", "trajectory_feedback_json",
            "previous_attempt_json",
            "state_SA", "state_SSE", "state_NM", "state_NaM", "state_TE",
            "run_SA", "run_SSE", "run_NM", "run_NaM", "run_TE", "guide_en",
        ])
        df_out.to_csv(out_csv_new, index=False, encoding="utf-8")
        pd.DataFrame(columns=[
            "timestamp", "ID", "pivot_agent", "chosen_feature", "chosen_subtask",
            "fallback_used", "fallback_reason", "objective_len", "objective_empty",
            "shap_path", "coeffs_path",
        ]).to_csv(audit_path_new, index=False, encoding="utf-8")
        return df_out

    stability_path = os.path.join(
        os.path.dirname(baseline_coeffs_path), "beta_stability.csv"
    )
    reliability = _load_beta_stability(stability_path)
    if reliability:
        n_unreliable = sum(1 for v in reliability.values() if not v)
        reliable_set = {k for k, v in reliability.items() if v}
        eligible_features = (eligible_features & reliable_set) if eligible_features else reliable_set
        print(f"[ORCH] Beta stability loaded: {n_unreliable}/{len(reliability)} "
              f"features marked unreliable (will be skipped in ranking).")
    else:
        print("[ORCH] Beta stability file not found — all features treated as reliable.")

    if not eligible_features:
        eligible_features = set(FEATURE_COLS)

    features_with_valid_beta = {
        f for f, b in betas.items()
        if b is not None
        and not (isinstance(b, float) and np.isnan(b))
        and abs(float(b)) > 1e-10
    }
    if features_with_valid_beta:
        eligible_features = eligible_features & features_with_valid_beta
    if not eligible_features:
        print("[WARN] No eligible features with valid non-zero beta — "
              "falling back to reliable set without beta filter")
        _reliable_set_safe = {k for k, v in reliability.items() if v} if reliability else set()
        eligible_features = _reliable_set_safe if _reliable_set_safe else set(FEATURE_COLS)

    # --- Trajectory feedback (agent-facing, selected feature only) ---
    trajectory = _load_trajectory(trajectory_path)
    if trajectory_path:
        if trajectory:
            print(f"[ORCH][TRAJ] Loaded trajectory feedback: {trajectory_path}")
        else:
            print(f"[ORCH][TRAJ] Trajectory feedback unavailable or empty: {trajectory_path}")

    # --- Channel-availability index (2026-05-20 channel-split refactor) ---
    # SSE/NM pivot can only operate if the corresponding round-0 channel
    # has anchors. We pre-load baseline_references.json once and use it
    # below to soft-blacklist features whose channel is empty for an ID.
    _channel_refs: Dict[str, Dict[str, Any]] = {}
    try:
        _br_path = os.path.join(
            os.path.dirname(os.path.dirname(baseline_coeffs_path)),
            "baseline_references.json",
        )
        if os.path.exists(_br_path):
            with open(_br_path, "r", encoding="utf-8") as _f_br:
                _channel_refs = json.load(_f_br) or {}
    except Exception as _e_br:
        print(f"[ORCH][WARN] failed to load baseline_references.json: {_e_br}")
        _channel_refs = {}

    _SSE_FEATS: Set[str] = {f for f, a in FEATURE_TO_AGENT.items() if a == "SSE"}
    _NM_FEATS:  Set[str] = {f for f, a in FEATURE_TO_AGENT.items() if a == "NM"}

    def _empty_channel_features_for(article_id: str) -> Tuple[Set[str], List[str]]:
        rec = _channel_refs.get(str(article_id), {}) if isinstance(_channel_refs, dict) else {}
        sse_t0 = rec.get("SSE_TARGETS_0", []) if isinstance(rec, dict) else []
        kn0 = rec.get("KN_0", []) if isinstance(rec, dict) else []
        empties: Set[str] = set()
        reasons: List[str] = []
        if not sse_t0:
            empties |= _SSE_FEATS
            reasons.append("SSE_TARGETS_0_empty")
        if not kn0:
            empties |= _NM_FEATS
            reasons.append("KN_0_empty")
        return empties, reasons

    # Infer round index from round_dir for backward snapshot search
    _m_ri = re.search(r"round[_\-]?(\d+)", os.path.abspath(round_dir), re.IGNORECASE)
    _round_idx_orch: int = int(_m_ri.group(1)) if _m_ri else 0

    rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []

    with open(guides_path_new, "w", encoding="utf-8") as gf_new:
        for _, r in df.iterrows():
            _id = str(r["ID"])

            shap_vals: Dict[str, float] = {}
            for feat in FEATURE_COLS:
                col = f"shap_{feat}"
                v = r.get(col, 0.0)
                try:
                    shap_vals[feat] = float(v)
                except Exception:
                    shap_vals[feat] = 0.0

            # §2.5 — soft-blacklist stagnant features for THIS round only.
            # If a feature's previous_delta was ~0, picking it again before the
            # article state changes will produce the same result; defer to next-best.
            _stagnant_features: Set[str] = set()
            for _f in FEATURE_COLS:
                if _derive_trajectory_feedback(trajectory, _id, _f).get("no_effect_flag"):
                    _stagnant_features.add(_f)
            # 2026-05-20 channel-split: exclude SSE/NM features whose round-0
            # channel is empty for this ID (e.g. article has no key_numbers ->
            # NM features have nothing to anchor on).
            _empty_chan_feats, _empty_chan_reasons = _empty_channel_features_for(_id)
            _stagnant_features |= _empty_chan_feats
            _eligible_for_pick = eligible_features - _stagnant_features
            if not _eligible_for_pick:
                _eligible_for_pick = eligible_features
            if anti_feature:
                # exp3: pick the LEAST important eligible feature instead of top-|SHAP|
                pick = _select_least_feature(shap_vals, eligible=_eligible_for_pick)
            else:
                pick = _select_next_feature(shap_vals, reliability, eligible=_eligible_for_pick)

            fallback_used = False
            fallback_reason = ""
            if pick is None:
                candidates = [f for f in FEATURE_COLS if f in eligible_features]
                if candidates:
                    feat = candidates[0]
                    shap_val = float(shap_vals.get(feat, 0.0))
                    pick = (feat, shap_val)
                    fallback_used = True
                    fallback_reason = "ranker_no_pick_fallback_first_unused"
                else:
                    audit_rows.append(_audit_row(
                        _id=_id,
                        pivot_agent="NONE",
                        chosen_feature="",
                        chosen_subtask="",
                        fallback_used=True,
                        fallback_reason="no_eligible_features",
                        objective="",
                        shap_path=shap_path,
                        coeffs_path=baseline_coeffs_path,
                    ))
                    continue

            feat, shap_val = pick
            pivot_agent = FEATURE_TO_AGENT.get(feat, "UNKNOWN")
            metric_name = FEATURE_TO_METRIC_NAME.get(feat, feat)
            chosen_subtask = FEATURE_TO_SUBTASK.get(feat, "")

            _raw_beta = betas.get(feat, None)
            if _raw_beta is None or (isinstance(_raw_beta, float) and np.isnan(_raw_beta)):
                print(f"[WARN] feature '{feat}' selected but has no valid beta in "
                      f"coefficients — eligible_features filter should have prevented "
                      f"this. Using 0.0 as fallback.")
                _raw_beta = 0.0
            beta = float(_raw_beta)
            direction = _direction_from_beta(beta)
            if invert_direction:
                # exp2: flip the direction handed to the agent. This single flip
                # propagates to the objective text AND to the deterministic title
                # actuators downstream (they read the 'direction' CSV column).
                direction = "DECREASE" if direction == "INCREASE" else "INCREASE"

            objective = get_objective_text(pivot_agent, metric_name, direction)
            trajectory_feedback = _derive_trajectory_feedback(trajectory, _id, feat)
            trajectory_feedback_json = _json_dumps_compact(trajectory_feedback)
            if not objective:
                print(f"[WARN][ORCH] No objective text found for agent={pivot_agent} metric={metric_name} direction={direction} — using generic fallback")
                fallback_used = True
                fallback_reason = (fallback_reason + "|objective_missing").strip("|")
                objective = "Revise the assigned content according to the instructions, preserving coherence, plausibility, and required output format."

            flags = _state_flags(pivot_agent)

            # §5 — load previous_attempt from agent snapshot for SSE/NM/NaM pivot
            previous_attempt: Optional[dict] = None
            if pivot_agent in ("SSE", "NM", "NaM") and agent_snapshots_base_dir:
                _prev_snap = _load_agent_snapshot_orch(
                    agent_snapshots_base_dir, pivot_agent, _id, _round_idx_orch
                )
                previous_attempt = _build_previous_attempt(_prev_snap, pivot_agent)
            previous_attempt_json: str = (
                json.dumps(previous_attempt, ensure_ascii=False) if previous_attempt else ""
            )

            row = {
                "ID": _id,
                "chosen_feature": feat,
                "chosen_subtask": chosen_subtask,

                # preferred columns (pipeline reads these)
                "pivot_agent": pivot_agent,
                "objective": objective,

                # legacy/debug columns (NOT used in prompts)
                "agent": pivot_agent,
                "metric": metric_name,
                "direction": direction,
                "chosen_shap": float(shap_val),
                "trajectory_current_value": trajectory_feedback.get("current_value"),
                "trajectory_baseline_median": trajectory_feedback.get("baseline_median"),
                "trajectory_previous_delta": trajectory_feedback.get("previous_delta"),
                "trajectory_cumulative_delta": trajectory_feedback.get("cumulative_delta"),
                "trajectory_no_effect_flag": bool(trajectory_feedback.get("no_effect_flag", False)),
                "trajectory_feedback_json": trajectory_feedback_json,
                "previous_attempt_json": previous_attempt_json,
                **flags,

                # keep as objective-only (safe)
                "guide_en": objective,
            }
            rows.append(row)

            payload = {
                "ID": _id,
                "objective": objective,
                "pivot_agent": pivot_agent,
                "trajectory_feedback": trajectory_feedback,
            }
            if previous_attempt is not None:
                payload["previous_attempt"] = previous_attempt
            gf_new.write(json.dumps(payload, ensure_ascii=False) + "\n")

            audit_rows.append(_audit_row(
                _id=_id,
                pivot_agent=pivot_agent,
                chosen_feature=feat,
                chosen_subtask=chosen_subtask,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                objective=objective,
                shap_path=shap_path,
                coeffs_path=baseline_coeffs_path,
            ))

    if rows:
        df_out = pd.DataFrame(rows)
    else:
        df_out = pd.DataFrame(columns=[
            "ID", "chosen_feature", "chosen_subtask", "pivot_agent", "objective",
            "agent", "metric", "direction", "chosen_shap",
            "trajectory_current_value", "trajectory_baseline_median",
            "trajectory_previous_delta", "trajectory_cumulative_delta",
            "trajectory_no_effect_flag", "trajectory_feedback_json",
            "previous_attempt_json",
            "state_SA", "state_SSE", "state_NM", "state_NaM", "state_TE",
            "run_SA", "run_SSE", "run_NM", "run_NaM", "run_TE", "guide_en",
        ])
    df_audit = pd.DataFrame(audit_rows)

    df_out.to_csv(out_csv_new, index=False, encoding="utf-8")
    df_audit.to_csv(audit_path_new, index=False, encoding="utf-8")

    print(f"[ORCH-ITER] Done for detector={detector_name}.")
    print(f"  - SHAP read from:     {shap_path}")
    print(f"  - COEFFS read from:   {baseline_coeffs_path}")
    print(f"  - Detector read from: {det_path}")
    print(f"  - Decisions:          {len(df_out)}")
    print(f"  - Out CSV (new):      {out_csv_new}")
    print(f"  - Audit (new):        {audit_path_new}")
    if trajectory_path:
        print(f"  - Trajectory read:    {trajectory_path}")
    print(f"  - Guides (new):       {guides_path_new}")

    return df_out


class IterativeOrchestrator:
    """Class wrapper for the iterative orchestrator (round > 0).

    The orchestration logic is unchanged; this wrapper exists to align the
    pipeline package with the class-based style used elsewhere.
    """

    def __init__(
        self,
        detector_name: str,
        round_dir: str,
        detector_csv_prefix: str = "detector_",
        filter_fake: bool = True,
        shap_dir: str | None = None,
        shap_filename: str = "shap_attributions_per_id.csv",
        baseline_coeffs_path: str | None = None,
    ):
        self.detector_name = detector_name
        self.round_dir = round_dir
        self.detector_csv_prefix = detector_csv_prefix
        self.filter_fake = filter_fake
        self.shap_dir = shap_dir
        self.shap_filename = shap_filename
        self.baseline_coeffs_path = baseline_coeffs_path

    def run(self):
        return run_orchestrator_round_iterative(
            detector_name=self.detector_name,
            round_dir=self.round_dir,
            detector_csv_prefix=self.detector_csv_prefix,
            filter_fake=self.filter_fake,
            shap_dir=self.shap_dir,
            shap_filename=self.shap_filename,
            baseline_coeffs_path=self.baseline_coeffs_path,
        )