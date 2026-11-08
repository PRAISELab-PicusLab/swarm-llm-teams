import os
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error

import shap
import joblib

from .metrics_runner import METRICS_REGISTRY
from .sparse_group_lasso import (
    SparseGroupLassoLogistic,
    SparseGroupLassoCV,
    build_group_indices,
)
from .constants import MIN_ACTIVE_FAKE

FEATURE_COLS_DEFAULT = list(METRICS_REGISTRY.keys())
FEATURE_TO_AGENT = {k: v.agent_id for k, v in METRICS_REGISTRY.items()}

DEFAULT_LAMS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

SA_NUMERIC_FEATURES = ["SA_numeric_coverage", "SA_numeric_compression_ratio"]
SA_APPLICABILITY_FLAG = "SA_has_numbers"


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _write_json(path: str, payload: dict) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _resolve_meta_input_paths(round_dir: str, detector_name: str, metrics_csv_name: str, detector_csv_prefix: str) -> tuple[str, str]:
    """Resolve the canonical paths for the metrics and detector CSVs.

    Canonical layout (single source of truth, no duplicates):
      - metrics: <round_root>/generation/<detector>/<metrics_csv_name>   (round > 0)
                 <round_root>/generation/<metrics_csv_name>              (round 0, shared)
      - detector: <round_root>/detectors/<detector>/<prefix><detector>.csv

    The caller can pass either the round root or a generation subdir as round_dir;
    this function normalizes both cases.
    """
    rd = os.path.abspath(round_dir)
    base = os.path.basename(rd).lower()
    parent = os.path.basename(os.path.dirname(rd)).lower()

    if parent == "generation":
        round_root = os.path.dirname(os.path.dirname(rd))
        gen_dir = rd
    elif base == "generation":
        round_root = os.path.dirname(rd)
        gen_dir = os.path.join(rd, detector_name)
    else:
        round_root = rd
        gen_dir = os.path.join(rd, "generation", detector_name)

    metrics_csv = os.path.join(gen_dir, metrics_csv_name)
    if not os.path.exists(metrics_csv):
        # Round 0: metrics live at generation/<name> (shared across detectors)
        metrics_csv_shared = os.path.join(round_root, "generation", metrics_csv_name)
        if os.path.exists(metrics_csv_shared):
            metrics_csv = metrics_csv_shared

    detector_csv = os.path.join(round_root, "detectors", detector_name, f"{detector_csv_prefix}{detector_name}.csv")

    return metrics_csv, detector_csv


def build_meta_join_dataframe(
    detector_name: str,
    round_dir: str,
    feature_cols=None,
    metrics_csv_name: str = "agent_metrics.csv",
    detector_csv_prefix: str = "detector_",
    strict_feature_check: bool = True,
):
    feature_cols = FEATURE_COLS_DEFAULT if feature_cols is None else list(feature_cols)
    metrics_csv, detector_csv = _resolve_meta_input_paths(round_dir, detector_name, metrics_csv_name, detector_csv_prefix)

    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Metriche non trovate: {metrics_csv}")
    if not os.path.exists(detector_csv):
        raise FileNotFoundError(f"CSV detector non trovato: {detector_csv}")

    dfm = pd.read_csv(metrics_csv, encoding="utf-8")
    if "ID" not in dfm.columns:
        raise ValueError(f"Metrics CSV missing 'ID': {metrics_csv}")
    dfm["ID"] = dfm["ID"].astype(str)

    needed = ["ID"] + list(feature_cols)
    missing = [c for c in needed if c not in dfm.columns]
    if missing and strict_feature_check:
        raise ValueError(f"Mancano colonne nelle metriche: {missing}")
    elif missing:
        print(f"[WARN] Colonne metriche mancanti (create NaN): {missing}")
        for c in missing:
            if c != "ID":
                dfm[c] = np.nan

    for c in feature_cols:
        if c != "ID":
            dfm[c] = pd.to_numeric(dfm[c], errors="coerce")

    _neutralize_sa_numeric_when_not_applicable(dfm)

    dfd = pd.read_csv(detector_csv, encoding="utf-8")
    dfd["ID"] = dfd["ID"].astype(str)
    if not {"ID", "pred_label", "confidence"}.issubset(dfd.columns):
        raise ValueError(f"{detector_csv} deve contenere: ID, pred_label, confidence")
    dfd["confidence"] = pd.to_numeric(dfd["confidence"], errors="coerce")
    dfd["pred_label"] = dfd["pred_label"].astype(str)

    df = dfm[["ID"] + list(feature_cols)].merge(dfd[["ID", "pred_label", "confidence"]], on="ID", how="inner").reset_index(drop=True)

    audit = {
        "metrics_csv": os.path.abspath(metrics_csv),
        "detector_csv": os.path.abspath(detector_csv),
        "n_metrics_ids": int(dfm["ID"].nunique()),
        "n_detector_ids": int(dfd["ID"].nunique()),
        "n_join_ids": int(df["ID"].nunique()),
        "metrics_ids": sorted(dfm["ID"].astype(str).unique().tolist()),
        "detector_ids": sorted(dfd["ID"].astype(str).unique().tolist()),
        "join_ids": sorted(df["ID"].astype(str).unique().tolist()),
        "excluded_metrics_only": sorted(set(dfm["ID"].astype(str)) - set(df["ID"].astype(str))),
        "excluded_detector_only": sorted(set(dfd["ID"].astype(str)) - set(df["ID"].astype(str))),
    }
    return df, audit


def _write_join_audit(out_dir: str, audit: dict) -> None:
    audit_small = {k: v for k, v in audit.items() if not k.endswith("_ids") and not k.startswith("excluded_")}
    _write_json(os.path.join(out_dir, "meta_join_audit.json"), audit_small)
    ids_rows = []
    max_len = max(len(audit.get("metrics_ids", [])), len(audit.get("detector_ids", [])), len(audit.get("join_ids", [])), len(audit.get("excluded_metrics_only", [])), len(audit.get("excluded_detector_only", [])))
    for i in range(max_len):
        ids_rows.append({
            "metrics_id": audit.get("metrics_ids", [None]*max_len)[i] if i < len(audit.get("metrics_ids", [])) else None,
            "detector_id": audit.get("detector_ids", [None]*max_len)[i] if i < len(audit.get("detector_ids", [])) else None,
            "join_id": audit.get("join_ids", [None]*max_len)[i] if i < len(audit.get("join_ids", [])) else None,
            "excluded_metrics_only": audit.get("excluded_metrics_only", [None]*max_len)[i] if i < len(audit.get("excluded_metrics_only", [])) else None,
            "excluded_detector_only": audit.get("excluded_detector_only", [None]*max_len)[i] if i < len(audit.get("excluded_detector_only", [])) else None,
        })
    pd.DataFrame(ids_rows).to_csv(os.path.join(out_dir, "meta_join_audit_ids.csv"), index=False, encoding="utf-8")


def _to_binary_real_fake(pred_label_series: pd.Series) -> np.ndarray:
    s = pred_label_series.astype(str).str.strip().str.upper()

    real_set = {"REAL", "TRUE", "POSITIVE", "LEGIT", "AUTHENTIC"}
    fake_set = {"FAKE", "FALSE", "NEGATIVE", "HOAX", "SYNTHETIC"}

    y = np.full(len(s), -1, dtype=int)

    y[s.isin(real_set)] = 1
    y[s.isin(fake_set)] = 0

    y[s.eq("LABEL_0")] = 0
    y[s.eq("LABEL_1")] = 1

    unk = (y == -1)
    if unk.any():
        tmp = pd.to_numeric(s[unk], errors="coerce")
        is01 = tmp.isin([0, 1])
        idx = np.where(unk)[0]
        if is01.any():
            y[idx[is01.values]] = tmp[is01].astype(int).values

    if (y == -1).any():
        bad = pd.Series(s[y == -1]).value_counts().head(25)
        raise ValueError(
            "Unmapped pred_label values encountered while binarizing REAL/FAKE. "
            f"Top unknown labels: {bad.to_dict()}."
        )

    uniq = np.unique(y)
    if uniq.size > 2:
        raise ValueError(
            f"Binarization produced {uniq.size} classes {uniq.tolist()}. "
            "Detector output is not binary."
        )

    return y


def _save_coefficients(pipe: Pipeline, feature_cols, out_dir: str, detector_name: str, target: str) -> None:
    clf = pipe.named_steps["clf"]
    model_type = type(clf).__name__

    beta = None
    intercept = None

    if hasattr(clf, "coef_"):
        coef = clf.coef_
        if isinstance(coef, np.ndarray) and coef.ndim == 2 and coef.shape[0] >= 1:
            beta = coef[-1].astype(float)
        else:
            beta = np.array(coef, dtype=float).reshape(-1)

    if hasattr(clf, "intercept_"):
        inter = clf.intercept_
        intercept = float(inter[-1]) if isinstance(inter, np.ndarray) else float(inter)

    if beta is None:
        meta = {
            "detector_name": detector_name,
            "target": target,
            "model_type": model_type,
            "note": "Model has no accessible coef_.",
        }
        with open(os.path.join(out_dir, "meta_model_coefficients.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return

    rows = []
    for fcol, b in zip(feature_cols, beta):
        rows.append({
            "feature": fcol,
            "agent": FEATURE_TO_AGENT.get(fcol, "UNKNOWN"),
            "beta": float(b),
            "abs_beta": float(abs(b)),
        })

    dfb = pd.DataFrame(rows).sort_values("abs_beta", ascending=False)
    dfb.to_csv(os.path.join(out_dir, "meta_model_coefficients.csv"), index=False, encoding="utf-8")

    meta = {
        "detector_name": detector_name,
        "target": target,
        "model_type": model_type,
        "intercept": intercept,
        "note": "Coefficients correspond to features AFTER imputation+standardization.",
        "coefficients_csv": os.path.abspath(os.path.join(out_dir, "meta_model_coefficients.csv")),
    }
    with open(os.path.join(out_dir, "meta_model_coefficients.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _neutralize_sa_numeric_when_not_applicable(dfm: pd.DataFrame) -> None:
    if SA_APPLICABILITY_FLAG not in dfm.columns:
        return
    mask = pd.to_numeric(dfm[SA_APPLICABILITY_FLAG], errors="coerce").fillna(0.0) <= 0.0
    for c in SA_NUMERIC_FEATURES:
        if c in dfm.columns:
            dfm.loc[mask, c] = 0.0




def _class_counts(y: np.ndarray) -> dict[int, int]:
    vals, cnts = np.unique(np.asarray(y), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


def _safe_n_splits(y: np.ndarray, desired: int = 5) -> int:
    counts = _class_counts(y)
    if not counts:
        return 2
    min_count = min(counts.values())
    return max(2, min(int(desired), int(min_count)))


def _compute_imbalance_ratio(y: np.ndarray) -> float:
    """Return the minority-class fraction over the total samples."""
    vals, counts = np.unique(np.asarray(y), return_counts=True)
    if len(counts) < 2:
        return 0.0
    return float(counts.min()) / float(counts.sum())


def _build_lasso_cv(random_state: int, y: np.ndarray):
    """LEGACY — kept as internal fallback only."""
    n_splits = _safe_n_splits(y, desired=5)
    cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    imb = _compute_imbalance_ratio(y)
    IMBALANCE_THRESHOLD = 0.20
    if imb < IMBALANCE_THRESHOLD:
        clf = LogisticRegressionCV(
            Cs=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
            l1_ratios=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            cv=cv_inner, scoring="balanced_accuracy", penalty="elasticnet",
            solver="saga", class_weight="balanced", max_iter=15000,
            random_state=random_state, n_jobs=None, refit=True,
        )
    else:
        clf = LogisticRegressionCV(
            Cs=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0],
            cv=cv_inner, scoring="balanced_accuracy", penalty="l1",
            solver="saga", class_weight="balanced", max_iter=10000,
            random_state=random_state, n_jobs=None, refit=True,
        )
    return clf, n_splits


def _write_insufficient_fake_meta(
    out_dir: str,
    detector_name: str,
    n_fake: int,
    n_real: int,
    threshold: int,
) -> None:
    _ensure_dir(out_dir)
    meta_path = os.path.join(out_dir, "feature_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta_model_type": "insufficient_fake",
            "detector_name": detector_name,
            "n_fake": n_fake,
            "n_real": n_real,
            "min_fake_threshold": threshold,
            "status": "STOP_INSUFFICIENT_FAKE",
        }, f, indent=2, ensure_ascii=False)


def _build_sgl_cv(
    random_state: int,
    y: np.ndarray,
    feature_cols: list[str],
    feature_to_agent: dict[str, str] | None = None,
    se_multiplier: float = 1.0,
):
    """Build a Sparse Group Lasso CV classifier.

    Groups are derived from the agent mapping (FEATURE_TO_AGENT).
    Returns (clf, n_splits, groups, agent_order).
    """
    if feature_to_agent is None:
        feature_to_agent = FEATURE_TO_AGENT

    groups, agent_order = build_group_indices(feature_cols, feature_to_agent)
    n_splits = _safe_n_splits(y, desired=5)
    n = len(y)

    if n < 100:
        lams = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        print(f"[META] n={n} -> SGL small-n lambda grid (top={lams[-1]:.4f})")
    elif n < 300:
        lams = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        print(f"[META] n={n} -> SGL medium-n lambda grid (top={lams[-1]:.4f})")
    else:
        lams = DEFAULT_LAMS
        print(f"[META] n={n} -> SGL standard lambda grid (top={lams[-1]:.4f})")

    effective_se_multiplier = se_multiplier
    if n < 100:
        effective_se_multiplier = max(se_multiplier, 2.0)
    elif n < 300:
        effective_se_multiplier = max(se_multiplier, 1.5)

    imb = _compute_imbalance_ratio(y)

    # Adapt alpha grid to imbalance:
    #   alpha capped at 0.5: above that L1 dominates and group structure
    #   is lost, defeating the purpose of SGL.
    #   - low imbalance  -> prefer lower alpha (more group structure)
    #   - good balance   -> wider alpha search up to 0.5
    if imb < 0.15:
        alphas = [0.05, 0.1, 0.15, 0.2, 0.3]
        print(f"[META] Imbalance ratio={imb:.3f} -> SGL with conservative alpha grid {alphas}")
    else:
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
        print(f"[META] Imbalance ratio={imb:.3f} -> SGL with standard alpha grid {alphas}")

    clf = SparseGroupLassoCV(
        lams=lams,
        alphas=alphas,
        groups=groups,
        cv=n_splits,
        scoring="balanced_accuracy",
        class_weight="balanced",
        max_iter=5000,
        tol=1e-7,
        random_state=random_state,
        se_multiplier=effective_se_multiplier,
    )

    print(f"[META] Using Sparse Group Lasso with {len(agent_order)} groups: {agent_order}")
    print(f"[META]   Group sizes: {[int(np.sum(groups == g)) for g in range(len(agent_order))]}")
    print(f"[META]   CV folds: {n_splits}, SE multiplier: {effective_se_multiplier:.2f} (caller: {se_multiplier:.2f})")

    return clf, n_splits, groups, agent_order


def _fit_bootstrap_stability(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    random_state: int,
    out_dir: str,
    threshold: float = 1.0,
    groups: np.ndarray | None = None,
) -> pd.DataFrame | None:
    """Bootstrap stability analysis — works with both SGL and legacy LASSO/ElasticNet."""

    clf_fitted = pipe.named_steps["clf"]

    # ── Determine which resampling model to use ──
    use_sgl = isinstance(clf_fitted, (SparseGroupLassoLogistic, SparseGroupLassoCV))

    if use_sgl:
        # Extract best hyperparameters from SGL CV
        if isinstance(clf_fitted, SparseGroupLassoCV):
            best_lam = clf_fitted.best_lam_
            best_alpha = clf_fitted.best_alpha_
            sgl_groups = clf_fitted.best_estimator_.groups_
        else:
            best_lam = clf_fitted.lam
            best_alpha = clf_fitted.alpha
            sgl_groups = clf_fitted.groups_
    elif isinstance(clf_fitted, LogisticRegressionCV):
        best_C = float(np.atleast_1d(clf_fitted.C_)[0])
        penalty_used = getattr(clf_fitted, "penalty", "l1")
        l1_ratio_used = None
        if penalty_used == "elasticnet":
            l1_ratios_attr = getattr(clf_fitted, "l1_ratio_", None)
            l1_ratio_used = float(np.atleast_1d(l1_ratios_attr)[0]) if l1_ratios_attr is not None else 0.5
    else:
        return None

    try:
        n_samples = len(X)
        all_coefs = []
        rng = np.random.RandomState(random_state if random_state is not None else 0)
        n_boot = 200

        for _b in range(n_boot):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_b = X.iloc[idx].copy()
            y_b = np.asarray(y)[idx]
            if len(np.unique(y_b)) < 2:
                continue

            X_b = X_b[list(feature_cols)]   # guarantee column order matches training
            X_b_imp = pipe.named_steps["impute"].transform(X_b)
            X_b_std = pipe.named_steps["scale"].transform(X_b_imp)

            if use_sgl:
                clf_b = SparseGroupLassoLogistic(
                    lam=best_lam,
                    alpha=best_alpha,
                    groups=sgl_groups,
                    class_weight="balanced",
                    max_iter=3000,
                    tol=1e-6,
                    random_state=random_state,
                )
            elif penalty_used == "elasticnet" and l1_ratio_used is not None:
                clf_b = LogisticRegression(
                    C=best_C, penalty="elasticnet", solver="saga",
                    l1_ratio=l1_ratio_used,
                    class_weight="balanced", max_iter=15000,
                    random_state=random_state,
                )
            else:
                clf_b = LogisticRegression(
                    C=best_C, penalty="l1", solver="saga",
                    class_weight="balanced", max_iter=10000,
                    random_state=random_state,
                )

            clf_b.fit(X_b_std, y_b)
            all_coefs.append(np.asarray(clf_b.coef_[0], dtype=float))

        if len(all_coefs) < 10:
            print("[META][WARN] Too few valid bootstrap iterations — beta_stability.csv not written.")
            return None

        coef_array = np.array(all_coefs)
        beta_mean = coef_array.mean(axis=0)
        beta_std = coef_array.std(axis=0)
        beta_cv = np.abs(beta_std / (np.abs(beta_mean) + 1e-9))
        beta_fit = np.asarray(clf_fitted.coef_[0], dtype=float)
        selected_by_sgl = np.abs(beta_fit) > 1e-8
        reliable = selected_by_sgl & (beta_cv < threshold)

        # Agent coverage info
        agent_col = [FEATURE_TO_AGENT.get(f, "UNKNOWN") for f in feature_cols]

        df_stability = pd.DataFrame({
            "feature": feature_cols,
            "agent": agent_col,
            "beta_fit": beta_fit,
            "beta_mean": beta_mean,
            "beta_std": beta_std,
            "beta_cv": beta_cv,
            "selected_by_sgl": selected_by_sgl,
            "reliable": reliable,
        }).sort_values(["reliable", "selected_by_sgl", "beta_cv"], ascending=[False, False, True])

        df_stability.to_csv(os.path.join(out_dir, "beta_stability.csv"), index=False, encoding="utf-8")

        n_reliable = int(df_stability["reliable"].sum())
        n_selected = int(df_stability["selected_by_sgl"].sum())
        agents_covered = df_stability.loc[df_stability["reliable"], "agent"].nunique()
        method_name = "SGL" if use_sgl else "LASSO"
        if use_sgl:
            print(f"[META] {method_name}+stability: {n_reliable}/{len(feature_cols)} reliable features, "
                  f"{n_selected} non-zero, agents covered={agents_covered}. "
                  f"Best λ={best_lam:.4f}, α={best_alpha:.2f}")
        else:
            print(f"[META] {method_name}+stability: {n_reliable}/{len(feature_cols)} reliable features, "
                  f"{n_selected} non-zero. Best C={best_C:.4f}")

        return df_stability

    except Exception as _bs_err:
        print(f"[META][WARN] Bootstrap stability failed: {_bs_err}")
        import traceback
        traceback.print_exc()
        return None


def _ensure_agent_coverage(
    selected_features: list[str],
    df_stability: pd.DataFrame | None,
    feature_cols: list[str],
    feature_to_agent: dict[str, str] | None = None,
) -> list[str]:
    """Guarantee at least one feature per agent (mandatory inclusion).

    For each agent not represented in *selected_features*, pick the
    feature with the highest |beta_mean| in bootstrap (most stable signal)
    and add it.  This is analogous to mandatory-inclusion constraints
    in epidemiological regression (Heinze et al., 2018).
    """
    if feature_to_agent is None:
        feature_to_agent = FEATURE_TO_AGENT

    all_agents = set(feature_to_agent.get(f) for f in feature_cols if f in feature_to_agent)
    covered = set(feature_to_agent.get(f) for f in selected_features if f in feature_to_agent)
    missing = all_agents - covered

    if not missing:
        return selected_features

    added = list(selected_features)
    for agent in sorted(missing):
        agent_feats = [f for f in feature_cols if feature_to_agent.get(f) == agent]
        if not agent_feats:
            continue

        best_feat = agent_feats[0]  # default fallback

        if df_stability is not None and not df_stability.empty:
            cands = df_stability[df_stability["feature"].isin(agent_feats)].copy()
            if not cands.empty:
                # Prefer feature with highest |beta_mean| (most consistent bootstrap signal)
                cands["abs_beta_mean"] = cands["beta_mean"].abs()
                best_feat = cands.sort_values("abs_beta_mean", ascending=False).iloc[0]["feature"]

        added.append(str(best_feat))
        print(f"[META][COVERAGE] Agent '{agent}' was uncovered → forced '{best_feat}' (mandatory inclusion)")

    return added


def _choose_final_features(
    df_stability: pd.DataFrame | None,
    feature_cols: list[str],
    ensure_coverage: bool = True,
) -> list[str]:
    """Select final feature set from stability analysis, with agent coverage guarantee."""

    if df_stability is None or df_stability.empty:
        return list(feature_cols)

    # Primary: reliable features (non-zero AND stable)
    reliable = df_stability.loc[df_stability["reliable"].fillna(False), "feature"].astype(str).tolist()
    if reliable:
        base = reliable
    else:
        # Fallback: non-zero features from SGL/LASSO
        sel_col = "selected_by_sgl" if "selected_by_sgl" in df_stability.columns else "selected_by_lasso"
        selected = df_stability.loc[df_stability[sel_col].fillna(False), "feature"].astype(str).tolist()
        if selected:
            print("[META][WARN] No reliable features survived stability filter; using non-zero features only.")
            base = selected
        else:
            top = df_stability.sort_values("beta_cv", ascending=True)["feature"].astype(str).head(3).tolist()
            print("[META][WARN] No non-zero features survived; falling back to top-3 most stable features.")
            base = top if top else list(feature_cols)

    # Agent coverage guarantee
    if ensure_coverage:
        base = _ensure_agent_coverage(base, df_stability, feature_cols)

    return base


def _save_selected_features(out_dir: str, selected_features: list[str]) -> None:
    pd.DataFrame({"feature": list(selected_features)}).to_csv(
        os.path.join(out_dir, "selected_features.csv"), index=False, encoding="utf-8"
    )

def train_meta_model_for_detector(
    detector_name: str,
    round_dir: str,
    target: str = "label",
    feature_cols=None,
    random_state: int = 42,
    metrics_csv_name: str = "agent_metrics.csv",
    detector_csv_prefix: str = "detector_",
    output_dir: str | None = None,
    output_base: str = "Output",
    strict_feature_check: bool = True,
    se_multiplier: float = 1.0,
):
    if feature_cols is None:
        feature_cols = FEATURE_COLS_DEFAULT
    else:
        feature_cols = list(feature_cols)

    out_dir = os.path.join(output_base, detector_name, "Baseline") if output_dir is None else output_dir
    _ensure_dir(out_dir)

    metrics_csv, detector_csv = _resolve_meta_input_paths(round_dir, detector_name, metrics_csv_name, detector_csv_prefix)

    print(f"\n[META-MODEL TRAIN] Detector: {detector_name}")
    print(f"  - Round dir:    {round_dir}")
    print(f"  - Metrics CSV:  {metrics_csv}")
    print(f"  - Detector CSV: {detector_csv}")
    print(f"  - Output dir:   {out_dir}")

    df, join_audit = build_meta_join_dataframe(
        detector_name=detector_name,
        round_dir=round_dir,
        feature_cols=feature_cols,
        metrics_csv_name=metrics_csv_name,
        detector_csv_prefix=detector_csv_prefix,
        strict_feature_check=strict_feature_check,
    )
    _write_join_audit(out_dir, join_audit)

    if df.empty:
        payload = {
            "detector_name": detector_name,
            "round_dir": os.path.abspath(round_dir),
            "reason": "empty_meta_join",
            "note": "Meta-model skipped because join between metrics and detector is empty.",
        }
        _write_json(os.path.join(out_dir, "meta_model_skipped.json"), payload)
        print("[META][WARN] Join vuoto: salto training meta-model.")
        return 0

    print(f"[JOIN] righe: {len(df)}")

    if target not in ("label", "prob"):
        raise ValueError("target deve essere 'label' o 'prob'")

    used_target = target
    used_mode = None

    if target == "label":
        y = _to_binary_real_fake(df["pred_label"])
        vc = df["pred_label"].astype(str).str.upper().value_counts().head(20).to_dict()
        print(f"[META][DEBUG] pred_label top: {vc}")
        print("[META][DEBUG] unique(y):", np.unique(y, return_counts=True))

        n_fake = int((y == 0).sum())
        n_real_gate = int((y == 1).sum())
        if n_fake < MIN_ACTIVE_FAKE:
            print(
                f"[META][GATE] n_fake={n_fake} < MIN_ACTIVE_FAKE={MIN_ACTIVE_FAKE}. "
                f"Insufficient FAKE samples for meaningful SGL training. Writing stop marker."
            )
            _write_insufficient_fake_meta(
                out_dir=out_dir,
                detector_name=detector_name,
                n_fake=n_fake,
                n_real=n_real_gate,
                threshold=MIN_ACTIVE_FAKE,
            )
            return None

        uniq = np.unique(y)
        if uniq.size < 2:
            print("[META][WARN] Single-class target after binary conversion "
                  "-> falling back to LASSO L1 on binary target.")
            clf = LogisticRegression(
                penalty="l1", solver="liblinear", C=1.0,
                class_weight="balanced", max_iter=1000, random_state=random_state,
            )
            used_mode = "lasso_l1_fallback"
        else:
            # Bug-1 fix: log imbalance ratio for auditability
            _imb = _compute_imbalance_ratio(np.asarray(y))
            print(f"[META][IMBALANCE] minority_ratio={_imb:.3f}, n_samples={len(y)}")
            try:
                pd.DataFrame([{
                    "detector": detector_name,
                    "minority_ratio": _imb,
                    "n_samples": len(y),
                }]).to_csv(os.path.join(out_dir, "imbalance_report.csv"), index=False)
            except Exception:
                pass
            clf, n_splits, sgl_groups, sgl_agent_order = _build_sgl_cv(
                random_state=random_state, y=y,
                feature_cols=feature_cols,
                se_multiplier=se_multiplier,
            )
            used_mode = "sgl_cv_label"
    else:
        conf = df["confidence"].fillna(0.0).astype(float).values
        y = 1.0 - conf
        try:
            clf = Ridge(alpha=1.0, random_state=random_state)
        except TypeError:
            clf = Ridge(alpha=1.0)
        used_mode = "ridge_real_score"
        sgl_groups = None
        sgl_agent_order = None

    X_full = df[feature_cols].copy()

    initial_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])
    initial_pipe.fit(X_full, y)

    # If all betas are zero after fit, fall back to Ridge on confidence.
    if hasattr(initial_pipe.named_steps["clf"], "coef_"):
        _coef_check = np.asarray(initial_pipe.named_steps["clf"].coef_[0], dtype=float)
        _n_nonzero = int(np.sum(np.abs(_coef_check) > 1e-8))
        print(f"[META] Non-zero betas after initial fit: {_n_nonzero}/{len(_coef_check)}")
        if _n_nonzero == 0:
            print(
                "[META][WARN] All betas are zero after initial fit "
                "-> falling back to LASSO L1 on binary target."
            )
            clf = LogisticRegression(
                penalty="l1", solver="liblinear", C=1.0,
                class_weight="balanced", max_iter=1000, random_state=random_state,
            )
            used_mode = "lasso_l1_fallback"
            sgl_groups = None
            initial_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", clf),
            ])
            initial_pipe.fit(X_full, y)

    df_stability = _fit_bootstrap_stability(
        pipe=initial_pipe,
        X=X_full,
        y=np.asarray(y),
        feature_cols=feature_cols,
        random_state=random_state,
        out_dir=out_dir,
        threshold=10.0,
        groups=sgl_groups,
    )

    selected_features = _choose_final_features(df_stability, feature_cols, ensure_coverage=False)
    _save_selected_features(out_dir, selected_features)
    print(f"[META] Final feature set ({len(selected_features)}): {selected_features}")
    agents_final = sorted(set(FEATURE_TO_AGENT.get(f, "?") for f in selected_features))
    print(f"[META] Agents covered: {agents_final}")

    X = df[selected_features].copy()

    # ── Refit on selected features ──
    if isinstance(initial_pipe.named_steps["clf"], (SparseGroupLassoCV, SparseGroupLassoLogistic)):
        # Rebuild group indices for the selected feature subset
        sel_groups, sel_agents = build_group_indices(selected_features, FEATURE_TO_AGENT)
        _sgl_cv = initial_pipe.named_steps["clf"]
        if isinstance(_sgl_cv, SparseGroupLassoCV):
            _best_lam = _sgl_cv.best_lam_
            _best_alpha = _sgl_cv.best_alpha_
        else:
            _best_lam = _sgl_cv.lam
            _best_alpha = _sgl_cv.alpha
        clf_final = SparseGroupLassoLogistic(
            lam=_best_lam,
            alpha=_best_alpha,
            groups=sel_groups,
            class_weight="balanced",
            max_iter=5000,
            tol=1e-7,
            random_state=random_state,
        )
    elif isinstance(initial_pipe.named_steps["clf"], LogisticRegressionCV):
        _cv_clf = initial_pipe.named_steps["clf"]
        _best_C = float(np.atleast_1d(_cv_clf.C_)[0])
        _penalty = getattr(_cv_clf, "penalty", "l1")
        if _penalty == "elasticnet":
            _l1_ratio = float(np.atleast_1d(getattr(_cv_clf, "l1_ratio_", [0.5]))[0])
            clf_final = LogisticRegression(
                C=_best_C, penalty="elasticnet", solver="saga",
                l1_ratio=_l1_ratio, class_weight="balanced",
                max_iter=15000, random_state=random_state,
            )
        else:
            clf_final = LogisticRegression(
                C=_best_C, penalty="l1", solver="saga",
                class_weight="balanced", max_iter=10000,
                random_state=random_state,
            )
    else:
        clf_final = clf

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf_final),
    ])
    pipe.fit(X, y)

    _save_coefficients(pipe, selected_features, out_dir, detector_name, used_target)

    if isinstance(pipe.named_steps["clf"], (LogisticRegression, LogisticRegressionCV, SparseGroupLassoLogistic, SparseGroupLassoCV)):
        ypred = pipe.predict(X)
        print("\n[TRAIN] Classification report")
        print(classification_report(y, ypred, digits=3))
        try:
            yprob = pipe.predict_proba(X)[:, 1]
            print("TRAIN AUC: {:.3f}".format(roc_auc_score(y, yprob)))
        except Exception:
            pass
    else:
        yhat = pipe.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, yhat)))
        print("\n[TRAIN] RMSE: {:.4f}".format(rmse))

    X_all = df[selected_features].copy()
    X_all = X_all[list(selected_features)]   # guarantee column order matches training
    X_all_imp = pipe.named_steps["impute"].transform(X_all)
    X_all_std = pipe.named_steps["scale"].transform(X_all_imp)
    np.save(os.path.join(out_dir, "X_all_std.npy"), X_all_std)

    masker = shap.maskers.Independent(X_all_std)
    explainer = shap.LinearExplainer(pipe.named_steps["clf"], masker)
    sv_all = explainer.shap_values(X_all_std)
    if isinstance(sv_all, list):
        sv_all = sv_all[-1]
    np.save(os.path.join(out_dir, "shap_values.npy"), sv_all)

    baseline_rows = []
    for c in feature_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        baseline_rows.append({
            "feature": c,
            "agent": FEATURE_TO_AGENT.get(c, "UNKNOWN"),
            "median": float(col.median(skipna=True)),
            "q20": float(col.quantile(0.20)),
            "q50": float(col.quantile(0.50)),
            "q80": float(col.quantile(0.80)),
            "mean": float(col.mean(skipna=True)),
            "std": float(col.std(skipna=True)),
            "min": float(col.min(skipna=True)),
            "max": float(col.max(skipna=True)),
        })
    pd.DataFrame(baseline_rows).to_csv(
        os.path.join(out_dir, "feature_baseline_stats.csv"),
        index=False,
        encoding="utf-8",
    )

    per_id_rows = []
    for i in range(len(df)):
        row_vals = {c: float(df.loc[i, c]) if pd.notna(df.loc[i, c]) else np.nan for c in feature_cols}
        shap_row = {f"shap_{c}": float(sv_all[i, j]) for j, c in enumerate(selected_features)}
        per_id_rows.append({
            "ID": str(df.loc[i, "ID"]),
            "target_value": float(y[i]) if i < len(y) else np.nan,
            **row_vals,
            **shap_row,
            "detector_pred_label": str(df.loc[i, "pred_label"]),
            "detector_confidence": float(df.loc[i, "confidence"]) if pd.notna(df.loc[i, "confidence"]) else np.nan,
        })

    pd.DataFrame(per_id_rows).to_csv(
        os.path.join(out_dir, "shap_attributions_per_id.csv"),
        index=False,
        encoding="utf-8",
    )

    pipe_path = os.path.join(out_dir, "meta_model_pipe.pkl")
    meta_path = os.path.join(out_dir, "feature_meta.json")
    joblib.dump(pipe, pipe_path)

    # §11 — compute fidelity baseline (Zhou 2004 / Gama 2004)
    _fidelity_baseline: dict = {}
    try:
        import math as _math
        _fake_mask = (np.asarray(y) == 0)
        _n0 = int(_fake_mask.sum())
        if _n0 > 0:
            _X_fake = X_all[_fake_mask]
            _surr_pred_fake = pipe.predict(_X_fake)
            _fid0 = float((_surr_pred_fake == 0).sum()) / _n0
            _e_min = 1.0 - _fid0
            _sigma_min = _math.sqrt(max(_e_min * (1.0 - _e_min) / _n0, 1e-9))
            _fidelity_baseline = {
                "e_min": round(_e_min, 4),
                "sigma_min": round(_sigma_min, 4),
                "n_0": _n0,
                "fidelity_0": round(_fid0, 4),
                "round": 0,
            }
            print(
                f"[META] Fidelity baseline: fidelity_0={_fid0:.3f} "
                f"e_min={_e_min:.3f} sigma_min={_sigma_min:.3f} n_0={_n0}"
            )
    except Exception as _e_fid:
        print(f"[META][WARN] Fidelity baseline computation failed: {_e_fid}")

    with open(meta_path, "w", encoding="utf-8") as f:
        _final_clf = pipe.named_steps["clf"]
        if isinstance(_final_clf, SparseGroupLassoLogistic):
            _meta_model_type = "sparse_group_lasso"
        elif used_mode == "lasso_l1_fallback":
            _meta_model_type = "lasso_l1_fallback"
        elif used_mode == "fallback_ridge_all_zero_betas":  # legacy safety — should not be reached
            _meta_model_type = "ridge_fallback"
        elif isinstance(_final_clf, (LogisticRegression, LogisticRegressionCV)):
            _meta_model_type = "lasso_cv"
        else:
            _meta_model_type = type(_final_clf).__name__.lower()

        meta_payload = {
            "detector_name": detector_name,
            "feature_cols": list(selected_features),
            "all_candidate_features": list(feature_cols),
            "feature_to_agent": FEATURE_TO_AGENT,
            "target": used_target,
            "mode": used_mode,
            "meta_model_type": _meta_model_type,
            "random_state": random_state,
            "metrics_csv": os.path.abspath(metrics_csv),
            "detector_csv": os.path.abspath(detector_csv),
            "output_dir": os.path.abspath(out_dir),
            "pipe_path": os.path.abspath(pipe_path),
            "coefficients_csv": os.path.abspath(os.path.join(out_dir, "meta_model_coefficients.csv")),
        }
        # SGL-specific metadata
        if isinstance(_final_clf, SparseGroupLassoLogistic):
            meta_payload["sgl_lam"] = float(_final_clf.lam)
            meta_payload["sgl_alpha"] = float(_final_clf.alpha)
            meta_payload["sgl_groups"] = _final_clf.groups_.tolist() if hasattr(_final_clf, "groups_") else None
            meta_payload["agents_covered"] = sorted(set(FEATURE_TO_AGENT.get(f, "?") for f in selected_features))
            meta_payload["sgl_se_multiplier"] = float(se_multiplier)
        if _fidelity_baseline:
            meta_payload["fidelity_baseline"] = _fidelity_baseline
        json.dump(meta_payload, f, indent=2, ensure_ascii=False)

    print(f"[META-MODEL TRAIN] Saved outputs -> {out_dir}")
    print(f"[META-MODEL TRAIN] Done for detector '{detector_name}'.")
    return 0


def _load_feature_cols_from_pipe_meta(baseline_pipe_path: str) -> list[str] | None:
    d = os.path.dirname(os.path.abspath(baseline_pipe_path))
    meta_path = os.path.join(d, "feature_meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cols = meta.get("feature_cols", None)
        if isinstance(cols, list) and all(isinstance(x, str) for x in cols) and len(cols) > 0:
            return cols
    except Exception:
        return None
    return None


def compute_shap_with_frozen_baseline(
    baseline_pipe_path: str,
    round_dir: str,
    detector_name: str,
    out_dir: str,
    metrics_csv_name: str = "agent_metrics.csv",
    detector_csv_prefix: str = "detector_",
    target: str = "label",
    feature_cols=None,
    only_ids: list[str] | None = None,
    append_if_exists: bool = False,
    carry_forward_values: dict | None = None,
):
    if not os.path.exists(baseline_pipe_path):
        raise FileNotFoundError(f"Missing baseline pipe: {baseline_pipe_path}")

    meta_cols = _load_feature_cols_from_pipe_meta(baseline_pipe_path)
    if meta_cols is not None:
        feature_cols_final = meta_cols
    else:
        feature_cols_final = FEATURE_COLS_DEFAULT if feature_cols is None else list(feature_cols)

    _ensure_dir(out_dir)

    metrics_csv, detector_csv = _resolve_meta_input_paths(round_dir, detector_name, metrics_csv_name, detector_csv_prefix)
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Missing metrics: {metrics_csv}")
    if not os.path.exists(detector_csv):
        raise FileNotFoundError(f"Missing detector csv: {detector_csv}")

    df, join_audit = build_meta_join_dataframe(
        detector_name=detector_name,
        round_dir=round_dir,
        feature_cols=feature_cols_final,
        metrics_csv_name=metrics_csv_name,
        detector_csv_prefix=detector_csv_prefix,
        strict_feature_check=False,
    )
    _write_join_audit(out_dir, join_audit)

    if only_ids is not None:
        only_set = set(str(x) for x in only_ids)
        df = df[df["ID"].astype(str).isin(only_set)].copy().reset_index(drop=True)

    if df.empty:
        print("[SHAP][WARN] No samples available for SHAP after joins/filters -> skip.")
        out_path = os.path.join(out_dir, "shap_attributions_per_id.csv")
        if not append_if_exists:
            pd.DataFrame(columns=["ID"]).to_csv(out_path, index=False, encoding="utf-8")
        return pd.DataFrame(columns=["ID"])

    used_target = target
    if target == "label":
        ybin = _to_binary_real_fake(df["pred_label"])
        uniq = np.unique(ybin)
        if uniq.size < 2:
            y = 1.0 - df["confidence"].fillna(0.0).astype(float).values
            used_target = "prob"
        else:
            y = ybin
    elif target == "prob":
        y = 1.0 - df["confidence"].fillna(0.0).astype(float).values
        used_target = "prob"
    else:
        raise ValueError("target must be 'label' or 'prob'")

    X = df[feature_cols_final].copy()

    # Carry-forward: fill per-ID NaN values from the most recent valid round
    # before falling back to the frozen imputer's round-0 training medians.
    # This reduces artificial distributional smoothing caused by median imputation
    # when a metric is temporarily unavailable (agent inactive this round).
    if carry_forward_values:
        ids = df["ID"].astype(str).tolist()
        n_filled = 0
        for i, _id in enumerate(ids):
            id_vals = carry_forward_values.get(_id, {})
            for feat in feature_cols_final:
                if feat in id_vals and feat in X.columns and pd.isna(X.iloc[i][feat]):
                    X.iloc[i, X.columns.get_loc(feat)] = id_vals[feat]
                    n_filled += 1
        if n_filled:
            print(f"[SHAP][CARRY-FWD] Applied {n_filled} carry-forward fills across {len(ids)} IDs.")

    pipe = joblib.load(baseline_pipe_path)
    if not all(k in pipe.named_steps for k in ("impute", "scale", "clf")):
        raise ValueError("Loaded baseline pipe must contain steps: impute, scale, clf")

    imputer = pipe.named_steps["impute"]
    scaler = pipe.named_steps["scale"]
    clf = pipe.named_steps["clf"]

    if hasattr(imputer, "n_features_in_") and X.shape[1] != imputer.n_features_in_:
        raise ValueError(
            f"Column count mismatch: X has {X.shape[1]} features, "
            f"imputer expects {imputer.n_features_in_}. "
            f"Possible column order or feature set mismatch."
        )
    X = X[feature_cols_final]   # guarantee column order matches training
    X_imp = imputer.transform(X)
    X_std = scaler.transform(X_imp)

    if hasattr(clf, "coef_"):
        n_model = int(np.array(clf.coef_).shape[-1])
        if X_std.shape[1] != n_model:
            raise ValueError(
                "Feature mismatch between current X and frozen model. "
                f"X_std.shape[1]={X_std.shape[1]} but model expects {n_model}."
            )

    explainer = shap.LinearExplainer(
        clf,
        shap.maskers.Independent(X_std),
    )
    sv_all = explainer.shap_values(X_std)
    if isinstance(sv_all, list):
        sv_all = sv_all[-1]

    rows = []
    for i in range(len(df)):
        row = {
            "ID": str(df.loc[i, "ID"]),
            "target_value": float(y[i]) if i < len(y) else np.nan,
            "detector_pred_label": str(df.loc[i, "pred_label"]),
            "detector_confidence": float(df.loc[i, "confidence"]) if pd.notna(df.loc[i, "confidence"]) else np.nan,
            "target_mode": used_target,
        }
        for c in feature_cols_final:
            row[c] = float(df.loc[i, c]) if pd.notna(df.loc[i, c]) else np.nan
        for j, c in enumerate(feature_cols_final):
            row[f"shap_{c}"] = float(sv_all[i, j])
        rows.append(row)

    out_path = os.path.join(out_dir, "shap_attributions_per_id.csv")
    df_out = pd.DataFrame(rows)

    if append_if_exists and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            prev = pd.read_csv(out_path, encoding="utf-8")
            if "ID" in prev.columns:
                prev["ID"] = prev["ID"].astype(str)
            if "ID" in df_out.columns:
                df_out["ID"] = df_out["ID"].astype(str)
            df_out = pd.concat([prev, df_out], ignore_index=True)
            if "ID" in df_out.columns:
                df_out = df_out.drop_duplicates(subset=["ID"], keep="last")
        except Exception:
            pass

    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[META-FROZEN] Saved SHAP per-ID -> {out_path}")