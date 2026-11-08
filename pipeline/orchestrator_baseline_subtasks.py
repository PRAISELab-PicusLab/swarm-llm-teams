import os
import json
from typing import Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd

from .metrics_runner import METRICS_REGISTRY
from .objectives import get_objective_text

FEATURE_COLS = list(METRICS_REGISTRY.keys())
FEATURE_TO_AGENT = {k: v.agent_id for k, v in METRICS_REGISTRY.items()}

FEATURE_META = {
    "SA_semantic_coverage": ("Semantic Coverage", "SA_REFRESH_SENTENCE_COVERAGE"),
    "SA_text_compression_ratio": ("Text Compression Ratio", "SA_REFRESH_SENTENCE_COMPRESSION"),
    "SA_numeric_coverage": ("Numeric Coverage", "SA_REFRESH_NUMBER_COVERAGE"),
    "SA_numeric_compression_ratio": ("Numeric Compression Ratio", "SA_REFRESH_NUMBER_COMPRESSION"),

    "SSE_semantic_drift": ("Semantic Drift", "SSE_EDIT_SEMANTIC_DRIFT"),
    "SSE_lexical_stability": ("Lexical Stability", "SSE_EDIT_LEXICAL_STABILITY"),
    "SSE_fluency_ppl": ("Fluency (Perplexity)", "SSE_EDIT_FLUENCY"),

    "NM_edit_success_rate": ("Edit Success Rate", "NM_IMPROVE_EDIT_SUCCESS"),
    "NM_change_magnitude": ("Change Magnitude", "NM_EDIT_MAGNITUDE"),

    "NaM_rewrite_distance": ("Global Rewrite Distance", "NaM_CONTROL_REWRITE_DISTANCE"),
    "NaM_contradiction_prob": ("Contradiction Probability", "NaM_REDUCE_CONTRADICTIONS"),

    "TE_title_entailment": ("Title Entailment", "TE_IMPROVE_ENTAILMENT"),
    "TE_title_contradiction_prob": ("Title Contradiction Probability", "TE_REDUCE_CONTRADICTION"),

    # Style-vector features: MUST mirror FEATURE_TO_METRIC_NAME/FEATURE_TO_SUBTASK in
    # orchestrator_iterative.py — any feature missing here is silently dropped at line ~312.
    "TE_style_titlecase": ("Title Case Ratio", "TE_STYLE_TITLECASE"),
    "TE_style_allcaps": ("All-Caps Rate", "TE_STYLE_ALLCAPS"),
    "TE_style_exclquest": ("Exclamation/Question Rate", "TE_STYLE_EXCLQUEST"),
    "TE_style_title_len": ("Title Length", "TE_STYLE_TITLE_LEN"),
    "TE_style_nnp": ("Proper-Noun Rate", "TE_STYLE_NNP"),

    "NaM_style_fk_grade": ("Reading Grade Level", "NaM_STYLE_READING_GRADE"),
    "NaM_style_ttr": ("Lexical Diversity", "NaM_STYLE_LEXICAL_DIVERSITY"),
    "NaM_style_quote_rate": ("Quotation Rate", "NaM_STYLE_QUOTE_RATE"),
    "NaM_style_punct_rate": ("Punctuation Rate", "NaM_STYLE_PUNCT_RATE"),
    "NaM_style_adjadv": ("Adjective/Adverb Density", "NaM_STYLE_ADJADV"),
}


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -------------------------------
# PATH-FIX: detector discovery
# -------------------------------
def _find_detector_csv(round_dir: str, detector_name: str, prefix: str) -> str:
    """
    NEW STRUCTURE compatible:
      1) <round_dir>/generation/detector_<detector>.csv
      2) <round_dir>/detectors/<detector>/detector_<detector>.csv
      3) legacy fallback: <round_dir>/detector_<detector>.csv
    """
    c1 = os.path.join(round_dir, "generation", f"{prefix}{detector_name}.csv")
    c2 = os.path.join(round_dir, "detectors", detector_name, f"{prefix}{detector_name}.csv")
    c3 = os.path.join(round_dir, f"{prefix}{detector_name}.csv")
    for p in (c1, c2, c3):
        if os.path.exists(p):
            return p
    # return most likely (so error msg is helpful)
    return c1


def _load_detector(round_dir: str, detector_name: str, prefix: str):
    det_path = _find_detector_csv(round_dir, detector_name, prefix)
    if not os.path.exists(det_path):
        raise FileNotFoundError(f"Missing detector csv (searched gen/detectors/legacy). Expected: {det_path}")

    df = pd.read_csv(det_path, encoding="utf-8")
    df["ID"] = df["ID"].astype(str)
    df["pred_label"] = df["pred_label"].astype(str).str.upper()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df[["ID", "pred_label", "confidence"]]


def _load_coefficients(meta_dir: str) -> Dict[str, float]:
    p = os.path.join(meta_dir, "meta_model_coefficients.csv")
    df = pd.read_csv(p, encoding="utf-8")
    if "feature" not in df.columns or "beta" not in df.columns:
        raise ValueError(f"meta_model_coefficients.csv must have columns: feature, beta. Found: {list(df.columns)}")
    df["feature"] = df["feature"].astype(str)
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    return dict(zip(df["feature"], df["beta"]))


def _direction_from_beta(beta: float) -> str:
    return "INCREASE" if beta > 0 else "DECREASE"


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


def _orch_guide_legacy(agent, metric, direction, objective) -> str:
    return "\n".join([
        "[ORCH_GUIDE]",
        f"Agent: {agent}",
        f"Metric: {metric}",
        f"Direction: {direction}",
        f"Objective: {objective}",
        "[/ORCH_GUIDE]",
    ])




def _load_feature_meta(meta_dir: str) -> dict:
    p = os.path.join(meta_dir, "feature_meta.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_reliable_features(meta_dir: str) -> Set[str]:
    p = os.path.join(meta_dir, "beta_stability.csv")
    if not os.path.exists(p):
        return set()
    try:
        df = pd.read_csv(p, encoding="utf-8")
        if "feature" not in df.columns:
            return set()
        if "reliable" in df.columns:
            return set(df.loc[df["reliable"].fillna(False), "feature"].astype(str).tolist())
        return set(df["feature"].astype(str).tolist())
    except Exception:
        return set()

def _rank_features_for_id(shap_vals: Dict[str, float], eligible: Optional[Set[str]] = None) -> list[str]:
    """Rank features by SHAP sign then magnitude. Features with SHAP < 0
    come first (sorted by |SHAP| descending), then features with SHAP >= 0
    (sorted by |SHAP| descending). Negative SHAP on a FAKE-classified
    article means the feature is actively pushing the prediction away from
    REAL and has the most room for improvement."""
    items = shap_vals.items() if eligible is None else [(k, v) for k, v in shap_vals.items() if k in eligible]
    neg = [(k, v) for k, v in items if v < 0]
    pos = [(k, v) for k, v in items if v >= 0]
    neg_sorted = sorted(neg, key=lambda kv: abs(kv[1]), reverse=True)
    pos_sorted = sorted(pos, key=lambda kv: abs(kv[1]), reverse=True)
    return [k for (k, _) in neg_sorted] + [k for (k, _) in pos_sorted]


def _select_next_feature(shap_vals: Dict[str, float], eligible: Optional[Set[str]] = None) -> Optional[Tuple[str, float]]:
    for f in _rank_features_for_id(shap_vals, eligible=eligible):
        return f, float(shap_vals.get(f, 0.0))
    return None


def _select_least_feature(shap_vals: Dict[str, float], eligible: Optional[Set[str]] = None) -> Optional[Tuple[str, float]]:
    """ANTI-FEATURE ablation (exp3): pick the LEAST important eligible feature
    (smallest |SHAP|). Mirrors orchestrator_iterative._select_least_feature so
    round-0 decisions (which drive round 1) follow the same experiment."""
    pool = eligible if eligible is not None else set(shap_vals.keys())
    cands = [(f, abs(float(shap_vals.get(f, 0.0)))) for f in pool if f in FEATURE_COLS]
    if not cands:
        return None
    f = min(cands, key=lambda kv: kv[1])[0]
    return f, float(shap_vals.get(f, 0.0))


# -------------------------------
# PATH-FIX: shap discovery
# -------------------------------
def _find_shap_csv(round_dir: str, detector_name: str, meta_dir: Optional[str], shap_filename: str) -> str:
    """
    Round0 (frozen) typically:  <meta_dir>/shap_attributions_per_id.csv
    Round>0 typically:          <round_dir>/shap/<detector>/shap_attributions_per_id.csv
    """
    candidates = []
    if meta_dir:
        candidates.append(os.path.join(meta_dir, shap_filename))
    candidates.append(os.path.join(round_dir, "shap", detector_name, shap_filename))

    for p in candidates:
        if os.path.exists(p):
            return p

    # helpful fallback
    return candidates[-1]


def run_orchestrator_round(
    detector_name: str,
    round_dir: str,
    detector_csv_prefix: str = "detector_",
    filter_fake: bool = True,
    meta_dir: str | None = None,
    shap_filename: str = "shap_attributions_per_id.csv",
    only_ids: set[str] | list[str] | None = None,
    append_if_exists: bool = False,
    invert_direction: bool = False,
    anti_feature: bool = False,
):
    """
    Outputs:
      <round_dir>/orchestrator/<detector>/...

    invert_direction (exp2) / anti_feature (exp3): framework-validity ablations,
    applied to the round-0 decisions so the FIRST loop round is already ablated.
    """
    if meta_dir is None:
        # keep your default, but note: in new structure round>0 SHAP is under round_dir/shap/
        meta_dir = os.path.join(round_dir, "meta", detector_name)

    shap_path = _find_shap_csv(round_dir, detector_name, meta_dir, shap_filename)  # PATH-FIX
    coeff_path = os.path.join(meta_dir, "meta_model_coefficients.csv")

    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"Missing SHAP per-ID attributions (searched meta_dir + round_dir/shap): {shap_path}")
    if not os.path.exists(coeff_path):
        raise FileNotFoundError(f"Missing meta-model coefficients: {coeff_path}")

    orch_dir_new = os.path.join(round_dir, "orchestrator", detector_name)
    _ensure_dir(orch_dir_new)

    df = pd.read_csv(shap_path, encoding="utf-8")
    if "ID" not in df.columns:
        raise ValueError(f"SHAP file missing 'ID' column: {shap_path}")
    df["ID"] = df["ID"].astype(str)

    if only_ids is not None:
        only_set = set(str(x) for x in only_ids)
        df = df[df["ID"].astype(str).isin(only_set)].copy()

    det = _load_detector(round_dir, detector_name, detector_csv_prefix)  # PATH-FIX
    df = df.merge(det, on="ID", how="inner")

    if filter_fake:
        df = df[df["pred_label"].astype(str).str.upper() == "FAKE"].copy()

    betas = _load_coefficients(meta_dir)
    feature_meta = _load_feature_meta(meta_dir)
    eligible_features = set(feature_meta.get("feature_cols", []) or [])
    reliable_features = _load_reliable_features(meta_dir)
    if reliable_features:
        eligible_features = (eligible_features & reliable_features) if eligible_features else reliable_features
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
        eligible_features = reliable_features if reliable_features else set(FEATURE_COLS)

    rows = []
    guides_path_new = os.path.join(orch_dir_new, "orch_guides.jsonl")
    out_csv_new = os.path.join(orch_dir_new, "orch_decisions_with_reasons.csv")

    _meta_model_type = feature_meta.get("meta_model_type", "sparse_group_lasso")
    _BLOCK_MODEL_TYPES = {"ridge_fallback", "insufficient_fake"}
    if _meta_model_type in _BLOCK_MODEL_TYPES:
        print(f"[ORCH][WARN] Detector '{detector_name}' meta_model_type='{_meta_model_type}': "
              f"SHAP attributions are not interpretable. Skipping orchestration.")
        open(guides_path_new, "w", encoding="utf-8").close()
        df_out = pd.DataFrame(columns=[
            "ID", "chosen_feature", "chosen_subtask", "pivot_agent", "objective",
            "agent", "metric", "direction", "chosen_shap", "already_used_count",
            "state_SA", "state_SSE", "state_NM", "state_NaM", "state_TE",
            "run_SA", "run_SSE", "run_NM", "run_NaM", "run_TE",
            "guide_legacy_internal", "guide_en",
        ])
        df_out.to_csv(out_csv_new, index=False, encoding="utf-8")
        return df_out

    prev_dec_new = None
    prev_guides_new = {}
    if append_if_exists and os.path.exists(out_csv_new) and os.path.getsize(out_csv_new) > 0:
        try:
            prev_dec_new = pd.read_csv(out_csv_new, encoding="utf-8")
            if "ID" in prev_dec_new.columns:
                prev_dec_new["ID"] = prev_dec_new["ID"].astype(str)
        except Exception:
            prev_dec_new = None

    if append_if_exists and os.path.exists(guides_path_new) and os.path.getsize(guides_path_new) > 0:
        try:
            with open(guides_path_new, "r", encoding="utf-8") as gf_prev:
                for line in gf_prev:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        _id = str(obj.get("ID", ""))
                        if _id:
                            prev_guides_new[_id] = obj
                    except Exception:
                        continue
        except Exception:
            prev_guides_new = {}

    with open(guides_path_new, "w", encoding="utf-8") as gf_new:
        for _, r in df.iterrows():
            _id = str(r["ID"])

            shap_vals = {}
            for c in FEATURE_COLS:
                v = r.get(f"shap_{c}", 0.0)
                try:
                    shap_vals[c] = float(v)
                except Exception:
                    shap_vals[c] = 0.0

            if anti_feature:
                pick = _select_least_feature(shap_vals, eligible=eligible_features)
            else:
                pick = _select_next_feature(shap_vals, eligible=eligible_features)
            if pick is None:
                continue

            feat, shap_val = pick
            if feat not in FEATURE_TO_AGENT or feat not in FEATURE_META:
                continue

            pivot_agent = FEATURE_TO_AGENT[feat]
            metric_name, subtask = FEATURE_META[feat]

            _raw_beta = betas.get(feat, None)
            if _raw_beta is None or (isinstance(_raw_beta, float) and np.isnan(_raw_beta)):
                print(f"[WARN] feature '{feat}' selected but has no valid beta in "
                      f"coefficients — eligible_features filter should have prevented "
                      f"this. Using 0.0 as fallback.")
                _raw_beta = 0.0
            beta = float(_raw_beta)
            direction = _direction_from_beta(beta)
            if invert_direction:
                direction = "DECREASE" if direction == "INCREASE" else "INCREASE"

            objective = get_objective_text(pivot_agent, metric_name, direction).strip()
            if not objective:
                objective = "Revise the assigned content according to the instructions, preserving coherence, plausibility, and required output format."

            guide_legacy = _orch_guide_legacy(pivot_agent, metric_name, direction, objective)
            flags = _state_flags(pivot_agent)

            row = {
                "ID": _id,
                "chosen_feature": feat,
                "chosen_subtask": subtask,

                "pivot_agent": pivot_agent,
                "objective": objective,

                "agent": pivot_agent,
                "metric": metric_name,
                "direction": direction,
                "chosen_shap": float(shap_val),
                "already_used_count": 0,
                **flags,

                "guide_legacy_internal": guide_legacy,
                "guide_en": objective,
            }
            rows.append(row)

            payload = {"ID": _id, "pivot_agent": pivot_agent, "objective": objective}
            prev_guides_new[_id] = payload

        for _id, payload in prev_guides_new.items():
            gf_new.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if rows:
        df_out = pd.DataFrame(rows)
    else:
        df_out = pd.DataFrame(columns=[
            "ID", "chosen_feature", "chosen_subtask", "pivot_agent", "objective",
            "agent", "metric", "direction", "chosen_shap", "already_used_count",
            "state_SA", "state_SSE", "state_NM", "state_NaM", "state_TE",
            "run_SA", "run_SSE", "run_NM", "run_NaM", "run_TE",
            "guide_legacy_internal", "guide_en",
        ])

    if append_if_exists and isinstance(prev_dec_new, pd.DataFrame) and not prev_dec_new.empty:
        try:
            if "ID" in df_out.columns:
                df_out["ID"] = df_out["ID"].astype(str)
            df_out = pd.concat([prev_dec_new, df_out], ignore_index=True)
            if "ID" in df_out.columns:
                df_out = df_out.drop_duplicates(subset=["ID"], keep="last")
        except Exception:
            pass

    df_out.to_csv(out_csv_new, index=False, encoding="utf-8")

    print(f"[ORCH] Done for detector {detector_name}. Wrote decisions to {orch_dir_new}")
    return df_out


class BaselineOrchestrator:
    def __init__(
        self,
        detector_name: str,
        round_dir: str,
        detector_csv_prefix: str = "detector_",
        filter_fake: bool = True,
        meta_dir: str | None = None,
        shap_filename: str = "shap_attributions_per_id.csv",
        only_ids: set[str] | list[str] | None = None,
        append_if_exists: bool = False,
    ):
        self.detector_name = detector_name
        self.round_dir = round_dir
        self.detector_csv_prefix = detector_csv_prefix
        self.filter_fake = filter_fake
        self.meta_dir = meta_dir
        self.shap_filename = shap_filename
        self.only_ids = only_ids
        self.append_if_exists = append_if_exists

    def run(self):
        return run_orchestrator_round(
            detector_name=self.detector_name,
            round_dir=self.round_dir,
            detector_csv_prefix=self.detector_csv_prefix,
            filter_fake=self.filter_fake,
            meta_dir=self.meta_dir,
            shap_filename=self.shap_filename,
            only_ids=self.only_ids,
            append_if_exists=self.append_if_exists,
        )