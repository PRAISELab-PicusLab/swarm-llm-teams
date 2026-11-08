import os
import json
import inspect
import tempfile
import shutil
import argparse
import pandas as pd
import numpy as np

from .metrics_runner import compute_agent_metrics, METRICS_REGISTRY
from .detector_runner import run_detector_on_df
from .orchestrator_iterative import run_orchestrator_round_iterative
from .pipeline_multiagent import run_generation_pipeline
from .meta_model import compute_shap_with_frozen_baseline
from .drift_monitor import compute_drift_report
from .retraining_buffer import (
    update_fake_accumulated,
    do_retrain_and_reload,
    _buffer_dir as _retrain_buffer_dir,
)
from .constants import MIN_ACTIVE_FAKE


RETRAIN_COOLDOWN_ROUNDS = 3

# Uniform statistical-power threshold: every round (including the entry to the
# loop right after baseline) must have at least MIN_ACTIVE_FAKE ACTIVE_FAKE IDs.
# Below this, KS feature drift, DDM surrogate fidelity, and SGL retrain all lose
# statistical meaning, so the loop stops with a STOP_INSUFFICIENT_ACTIVE marker.
# Can be overridden via CLI --min-active-fake (e.g. 10 for small "test mini"
# cohorts). Sourced from pipeline.constants (single source of truth, shared with
# meta_model / drift_monitor / retraining_buffer).
DEFAULT_MIN_ACTIVE_FAKE = MIN_ACTIVE_FAKE


METRICS_CSV_NAME = "agent_metrics.csv"
DETECTOR_PREFIX = "detector_"


def _round_tag(r: int) -> str:
    # :03d pads to minimum 3 digits; Python never truncates, so round_1000 is correct.
    return f"round_{r:03d}"

def _round_dir(root: str, r: int) -> str:
    return os.path.join(root, "rounds", _round_tag(r))

def _gen_root_dir(root: str, r: int) -> str:
    """Generation ROOT for a given round (shared across detectors).

    Round 0 is always shared. For round>=1 we keep a shared root to store
    detector_<name>.csv (compat with orchestrator/SHAP), and put the
    generation artifacts (SSE/NM/Generated_Fake_News/panel parquet/metrics)
    inside a per-detector subfolder.
    """
    return os.path.join(_round_dir(root, r), "generation")


def _gen_dir(root: str, r: int, detector: str | None = None) -> str:
    """Generation directory where pipeline_multiagent writes and reads intermediates.

    - round 0: always shared
    - round >= 1: per-detector subfolder to avoid mixing intermediates.
    """
    gen_root = _gen_root_dir(root, r)
    if r == 0 or not detector:
        return gen_root
    return os.path.join(gen_root, detector)

def _det_dir(root: str, r: int, detector: str) -> str:
    return os.path.join(_round_dir(root, r), "detectors", detector)

def _orch_dir(root: str, r: int, detector: str) -> str:
    return os.path.join(_round_dir(root, r), "orchestrator", detector)

def _shap_dir(root: str, r: int, detector: str) -> str:
    return os.path.join(_round_dir(root, r), "shap", detector)

def _meta_frozen_dir(root: str, detector: str) -> str:
    return os.path.join(root, "meta_frozen", detector)

def _tracking_dir(root: str, detector: str) -> str:
    return os.path.join(root, "tracking", detector)



def _trajectory_path(root: str, detector: str) -> str:
    return os.path.join(root, "trajectory", detector, "trajectory.json")

def _agent_snapshots_dir(root: str, detector: str, round_idx: int) -> str:
    return os.path.join(root, "trajectory", detector, "agent_snapshots", f"round_{round_idx:03d}")

def _article_snapshots_dir(root: str, detector: str, round_idx: int) -> str:
    return os.path.join(root, "trajectory", detector, "article_snapshots", f"round_{round_idx:03d}")

def _json_safe_float(x):
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None

def _load_json_dict(path: str) -> dict:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_json_dict(data: dict, path: str) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)

def _update_trajectory_json(
    trajectory_path: str,
    metrics_csv_path: str,
    round_idx: int,
    detector: str,
) -> None:
    """Append metric values for one round to the cumulative trajectory artifact.

    This is a separate feedback channel for agents. It does not alter metric
    computation, residualization, SHAP, detector outputs, or meta-model state.
    """
    if not os.path.exists(metrics_csv_path):
        print(f"[TRAJ] Metrics file missing, cannot update trajectory: {metrics_csv_path}")
        return

    try:
        dfm = pd.read_csv(metrics_csv_path, encoding="utf-8")
    except Exception as e:
        print(f"[TRAJ] Could not read metrics file {metrics_csv_path}: {e}")
        return
    if "ID" not in dfm.columns:
        print(f"[TRAJ] Metrics file has no ID column, cannot update trajectory: {metrics_csv_path}")
        return

    dfm = dfm.copy()
    dfm["ID"] = dfm["ID"].astype(str)
    feature_cols = [f for f in METRICS_REGISTRY.keys() if f in dfm.columns]
    if not feature_cols:
        print(f"[TRAJ] No registry metric columns found in {metrics_csv_path}")
        return

    data = _load_json_dict(trajectory_path)
    data.setdefault("schema_version", 1)
    data["detector"] = str(detector)
    data.setdefault("feature_cols", list(METRICS_REGISTRY.keys()))
    data.setdefault("rounds_updated", [])
    data.setdefault("baseline_medians", {})
    data.setdefault("by_id", {})

    # Baseline medians are fitted once from round 0 and then reused.
    if int(round_idx) == 0 or not data.get("baseline_medians"):
        for feat in feature_cols:
            vals = pd.to_numeric(dfm[feat], errors="coerce")
            med = _json_safe_float(vals.median(skipna=True))
            data["baseline_medians"][feat] = med

    rkey = str(int(round_idx))
    for _, row in dfm.iterrows():
        sid = str(row["ID"])
        id_rec = data["by_id"].setdefault(sid, {"features": {}})
        feats = id_rec.setdefault("features", {})
        for feat in feature_cols:
            rec = feats.setdefault(feat, {})
            rec["baseline_median"] = data.get("baseline_medians", {}).get(feat)
            rec.setdefault("rounds", {})[rkey] = _json_safe_float(row.get(feat))

    rounds = {int(x) for x in data.get("rounds_updated", []) if str(x).lstrip("-").isdigit()}
    rounds.add(int(round_idx))
    data["rounds_updated"] = sorted(rounds)
    data["last_updated_round"] = int(round_idx)
    _save_json_dict(data, trajectory_path)
    print(f"[TRAJ] Updated trajectory round={round_idx} -> {trajectory_path}")


def _build_carry_forward_values(
    trajectory_path: str,
    current_round: int,
    feature_cols: list,
) -> dict:
    """For each ID/feature, return the most recent non-None metric value
    from rounds strictly before *current_round* (as recorded in the trajectory).

    Used to pre-fill NaN values in the metrics DataFrame before SHAP computation
    so the frozen imputer's round-0 medians are only used as a last resort.

    Returns {id_str: {feat_str: float}}.
    """
    data = _load_json_dict(trajectory_path)
    if not data or "by_id" not in data:
        return {}

    by_id = data.get("by_id", {})
    result: dict = {}

    for _id, id_rec in by_id.items():
        feats_rec = id_rec.get("features", {})
        id_vals: dict = {}
        for feat in feature_cols:
            if feat not in feats_rec:
                continue
            rounds_dict = feats_rec[feat].get("rounds", {})
            # Find the most recent non-None value strictly before current_round.
            best_val = None
            best_r = -1
            for rkey, val in rounds_dict.items():
                try:
                    r_int = int(rkey)
                except ValueError:
                    continue
                if r_int >= current_round:
                    continue
                if val is not None and r_int > best_r:
                    best_r = r_int
                    best_val = float(val)
            if best_val is not None:
                id_vals[feat] = best_val
        if id_vals:
            result[str(_id)] = id_vals

    return result


def _ensure_dir(p: str) -> None:
    # Idempotente e tollerante ai FileExistsError spuri di makedirs(exist_ok=True)
    # sul filesystem parallelo Lustre sotto carico concorrente.
    try:
        os.makedirs(p, exist_ok=True)
    except FileExistsError:
        if not os.path.isdir(p):
            raise

def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")

def _save_csv(df: pd.DataFrame, path: str) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False, encoding="utf-8")

def _require_file(path: str, msg: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{msg}: {path}")

def _normalize_id(df: pd.DataFrame) -> pd.DataFrame:
    if "ID" not in df.columns:
        raise ValueError("Missing column 'ID'")
    out = df.copy()
    out["ID"] = out["ID"].astype(str)
    return out

def _load_detector_df(detector_csv_path: str) -> pd.DataFrame:
    dfd = _load_csv(detector_csv_path)
    dfd["ID"] = dfd["ID"].astype(str)
    dfd["pred_label"] = dfd["pred_label"].astype(str).str.upper()
    dfd["confidence"] = pd.to_numeric(dfd["confidence"], errors="coerce")
    return dfd[["ID", "pred_label", "confidence"]].copy()

def _count_labels(det_df: pd.DataFrame):
    s = det_df["pred_label"].astype(str).str.upper()
    return int((s == "FAKE").sum()), int((s == "REAL").sum())

def _read_ids_from_csv(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return set()
    if "ID" not in df.columns:
        return set()
    return set(df["ID"].astype(str).tolist())

def _append_dedupe_by_id(existing_path: str, df_new: pd.DataFrame) -> None:
    df_new = df_new.copy()
    if "ID" in df_new.columns:
        df_new["ID"] = df_new["ID"].astype(str)

    if os.path.exists(existing_path) and os.path.getsize(existing_path) > 0:
        try:
            prev = pd.read_csv(existing_path, encoding="utf-8")
            if "ID" in prev.columns:
                prev["ID"] = prev["ID"].astype(str)
            df_new = pd.concat([prev, df_new], ignore_index=True)
            if "ID" in df_new.columns:
                df_new = df_new.drop_duplicates(subset=["ID"], keep="last")
        except Exception:
            pass

    _save_csv(df_new, existing_path)

def _missing_ids_against_csv(gen_csv_path: str, target_ids: set[str]) -> set[str]:
    if not os.path.exists(gen_csv_path) or os.path.getsize(gen_csv_path) == 0:
        return set(target_ids)
    try:
        df = pd.read_csv(gen_csv_path, encoding="utf-8")
    except Exception:
        return set(target_ids)
    if "ID" not in df.columns:
        return set(target_ids)
    done = set(df["ID"].astype(str).tolist())
    return set(target_ids) - done

def _generated_ids_from_csv(gen_csv_path: str) -> set[str]:
    if not os.path.exists(gen_csv_path) or os.path.getsize(gen_csv_path) == 0:
        return set()
    try:
        df = pd.read_csv(gen_csv_path, encoding="utf-8")
    except Exception:
        return set()
    if "ID" not in df.columns:
        return set()
    return set(df["ID"].astype(str).tolist())

def _load_panel_as_prev_df(panel_parquet_path: str) -> pd.DataFrame:
    _require_file(panel_parquet_path, "Missing panel parquet from previous round generation")
    dfp = pd.read_parquet(panel_parquet_path)
    if "ID" not in dfp.columns:
        raise ValueError(f"Panel parquet missing ID: {panel_parquet_path}")
    if "title_round_r" not in dfp.columns or "text_round_r" not in dfp.columns:
        raise ValueError(f"Panel parquet missing title_round_r/text_round_r: {panel_parquet_path}")
    out = dfp[["ID", "title_round_r", "text_round_r"]].copy()
    out["ID"] = out["ID"].astype(str)
    out = out.rename(columns={"title_round_r": "title", "text_round_r": "text"})
    return out


def _detector_csv_path(output_root: str, r: int, detector: str) -> str:
    """Canonical detector CSV location — single source of truth for every round."""
    return os.path.join(_det_dir(output_root, r, detector),
                        f"{DETECTOR_PREFIX}{detector}.csv")

def _ensure_detector_csv(output_root: str, r: int, detector: str) -> str:
    """Return the canonical detector CSV path, ensuring its parent dir exists."""
    p = _detector_csv_path(output_root, r, detector)
    _ensure_dir(os.path.dirname(p))
    return p


def _load_baseline_fake_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    df = _load_csv(path)
    if "ID" not in df.columns:
        return set()
    return set(df["ID"].astype(str).tolist())

def _save_baseline_fake_ids(ids: set[str], path: str):
    _ensure_dir(os.path.dirname(path) or ".")
    _save_csv(pd.DataFrame({"ID": sorted(list(ids))}), path)

def _init_or_load_ledger(ledger_path: str, baseline_fake_ids: set[str], detector_round0: pd.DataFrame | None) -> pd.DataFrame:
    if os.path.exists(ledger_path):
        led = _load_csv(ledger_path)
        led["ID"] = led["ID"].astype(str)
    else:
        led = pd.DataFrame(columns=[
            "ID",
            "status",
            "baseline_confidence",
            "last_pred_label",
            "last_confidence",
            "last_round_seen",
            "became_real_round",
            "rounds_to_real",
            "last_decision_round",
            "last_chosen_feature",
            "last_chosen_subtask",
        ])

    led = led.drop_duplicates(subset=["ID"], keep="last")
    existing = set(led["ID"].astype(str).tolist()) if not led.empty else set()
    missing = [i for i in baseline_fake_ids if i not in existing]

    base_conf_map = {}
    if detector_round0 is not None and not detector_round0.empty:
        tmp = detector_round0.copy()
        tmp["ID"] = tmp["ID"].astype(str)
        tmp["confidence"] = pd.to_numeric(tmp["confidence"], errors="coerce")
        base_conf_map = dict(zip(tmp["ID"], tmp["confidence"]))

    if missing:
        add = pd.DataFrame({
            "ID": missing,
            "status": ["ACTIVE_FAKE"] * len(missing),
            "baseline_confidence": [float(base_conf_map.get(i, np.nan)) for i in missing],
            "last_pred_label": ["FAKE"] * len(missing),
            "last_confidence": [float(base_conf_map.get(i, np.nan)) for i in missing],
            "last_round_seen": [0] * len(missing),
            "became_real_round": [np.nan] * len(missing),
            "rounds_to_real": [np.nan] * len(missing),
            "last_decision_round": [np.nan] * len(missing),
            "last_chosen_feature": [pd.NA] * len(missing),
            "last_chosen_subtask": [pd.NA] * len(missing),
        })
        led = pd.concat([led, add], ignore_index=True)

    led = led[led["ID"].astype(str).isin(set(baseline_fake_ids))].copy()
    led["ID"] = led["ID"].astype(str)
    led["status"] = led["status"].fillna("ACTIVE_FAKE").astype(str)

    if "last_chosen_feature" not in led.columns:
        led["last_chosen_feature"] = pd.Series([pd.NA] * len(led), dtype="string")
    else:
        led["last_chosen_feature"] = led["last_chosen_feature"].astype("string")

    if "last_chosen_subtask" not in led.columns:
        led["last_chosen_subtask"] = pd.Series([pd.NA] * len(led), dtype="string")
    else:
        led["last_chosen_subtask"] = led["last_chosen_subtask"].astype("string")

    return led

def _save_ledger(ledger_path: str, led: pd.DataFrame) -> None:
    _ensure_dir(os.path.dirname(ledger_path) or ".")
    _save_csv(led, ledger_path)


def _record_decisions_only(led: pd.DataFrame, decisions_df: pd.DataFrame, decision_round: int) -> pd.DataFrame:
    if decisions_df is None or decisions_df.empty:
        return led

    tmp = decisions_df.copy()
    tmp["ID"] = tmp["ID"].astype(str)

    for c in ["ID", "chosen_feature", "chosen_subtask"]:
        if c not in tmp.columns:
            raise ValueError(f"Missing '{c}' in orchestrator decisions.")

    tmp = tmp[["ID", "chosen_feature", "chosen_subtask"]].drop_duplicates(subset=["ID"], keep="last")

    out = led.copy()
    out["ID"] = out["ID"].astype(str)

    for col in ["last_chosen_feature", "last_chosen_subtask"]:
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), dtype="string")
        else:
            out[col] = out[col].astype("string")

    if "last_decision_round" not in out.columns:
        out["last_decision_round"] = np.nan
    out["last_decision_round"] = pd.to_numeric(out["last_decision_round"], errors="coerce")

    m = out.merge(tmp, on="ID", how="left", suffixes=("", "_new"))

    m["chosen_feature"] = m["chosen_feature"].astype("string")
    m["chosen_subtask"] = m["chosen_subtask"].astype("string")

    has = (
        m["chosen_feature"].notna()
        & (m["chosen_feature"].str.strip() != "")
        & (m["chosen_feature"].str.lower() != "nan")
    )

    m.loc[has, "last_chosen_feature"] = m.loc[has, "chosen_feature"]
    m.loc[has, "last_chosen_subtask"] = m.loc[has, "chosen_subtask"]
    m.loc[has, "last_decision_round"] = float(decision_round)

    m = m.drop(columns=["chosen_feature", "chosen_subtask"])
    return m




def _call_orchestrator_iterative_adaptive(**context) -> pd.DataFrame:
    fn = run_orchestrator_round_iterative
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    kwargs = {k: v for k, v in context.items() if k in allowed}
    return fn(**kwargs)


def _ensure_round0_signals_and_decisions(
    OUTPUT_ROOT: str,
    DETECTOR: str,
    df_real: pd.DataFrame,
    BASELINE_PIPE_PATH: str,
    BASELINE_COEFFS_PATH: str,
    baseline_fake_ids: set[str],
    RESUME: bool,
):
    r0 = 0
    rd_round0 = _round_dir(OUTPUT_ROOT, r0)
    rd_gen0_root = _gen_root_dir(OUTPUT_ROOT, r0)
    rd_gen0 = _gen_dir(OUTPUT_ROOT, r0)
    rd_det0 = _det_dir(OUTPUT_ROOT, r0, DETECTOR)
    rd_orch0 = _orch_dir(OUTPUT_ROOT, r0, DETECTOR)
    # round-0 SHAP lives in Output/meta_frozen/<det>/ (frozen baseline).
    # Only round>0 rounds populate rounds/round_NNN/shap/<det>/.
    meta_frozen_dir = _meta_frozen_dir(OUTPUT_ROOT, DETECTOR)

    _ensure_dir(rd_round0)
    _ensure_dir(rd_gen0_root)
    _ensure_dir(rd_gen0)
    _ensure_dir(rd_det0)
    _ensure_dir(rd_orch0)

    decisions_path0 = os.path.join(rd_orch0, "orch_decisions_with_reasons.csv")
    if os.path.exists(decisions_path0) and RESUME:
        print(f"[BOOT] Found round0 decisions -> {decisions_path0} (skip)")
        return

    gen0_path = os.path.join(rd_gen0, "Generated_Fake_News.csv")
    det0_path = os.path.join(rd_det0, f"{DETECTOR_PREFIX}{DETECTOR}.csv")
    _require_file(gen0_path, "Missing round0 generation/Generated_Fake_News.csv")
    _require_file(det0_path, "Missing round0 detector output")

    metrics0_path = os.path.join(rd_round0, METRICS_CSV_NAME)
    if (not os.path.exists(metrics0_path)) or (not RESUME):
        gen0_ids = _generated_ids_from_csv(gen0_path)
        scope_ids = sorted(list(set(baseline_fake_ids) & set(gen0_ids)))
        if len(scope_ids) == 0:
            raise RuntimeError("[BOOT] round0 scope_ids empty: cannot compute metrics/SHAP for decisions.")
        df_orig0 = df_real[df_real["ID"].astype(str).isin(scope_ids)].copy()
        print(f"[BOOT][METRICS] Computing round0 metrics for decisions (n={len(scope_ids)}) -> {metrics0_path}")
        compute_agent_metrics(
            df_orig=df_orig0,
            base_out_dir=rd_round0,
            ids_mode="csv_in_dataset",
            out_csv_name=METRICS_CSV_NAME,
            residualizer_path=os.path.join(OUTPUT_ROOT, "meta_frozen", "residualizer.json"),
            baseline_references_path=os.path.join(OUTPUT_ROOT, "meta_frozen", "baseline_references.json"),
        )
    else:
        print(f"[BOOT][METRICS] Found round0 metrics -> {metrics0_path}")

    trajectory_path = _trajectory_path(OUTPUT_ROOT, DETECTOR)
    _update_trajectory_json(trajectory_path, metrics0_path, round_idx=0, detector=DETECTOR)

    shap0_csv = os.path.join(meta_frozen_dir, "shap_attributions_per_id.csv")
    if (os.path.exists(shap0_csv) and RESUME):
        print(f"[BOOT][SHAP] Found round0 SHAP -> {shap0_csv} (skip)")
    else:
        print(f"[BOOT][SHAP] Computing round0 SHAP (frozen baseline) -> {shap0_csv}")
        compute_shap_with_frozen_baseline(
            baseline_pipe_path=BASELINE_PIPE_PATH,
            round_dir=rd_round0,
            detector_name=DETECTOR,
            out_dir=meta_frozen_dir,
            metrics_csv_name=METRICS_CSV_NAME,
            detector_csv_prefix=DETECTOR_PREFIX,
            target="label",
        )

    print("[BOOT][ORCH] Creating round0 decisions for round1...")
    context = dict(
        detector_name=DETECTOR,
        round_dir=rd_round0,
        detector_csv_prefix=DETECTOR_PREFIX,
        filter_fake=True,
        shap_dir=meta_frozen_dir,
        shap_filename="shap_attributions_per_id.csv",
        baseline_coeffs_path=BASELINE_COEFFS_PATH,
        trajectory_path=trajectory_path,
        invert_direction=_INVERT_DIRECTION,
        anti_feature=_ANTI_FEATURE,
    )
    dec0 = _call_orchestrator_iterative_adaptive(**context)
    if isinstance(dec0, pd.DataFrame) and (not dec0.empty):
        dec0["ID"] = dec0["ID"].astype(str)
        _save_csv(dec0, decisions_path0)
        print(f"[BOOT][ORCH] Saved -> {decisions_path0}")
    else:
        raise RuntimeError("[BOOT][ORCH] Orchestrator returned empty decisions for round0; cannot run round1 generation.")


# Framework-validity ablation flags (exp2/exp3). Set once per loop run in _run()
# and read by both orchestrator call sites (round-0 boot + per-round). Module-level
# because _ensure_round0_signals_and_decisions builds its own orchestrator context.
_INVERT_DIRECTION: bool = False
_ANTI_FEATURE: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run iterative rounds loop (NEW strategy: next round IDs = FAKE among generated)")
    parser.add_argument("--nomedataset", type=str, required=True, help="Path dataset CSV con colonna ID")
    parser.add_argument("--nomedetector", type=str, default=os.getenv("DETECTOR", "valhalla"), help="Detector name")
    parser.add_argument("--output_root", type=str, default="Output", help="Output root (default: Output)")
    parser.add_argument("--max_round", type=int, default=0, help="Stop after this round number (0 = no limit)")
    parser.add_argument("--resume", action="store_true", help="Resume mode (default True)")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume mode")
    parser.add_argument("--min_active_fake", type=int, default=DEFAULT_MIN_ACTIVE_FAKE,
                        help=f"Minimum ACTIVE_FAKE per round to keep iterating (default: {DEFAULT_MIN_ACTIVE_FAKE}). "
                             f"Use a smaller value (e.g. 10) for small 'test mini' cohorts.")
    parser.add_argument("--no_drift_retrain", action="store_true",
                        help="Disable drift monitoring AND SGL retrain entirely. The loop still "
                             "scores/edits articles and tracks REAL flips, but skips fake-buffer "
                             "accumulation, drift_report.json, and any meta-model retrain.")
    parser.add_argument("--invert_direction", action="store_true",
                        help="exp2 ablation: keep the orchestrator's chosen feature but FLIP the "
                             "edit direction handed to agents (INCREASE<->DECREASE).")
    parser.add_argument("--anti_feature", action="store_true",
                        help="exp3 ablation: select the LEAST important eligible feature (min |SHAP|) "
                             "instead of the most important; direction stays normal.")
    return parser

def _run(args) -> int:

    DETECTOR = args.nomedetector
    DATASET_PATH = args.nomedataset
    OUTPUT_ROOT = args.output_root
    MAX_ROUND = int(getattr(args, "max_round", 0) or 0)
    MIN_ACTIVE_FAKE_LOCAL = int(getattr(args, "min_active_fake", DEFAULT_MIN_ACTIVE_FAKE) or DEFAULT_MIN_ACTIVE_FAKE)
    ENABLE_DRIFT_RETRAIN = not bool(getattr(args, "no_drift_retrain", False))

    global _INVERT_DIRECTION, _ANTI_FEATURE
    _INVERT_DIRECTION = bool(getattr(args, "invert_direction", False))
    _ANTI_FEATURE = bool(getattr(args, "anti_feature", False))
    if _INVERT_DIRECTION or _ANTI_FEATURE:
        print(f"[LOOP][CONFIG] ABLATION invert_direction={_INVERT_DIRECTION} "
              f"anti_feature={_ANTI_FEATURE} (framework-validity experiment)")
    if not ENABLE_DRIFT_RETRAIN:
        print("[LOOP][CONFIG] --no_drift_retrain: drift monitoring + SGL retrain DISABLED "
              "(no fake-buffer accumulation, no drift_report.json, no meta-model retrain).")

    RESUME = True
    if args.no_resume:
        RESUME = False
    elif args.resume:
        RESUME = True

    os.environ["DETECTOR"] = DETECTOR
    os.environ["DATASET_PATH"] = DATASET_PATH

    TRACK_DIR = _tracking_dir(OUTPUT_ROOT, DETECTOR)
    _ensure_dir(TRACK_DIR)

    BASELINE_FAKE_IDS_PATH = os.path.join(TRACK_DIR, "baseline_fake_ids.csv")
    LEDGER_PATH = os.path.join(TRACK_DIR, "baseline_fakes_ledger.csv")
    FINAL_REPORT_PATH = os.path.join(TRACK_DIR, "final_report_baseline_fakes.csv")
    TRAJECTORY_PATH = _trajectory_path(OUTPUT_ROOT, DETECTOR)

    print(f"[INFO] DETECTOR={DETECTOR}")
    print(f"[INFO] DATASET={DATASET_PATH}")
    print(f"[INFO] OUTPUT_ROOT={OUTPUT_ROOT}")
    print(f"[INFO] RESUME={RESUME}")

    meta_frozen = _meta_frozen_dir(OUTPUT_ROOT, DETECTOR)
    BASELINE_PIPE_PATH = os.path.join(meta_frozen, "meta_model_pipe.pkl")
    BASELINE_COEFFS_PATH = os.path.join(meta_frozen, "meta_model_coefficients.csv")
    _FEATURE_META_PATH = os.path.join(meta_frozen, "feature_meta.json")

    _require_file(BASELINE_PIPE_PATH, "Missing frozen baseline meta-model pipe (run round0 baseline first)")
    _require_file(BASELINE_COEFFS_PATH, "Missing frozen baseline coefficients (run round0 baseline first)")

    # §11 — load DDM fidelity baseline from feature_meta.json
    _ddm_e_min: float = 0.0
    _ddm_sigma_min: float = 0.0
    if os.path.exists(_FEATURE_META_PATH):
        try:
            with open(_FEATURE_META_PATH, "r", encoding="utf-8") as _fm:
                _feat_meta = json.load(_fm)
            _fb = _feat_meta.get("fidelity_baseline", {})
            if _fb:
                _ddm_e_min = float(_fb.get("e_min", 0.0))
                _ddm_sigma_min = float(_fb.get("sigma_min", 0.0))
                print(
                    f"[DDM] Loaded fidelity baseline: e_min={_ddm_e_min:.4f} "
                    f"sigma_min={_ddm_sigma_min:.4f}"
                )
        except Exception as _e_meta:
            print(f"[DDM][WARN] Could not load fidelity_baseline from feature_meta.json: {_e_meta}")

    _require_file(DATASET_PATH, "Dataset not found")
    df_real = _normalize_id(_load_csv(DATASET_PATH))
    print(f"[DATASET] rows={len(df_real)} path={DATASET_PATH}")

    round0_gen = _gen_dir(OUTPUT_ROOT, 0)
    round0_det = _det_dir(OUTPUT_ROOT, 0, DETECTOR)

    _require_file(os.path.join(round0_gen, "Generated_Fake_News.csv"), "Missing round0 generation/Generated_Fake_News.csv")
    _require_file(os.path.join(round0_det, f"{DETECTOR_PREFIX}{DETECTOR}.csv"), "Missing round0 detector output")
    _require_file(os.path.join(round0_gen, "panel_articles_round_r.parquet"), "Missing round0 panel parquet")

    det0 = _load_detector_df(os.path.join(round0_det, f"{DETECTOR_PREFIX}{DETECTOR}.csv"))

    baseline_fake_ids = _load_baseline_fake_ids(BASELINE_FAKE_IDS_PATH)
    fakes_from_det0 = set(det0[det0["pred_label"] == "FAKE"]["ID"].astype(str).tolist())

    if not baseline_fake_ids:
        baseline_fake_ids = set(fakes_from_det0)
        _save_baseline_fake_ids(baseline_fake_ids, BASELINE_FAKE_IDS_PATH)
        print(f"[BASELINE] Saved baseline_fake_ids: {len(baseline_fake_ids)}")
    else:
        merged = set(baseline_fake_ids) | set(fakes_from_det0)
        if merged != baseline_fake_ids:
            baseline_fake_ids = merged
            _save_baseline_fake_ids(baseline_fake_ids, BASELINE_FAKE_IDS_PATH)
            print(f"[BASELINE] Updated baseline_fake_ids (union): {len(baseline_fake_ids)}")
        else:
            print(f"[BASELINE] Loaded baseline_fake_ids: {len(baseline_fake_ids)}")

    led = _init_or_load_ledger(LEDGER_PATH, baseline_fake_ids, det0)
    _save_ledger(LEDGER_PATH, led)
    print(f"[LEDGER] rows={len(led)} -> {LEDGER_PATH}")

    _ensure_round0_signals_and_decisions(
        OUTPUT_ROOT=OUTPUT_ROOT,
        DETECTOR=DETECTOR,
        df_real=df_real,
        BASELINE_PIPE_PATH=BASELINE_PIPE_PATH,
        BASELINE_COEFFS_PATH=BASELINE_COEFFS_PATH,
        baseline_fake_ids=baseline_fake_ids,
        RESUME=RESUME,
    )

    r = 1
    active_ids_in = sorted(list(baseline_fake_ids))
    # PUNTO 4 — retrain cooldown bookkeeping
    _last_retrain_round = -RETRAIN_COOLDOWN_ROUNDS

    # Resume fast-forward: avoid iterating through all completed rounds from r=1.
    # Find the minimum last_round_seen among ACTIVE_FAKE IDs; if ≥1 and the
    # fake_ids_out_next_round.csv exists for that round, jump directly to r+1.
    if RESUME:
        active_only_led = led[led["status"].astype(str) == "ACTIVE_FAKE"]
        if not active_only_led.empty:
            last_seen_vals = pd.to_numeric(active_only_led["last_round_seen"], errors="coerce").dropna()
            if not last_seen_vals.empty:
                min_last_seen = int(last_seen_vals.min())
                if min_last_seen >= 1:
                    ff_gen = _gen_dir(OUTPUT_ROOT, min_last_seen, DETECTOR)
                    ff_fake_path = os.path.join(ff_gen, "fake_ids_out_next_round.csv")
                    if os.path.exists(ff_fake_path):
                        ff_fakes = _read_ids_from_csv(ff_fake_path)
                        ff_active = [i for i in sorted(ff_fakes) if str(i) in set(active_only_led["ID"].astype(str))]
                        if ff_active:
                            active_ids_in = ff_active
                            r = min_last_seen + 1
                            print(f"[RESUME] Fast-forward to round r={r} "
                                  f"(last_round_seen for ACTIVE_FAKE IDs = {min_last_seen})")

    while True:
        if MAX_ROUND and r > MAX_ROUND:
            print(f"[STOP] Reached max_round={MAX_ROUND} -> loop ends.")
            break

        print("\n" + "=" * 95)
        print(f"[ROUND {r}] INPUT IDs (active_ids_in) = {len(active_ids_in)}")
        print("=" * 95)

        if len(active_ids_in) == 0:
            print("[STOP] active_ids_in empty -> loop ends.")
            break

        if len(active_ids_in) < MIN_ACTIVE_FAKE_LOCAL:
            stop_marker = os.path.join(
                OUTPUT_ROOT, "rounds", f"round_{r:03d}",
                "orchestrator", DETECTOR, "STOP_INSUFFICIENT_ACTIVE.txt",
            )
            os.makedirs(os.path.dirname(stop_marker), exist_ok=True)
            with open(stop_marker, "w", encoding="utf-8") as _f:
                _f.write(
                    f"round={r} detector={DETECTOR} "
                    f"n_active_fake={len(active_ids_in)} threshold={MIN_ACTIVE_FAKE_LOCAL}\n"
                    "Loop terminated: insufficient ACTIVE_FAKE for reliable drift/SGL inference.\n"
                )
            print(
                f"[STOP] n_active_fake={len(active_ids_in)} < MIN_ACTIVE_FAKE_LOCAL={MIN_ACTIVE_FAKE_LOCAL} "
                f"-> loop ends. Marker: {stop_marker}"
            )
            break

        rd_round = _round_dir(OUTPUT_ROOT, r)
        rd_gen_root = _gen_root_dir(OUTPUT_ROOT, r)
        rd_gen = _gen_dir(OUTPUT_ROOT, r, DETECTOR)
        rd_det = _det_dir(OUTPUT_ROOT, r, DETECTOR)
        rd_orch = _orch_dir(OUTPUT_ROOT, r, DETECTOR)
        rd_shap = _shap_dir(OUTPUT_ROOT, r, DETECTOR)

        prev_r = r - 1
        prev_gen = _gen_dir(OUTPUT_ROOT, prev_r, None if prev_r == 0 else DETECTOR)
        prev_orch_dir = _orch_dir(OUTPUT_ROOT, prev_r, DETECTOR)

        _ensure_dir(rd_round)
        _ensure_dir(rd_gen_root)
        _ensure_dir(rd_gen)
        _ensure_dir(rd_det)
        _ensure_dir(rd_orch)
        _ensure_dir(rd_shap)

        prev_panel_path = os.path.join(prev_gen, "panel_articles_round_r.parquet")
        df_prev_panel = _load_panel_as_prev_df(prev_panel_path)

        df_prev_active = df_prev_panel[df_prev_panel["ID"].isin(set(active_ids_in))].copy()
        df_real_active_in = df_real[df_real["ID"].isin(set(active_ids_in))].copy()
        if df_real_active_in.empty:
            print("[WARN] df_real_active_in empty -> stop.")
            break

        ids_path = os.path.join(rd_gen, "active_ids_in.csv")
        _save_csv(pd.DataFrame({"ID": active_ids_in}), ids_path)

        gen_path = os.path.join(rd_gen, "Generated_Fake_News.csv")
        missing_ids = _missing_ids_against_csv(gen_path, set(active_ids_in)) if RESUME else set(active_ids_in)
        print(f"[GEN-CHECK] round {r}: missing={len(missing_ids)} done={len(set(active_ids_in)-missing_ids)} target={len(active_ids_in)}")

        if (not RESUME) or (len(missing_ids) > 0):
            print(f"[GEN] Running pipeline round {r} on INPUT IDs (n={len(active_ids_in)}), resume-missing implied.")
            run_generation_pipeline(
                df_input=df_real_active_in,
                base_out_dir=rd_gen,
                sample_size=None,
                sample_ids_path=ids_path,
                resume=True,
                df_prev=df_prev_active,
                prev_generation_dir=prev_gen,
                prev_orch_dir=prev_orch_dir,
                detector_name=DETECTOR,
            )
        else:
            print(f"[GEN] Generated already complete for round {r} INPUT IDs -> skip pipeline.")

        if not os.path.exists(gen_path):
            print(f"[WARN] Generated_Fake_News.csv not found after generation (all IDs may have failed/been skipped). "
                  f"Treating as empty -> will hit len(generated_in_scope)==0 check below.")
            # Do NOT break here: fall through to the generated_in_scope==0 check
            # which will break gracefully with a proper log message.

        generated_ids = sorted(list(_generated_ids_from_csv(gen_path)))
        generated_ids_set = set(generated_ids)
        generated_in_scope = sorted(list(set(active_ids_in) & generated_ids_set))
        print(f"[GEN] round {r}: generated_success={len(generated_in_scope)} / input={len(active_ids_in)}")

        if len(generated_in_scope) == 0:
            print("[STOP] No generated successes -> cannot run detector/SHAP/orch. End loop.")
            break

        df_gen = _normalize_id(_load_csv(gen_path))
        df_gen = df_gen[df_gen["ID"].isin(set(generated_in_scope))].copy()

        metrics_path = os.path.join(rd_gen, METRICS_CSV_NAME)
        missing_metrics_ids = set(generated_in_scope) - _read_ids_from_csv(metrics_path)
        if (not os.path.exists(metrics_path)) or (not RESUME):
            df_real_for_metrics = df_real[df_real["ID"].isin(set(generated_in_scope))].copy()
            print(f"[METRICS] Computing (generated-only) n={len(df_real_for_metrics)} -> {metrics_path}")
            compute_agent_metrics(
                df_orig=df_real_for_metrics,
                base_out_dir=rd_gen,
                ids_mode="csv_in_dataset",
                out_csv_name=METRICS_CSV_NAME,
                residualizer_path=os.path.join(OUTPUT_ROOT, "meta_frozen", "residualizer.json"),
                baseline_references_path=os.path.join(OUTPUT_ROOT, "meta_frozen", "baseline_references.json"),
            )
        elif len(missing_metrics_ids) > 0:
            df_real_for_metrics = df_real[df_real["ID"].isin(set(missing_metrics_ids))].copy()
            print(f"[METRICS][FIX2] Backfill missing metrics IDs={len(missing_metrics_ids)} -> append")
            compute_agent_metrics(
                df_orig=df_real_for_metrics,
                base_out_dir=rd_gen,
                ids_mode="csv_in_dataset",
                out_csv_name=METRICS_CSV_NAME,
                only_ids=sorted(list(missing_metrics_ids)),
                append_if_exists=True,
                residualizer_path=os.path.join(OUTPUT_ROOT, "meta_frozen", "residualizer.json"),
                baseline_references_path=os.path.join(OUTPUT_ROOT, "meta_frozen", "baseline_references.json"),
            )
        else:
            print(f"[METRICS] Found -> {metrics_path}")

        if not os.path.exists(metrics_path):
            print("[STOP] Missing metrics -> cannot compute SHAP/orchestrator. End loop.")
            break

        _update_trajectory_json(TRAJECTORY_PATH, metrics_path, round_idx=r, detector=DETECTOR)

        det_path_canonical = _ensure_detector_csv(OUTPUT_ROOT, r, DETECTOR)

        missing_det_ids = set(generated_in_scope) - _read_ids_from_csv(det_path_canonical)
        if (not os.path.exists(det_path_canonical)) or (not RESUME):
            print(f"[DETECT] Running detector '{DETECTOR}' round {r} on generated-only (n={len(df_gen)}) -> {det_path_canonical}")
            dfr = run_detector_on_df(detector=DETECTOR, df=df_gen, device=None)
            dfr = _normalize_id(dfr)
            _save_csv(dfr, det_path_canonical)
        elif len(missing_det_ids) > 0:
            print(f"[DETECT][FIX2] Backfill missing detector IDs={len(missing_det_ids)} -> append")
            df_sub = df_gen[df_gen["ID"].astype(str).isin(set(missing_det_ids))].copy()
            dfr_new = run_detector_on_df(detector=DETECTOR, df=df_sub, device=None)
            dfr_new = _normalize_id(dfr_new)
            _append_dedupe_by_id(det_path_canonical, dfr_new)
            dfr = _load_detector_df(det_path_canonical)
        else:
            print(f"[DETECT] Found -> {det_path_canonical} (skip)")
            dfr = _load_detector_df(det_path_canonical)

        # PUNTO 4 — accumulate FAKE entries from this round into the retraining buffer.
        if ENABLE_DRIFT_RETRAIN:
            try:
                _retrain_buf = _retrain_buffer_dir(OUTPUT_ROOT, DETECTOR)
                _n_buf = update_fake_accumulated(
                    buffer_dir=_retrain_buf,
                    current_round=r,
                    round_metrics_csv=metrics_path,
                    round_detector_csv=det_path_canonical,
                )
                print(f"[BUFFER] fake_accumulated size after round {r}: {_n_buf}")
            except Exception as _e_buf:
                print(f"[BUFFER][WARN] update_fake_accumulated failed: {_e_buf}")

        dfr["ID"] = dfr["ID"].astype(str)
        fake_ids_out = sorted(dfr[dfr["pred_label"].astype(str).str.upper() == "FAKE"]["ID"].astype(str).tolist())
        real_ids_out = sorted(dfr[dfr["pred_label"].astype(str).str.upper() == "REAL"]["ID"].astype(str).tolist())

        n_fake, n_real = len(fake_ids_out), len(real_ids_out)
        print(f"[DETECT] round {r}: FAKE={n_fake} | REAL={n_real} (generated-only)")

        _save_csv(pd.DataFrame({"ID": fake_ids_out}), os.path.join(rd_gen, "fake_ids_out_next_round.csv"))
        _save_csv(pd.DataFrame({"ID": generated_in_scope}), os.path.join(rd_gen, "generated_ids_in_scope.csv"))

        # Carry-forward: use the most recent valid per-ID metric value from
        # the trajectory before falling back to the frozen imputer medians.
        _carry_forward = _build_carry_forward_values(TRAJECTORY_PATH, r, list(METRICS_REGISTRY.keys()))
        if _carry_forward:
            _cf_ids = len(_carry_forward)
            _cf_slots = sum(len(v) for v in _carry_forward.values())
            print(f"[CARRY-FWD] round={r}: {_cf_slots} values available for {_cf_ids} IDs")

        shap_csv = os.path.join(rd_shap, "shap_attributions_per_id.csv")
        missing_shap_ids = set(generated_in_scope) - _read_ids_from_csv(shap_csv)
        if os.path.exists(shap_csv) and RESUME and len(missing_shap_ids) == 0:
            print(f"[SHAP] Found -> {shap_csv} (skip)")
        elif os.path.exists(shap_csv) and RESUME and len(missing_shap_ids) > 0:
            print(f"[SHAP][FIX2] Backfill missing SHAP IDs={len(missing_shap_ids)} -> append")
            compute_shap_with_frozen_baseline(
                baseline_pipe_path=BASELINE_PIPE_PATH,
                round_dir=rd_gen,
                detector_name=DETECTOR,
                out_dir=rd_shap,
                metrics_csv_name=METRICS_CSV_NAME,
                detector_csv_prefix=DETECTOR_PREFIX,
                target="label",
                only_ids=sorted(list(missing_shap_ids)),
                append_if_exists=True,
                carry_forward_values=_carry_forward or None,
            )
            print(f"[SHAP] Frozen-baseline SHAP saved -> {shap_csv}")
        else:
            compute_shap_with_frozen_baseline(
                baseline_pipe_path=BASELINE_PIPE_PATH,
                round_dir=rd_gen,
                detector_name=DETECTOR,
                out_dir=rd_shap,
                metrics_csv_name=METRICS_CSV_NAME,
                detector_csv_prefix=DETECTOR_PREFIX,
                target="label",
                carry_forward_values=_carry_forward or None,
            )
            print(f"[SHAP] Frozen-baseline SHAP saved -> {shap_csv}")

        # Drift monitoring + SGL retrain — gated by --no_drift_retrain.
        if ENABLE_DRIFT_RETRAIN:
            _drift_report_path = os.path.join(rd_round, "drift_report.json")
            _metrics0_path = os.path.join(_gen_dir(OUTPUT_ROOT, 0), METRICS_CSV_NAME)
            _shap0_path = os.path.join(_shap_dir(OUTPUT_ROOT, 0, DETECTOR), "shap_attributions_per_id.csv")
            _det0_path = os.path.join(round0_det, f"{DETECTOR_PREFIX}{DETECTOR}.csv")
            _retrain_required, _drift_rpt = compute_drift_report(
                metrics0_path=_metrics0_path,
                metrics_t_path=metrics_path,
                shap0_path=_shap0_path,
                shap_t_path=shap_csv,
                feature_cols=list(METRICS_REGISTRY.keys()),
                round_idx=r,
                detector=DETECTOR,
                out_path=_drift_report_path,
                resume=RESUME,
                pipe_path=BASELINE_PIPE_PATH,
                det0_path=_det0_path,
                det_t_path=det_path_canonical,
                e_min=_ddm_e_min,
                sigma_min=_ddm_sigma_min,
                min_active_fake=MIN_ACTIVE_FAKE_LOCAL,
            )
            # §11 — update (e_min, sigma_min) from report and persist to feature_meta.json
            _sf = _drift_rpt.get("surrogate_fidelity", {})
            _new_e_min = _sf.get("e_min")
            _new_sigma_min = _sf.get("sigma_min")
            if isinstance(_new_e_min, float) and isinstance(_new_sigma_min, float):
                if _new_e_min != _ddm_e_min or _new_sigma_min != _ddm_sigma_min:
                    _ddm_e_min = _new_e_min
                    _ddm_sigma_min = _new_sigma_min
                    try:
                        if os.path.exists(_FEATURE_META_PATH):
                            with open(_FEATURE_META_PATH, "r", encoding="utf-8") as _fmr:
                                _meta_data = json.load(_fmr)
                        else:
                            _meta_data = {}
                        if "fidelity_baseline" not in _meta_data:
                            _meta_data["fidelity_baseline"] = {}
                        _meta_data["fidelity_baseline"]["e_min"] = _ddm_e_min
                        _meta_data["fidelity_baseline"]["sigma_min"] = _ddm_sigma_min
                        _tmp_fd, _tmp_path = tempfile.mkstemp(
                            dir=os.path.dirname(os.path.abspath(_FEATURE_META_PATH)),
                            suffix=".tmp",
                        )
                        try:
                            with os.fdopen(_tmp_fd, "w", encoding="utf-8") as _fmw:
                                json.dump(_meta_data, _fmw, indent=2, ensure_ascii=False)
                            os.replace(_tmp_path, _FEATURE_META_PATH)
                        except Exception:
                            try:
                                os.unlink(_tmp_path)
                            except OSError:
                                pass
                            raise
                    except Exception as _e_persist:
                        print(f"[DDM][WARN] Could not persist updated e_min/sigma_min: {_e_persist}")
            _fd = _drift_rpt.get("feature_drift", {})
            _sa = _drift_rpt.get("surrogate_fidelity", _drift_rpt.get("surrogate_alignment", {}))
            _overall = _drift_rpt.get("overall", {})
            _overall_action = str(_overall.get("action", "OK"))

            if _fd.get("status") in ("WARNING", "STOP"):
                _drifted_feats = [
                    t["feature"] for t in _fd.get("tests", []) if t.get("drifted")
                ]
                print(
                    f"[DRIFT][{_fd.get('status')}] round={r}: feature drift detected "
                    f"({len(_drifted_feats)} features) — diagnostic, loop continues."
                )
            if _sa.get("status") in ("INSUFFICIENT_SAMPLE", "INSUFFICIENT_SAMPLE_TAIL"):
                print(
                    f"[DRIFT] round={r}: surrogate fidelity insufficient sample "
                    f"(n_r={_sa.get('n_r', _sa.get('n', '?'))}) — loop continues."
                )

            # Three-state drift action (replaces legacy STOP branch).
            # - OK / WARNING: loop continues.
            # - RETRAIN: SGL retraining is triggered (PUNTO 4). Until PUNTO 4 lands,
            #   this is a no-op print so the loop keeps progressing instead of dying.
            if _overall_action == "RETRAIN" or _retrain_required:
                print(
                    f"[DRIFT][RETRAIN] round={r}: KS+DDM convergent — surrogate retrain triggered. "
                    f"{_overall.get('reason', '')}"
                )
                if (r - _last_retrain_round) < RETRAIN_COOLDOWN_ROUNDS:
                    print(
                        f"[DRIFT][RETRAIN] cooldown active "
                        f"(last={_last_retrain_round}, current={r}, "
                        f"cooldown={RETRAIN_COOLDOWN_ROUNDS}) — skip"
                    )
                else:
                    try:
                        _retrain_result = do_retrain_and_reload(
                            output_root=OUTPUT_ROOT,
                            detector=DETECTOR,
                            current_round=r,
                        )
                        _status = _retrain_result.get("status", "?")
                        _summary_parts = [f"status={_status}"]
                        for _k in ("n_real", "n_fake", "n_fake_pre_balance",
                                   "n_fake_post_balance", "n_total"):
                            _v = _retrain_result.get(_k)
                            if _v is not None:
                                _summary_parts.append(f"{_k}={_v}")
                        if _status == "OK":
                            _n_flipped = len(_retrain_result.get("beta_sign_changes", []) or [])
                            _summary_parts.append(f"n_flipped={_n_flipped}")
                        print("[RETRAIN] result: " + " ".join(_summary_parts))
                        if _retrain_result.get("status") == "OK":
                            _last_retrain_round = r
                            # Reload e_min/sigma_min from updated feature_meta.json
                            try:
                                if os.path.exists(_FEATURE_META_PATH):
                                    with open(_FEATURE_META_PATH, "r", encoding="utf-8") as _fm:
                                        _new_fm = json.load(_fm)
                                    _fb = _new_fm.get("fidelity_baseline", {})
                                    if _fb:
                                        _ddm_e_min = float(_fb.get("e_min", _ddm_e_min))
                                        _ddm_sigma_min = float(_fb.get("sigma_min", _ddm_sigma_min))
                                        print(f"[DDM] Reloaded fidelity_baseline post-retrain: "
                                              f"e_min={_ddm_e_min:.4f} sigma_min={_ddm_sigma_min:.4f}")
                            except Exception as _e_rl:
                                print(f"[DDM][WARN] could not reload fidelity_baseline post-retrain: {_e_rl}")
                    except Exception as _e_retrain:
                        print(f"[RETRAIN][ERROR] do_retrain_and_reload failed: {_e_retrain}")
                # Loop continues regardless.
            elif _overall_action == "WARNING":
                print(f"[DRIFT][WARN] round={r}: {_overall.get('reason', '')}")
            # No drift-driven `break` here — only max_round / ACTIVE_FAKE==0 stop the loop.

        pred_map = dict(zip(dfr["ID"].astype(str), dfr["pred_label"].astype(str)))
        conf_map = dict(zip(dfr["ID"].astype(str), pd.to_numeric(dfr["confidence"], errors="coerce")))

        led = led.copy()
        led["ID"] = led["ID"].astype(str)

        for _id in generated_in_scope:
            ix = led["ID"] == str(_id)
            if not ix.any():
                continue

            pl = str(pred_map.get(str(_id), "")).upper()
            cf = float(conf_map.get(str(_id), np.nan)) if conf_map.get(str(_id), None) is not None else np.nan

            led.loc[ix, "last_pred_label"] = pl
            led.loc[ix, "last_confidence"] = cf
            led.loc[ix, "last_round_seen"] = float(r)

            if pl == "REAL":
                _was_first_flip = pd.isna(led.loc[ix, "became_real_round"]).all()
                led.loc[ix, "status"] = "DONE_REAL"
                if _was_first_flip:
                    led.loc[ix, "became_real_round"] = float(r)
                    led.loc[ix, "rounds_to_real"] = float(r)
                    # Post-flip REAL adversarial signal feeds the retrain buffer only.
                    if ENABLE_DRIFT_RETRAIN:
                        try:
                            from .retraining_buffer import append_done_real_post_flip, _buffer_dir
                            append_done_real_post_flip(
                                buffer_dir=_buffer_dir(OUTPUT_ROOT, DETECTOR),
                                article_id=str(_id),
                                metrics_csv_path=metrics_path,
                                confidence=cf,
                                current_round=r,
                            )
                        except Exception as _e_buf:
                            print(f"[BUFFER][WARN] could not register DONE_REAL post-flip for ID={_id}: {_e_buf}")
            elif pl == "FAKE":
                led.loc[ix, "status"] = "ACTIVE_FAKE"
            else:
                led.loc[ix, "status"] = "ERROR"

        _save_ledger(LEDGER_PATH, led)

        print(f"[ORCH] Running orchestrator to create decisions for next round using round {r} signals...")

        context = dict(
            detector_name=DETECTOR,
            round_dir=rd_round,
            detector_csv_prefix=DETECTOR_PREFIX,
            filter_fake=True,
            shap_dir=rd_shap,
            shap_filename="shap_attributions_per_id.csv",
            baseline_coeffs_path=BASELINE_COEFFS_PATH,
            trajectory_path=TRAJECTORY_PATH,
            agent_snapshots_base_dir=os.path.join(OUTPUT_ROOT, "trajectory", DETECTOR, "agent_snapshots"),
            invert_direction=_INVERT_DIRECTION,
            anti_feature=_ANTI_FEATURE,
        )

        dec = _call_orchestrator_iterative_adaptive(**context)
        decisions_path = os.path.join(rd_orch, "orch_decisions_with_reasons.csv")

        if isinstance(dec, pd.DataFrame) and not dec.empty:
            dec["ID"] = dec["ID"].astype(str)
            _save_csv(dec, decisions_path)
            print(f"[ORCH] Decisions saved -> {decisions_path}")
            led = _record_decisions_only(led, dec, decision_round=r)
            _save_ledger(LEDGER_PATH, led)
        else:
            print("[ORCH] WARN: orchestrator returned no decisions dataframe. (This is expected if FAKE=0).")
            if len(fake_ids_out) > 0:
                print(f"[STOP] Orchestrator produced 0 decisions with {len(fake_ids_out)} FAKE IDs remaining "
                      "(all eligible features exhausted) -> loop ends.")
                break

        if len(fake_ids_out) == 0:
            print("[STOP] No FAKE remaining among generated -> loop ends.")
            break

        active_ids_in = fake_ids_out
        r += 1

    print("\n" + "=" * 95)
    print("[DONE] Loop finished. Saving final report...")
    print("=" * 95)

    led_final = _load_csv(LEDGER_PATH) if os.path.exists(LEDGER_PATH) else led
    led_final["ID"] = led_final["ID"].astype(str)
    _save_csv(led_final, FINAL_REPORT_PATH)
    print(f"[REPORT] Saved -> {FINAL_REPORT_PATH}")
    return 0


class LoopPipeline:
    """Loop (round >0) pipeline runner."""

    def __init__(
        self,
        dataset_path: str | None = None,
        detector: str | None = None,
        output_root: str = "Output",
        max_round: int = 0,
        resume: bool = True,
        min_active_fake: int = DEFAULT_MIN_ACTIVE_FAKE,
        enable_drift_retrain: bool = True,
        invert_direction: bool = False,
        anti_feature: bool = False,
    ):
        self.dataset_path = dataset_path
        self.detector = detector
        self.output_root = output_root
        self.max_round = int(max_round or 0)
        self.resume = bool(resume)
        self.min_active_fake = int(min_active_fake or DEFAULT_MIN_ACTIVE_FAKE)
        self.enable_drift_retrain = bool(enable_drift_retrain)
        self.invert_direction = bool(invert_direction)
        self.anti_feature = bool(anti_feature)

    def run(self, argv=None) -> int:
        if argv is not None:
            parser = build_parser()
            args = parser.parse_args(argv)
            return _run(args)

        if not self.dataset_path:
            raise ValueError("dataset_path is required")
        if not self.detector:
            raise ValueError("detector is required")

        class _Args:
            pass

        args = _Args()
        args.nomedataset = self.dataset_path
        args.nomedetector = self.detector
        args.output_root = self.output_root
        args.max_round = self.max_round
        args.resume = bool(self.resume)
        args.no_resume = not bool(self.resume)
        args.min_active_fake = int(self.min_active_fake)
        args.no_drift_retrain = not bool(self.enable_drift_retrain)
        args.invert_direction = bool(self.invert_direction)
        args.anti_feature = bool(self.anti_feature)
        return _run(args)

def main(argv=None) -> int:
    return LoopPipeline().run(argv)

if __name__ == "__main__":
    raise SystemExit(main())