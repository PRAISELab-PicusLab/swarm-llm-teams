import os
import argparse
import shutil
import json
import pandas as pd
import torch

from .detector_runner import run_detector_on_df
from .meta_model import train_meta_model_for_detector, compute_shap_with_frozen_baseline, build_meta_join_dataframe
from .orchestrator_baseline_subtasks import run_orchestrator_round
from .retraining_buffer import build_real_baseline, _buffer_dir as _retrain_buffer_dir
from .constants import MIN_ACTIVE_FAKE


# ---------------- Helpers ----------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _require_file(path: str, msg: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{msg}: {path}")

def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")

def _save_csv(df: pd.DataFrame, path: str) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False, encoding="utf-8")

def _normalize_id(df: pd.DataFrame) -> pd.DataFrame:
    if "ID" not in df.columns:
        raise ValueError("Missing column 'ID'")
    out = df.copy()
    out["ID"] = out["ID"].astype(str)
    return out

def _ids_in_csv(path: str) -> set[str]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
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

def _count_labels(det_df: pd.DataFrame):
    s = det_df["pred_label"].astype(str).str.upper()
    return int((s == "FAKE").sum()), int((s == "REAL").sum())


def _write_json(path: str, payload: dict) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_rmtree(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _collect_round0_baseline_audit(detector: str, truth_ids: set[str], metrics_round_path: str, detector_round_path: str, meta_frozen_dir: str, orch_decisions_path: str, round0_dir: str, metrics_csv_name: str, detector_prefix: str) -> dict:
    metrics_ids = _ids_in_csv(metrics_round_path)
    detector_ids = _ids_in_csv(detector_round_path)
    shap_ids = _ids_in_csv(os.path.join(meta_frozen_dir, "shap_attributions_per_id.csv"))
    orch_ids = _ids_in_csv(orch_decisions_path)

    join_ids = set()
    try:
        df_join, _ = build_meta_join_dataframe(
            detector_name=detector,
            round_dir=round0_dir,
            feature_cols=None,
            metrics_csv_name=metrics_csv_name,
            detector_csv_prefix=detector_prefix,
            strict_feature_check=False,
        )
        join_ids = set(df_join["ID"].astype(str).tolist()) if not df_join.empty else set()
    except Exception:
        join_ids = set()

    fake_truth_ids = set()
    if os.path.exists(detector_round_path):
        try:
            dfd = pd.read_csv(detector_round_path, encoding="utf-8")
            if {"ID", "pred_label"}.issubset(dfd.columns):
                dfd["ID"] = dfd["ID"].astype(str)
                fake_truth_ids = set(dfd[dfd["pred_label"].astype(str).str.upper() == "FAKE"]["ID"].astype(str).tolist()) & set(truth_ids)
        except Exception:
            fake_truth_ids = set()

    meta_ready = all(os.path.exists(os.path.join(meta_frozen_dir, n)) for n in ["meta_model_coefficients.csv", "feature_baseline_stats.csv", "meta_model_pipe.pkl", "shap_attributions_per_id.csv"])
    return {
        "truth_ids": sorted(truth_ids),
        "metrics_ids": sorted(metrics_ids),
        "detector_ids": sorted(detector_ids),
        "join_ids": sorted(join_ids),
        "shap_ids": sorted(shap_ids),
        "orch_ids": sorted(orch_ids),
        "fake_truth_ids": sorted(fake_truth_ids),
        "meta_ready": bool(meta_ready),
    }


def _baseline_rebuild_reasons(audit: dict) -> list[str]:
    reasons = []
    truth_ids = set(audit.get("truth_ids", []))
    metrics_ids = set(audit.get("metrics_ids", []))
    detector_ids = set(audit.get("detector_ids", []))
    join_ids = set(audit.get("join_ids", []))
    shap_ids = set(audit.get("shap_ids", []))
    orch_ids = set(audit.get("orch_ids", []))
    fake_truth_ids = set(audit.get("fake_truth_ids", []))

    if metrics_ids != truth_ids:
        reasons.append("metrics_ids_mismatch_vs_truth")
    if detector_ids != truth_ids:
        reasons.append("detector_ids_mismatch_vs_truth")
    if join_ids != (metrics_ids & detector_ids):
        reasons.append("join_ids_mismatch_vs_metrics_detector")
    if audit.get("meta_ready") and shap_ids != join_ids:
        reasons.append("shap_ids_mismatch_vs_join")
    if orch_ids and orch_ids != fake_truth_ids:
        reasons.append("orch_ids_mismatch_vs_fake_truth")
    if (not audit.get("meta_ready")) and (join_ids or detector_ids):
        reasons.append("meta_artifacts_missing_or_partial")
    return reasons


def _reset_detector_specific_baseline(det_dir: str, orch_dir: str, meta_frozen_dir: str, detector_round_path: str) -> None:
    _safe_rmtree(det_dir)
    _safe_rmtree(orch_dir)
    _safe_rmtree(meta_frozen_dir)
    _safe_remove(detector_round_path)
    _ensure_dir(det_dir)
    _ensure_dir(orch_dir)
    _ensure_dir(meta_frozen_dir)


def _write_all_real_markers(
    *,
    detector: str,
    orch_dir: str,
    meta_frozen_dir: str,
    round0_dir: str,
    detector_round_path: str,
    metrics_round_path: str,
    generated_path: str,
    metrics_master_path: str,
    n_fake: int,
    n_real: int,
) -> None:
    """
    Writes minimal artifacts that mark this detector as ALL-REAL in baseline0.
    This allows the pipeline runner to safely skip loop and move to next detector.
    """
    _ensure_dir(orch_dir)
    _ensure_dir(meta_frozen_dir)

    # marker file
    stop_txt = os.path.join(orch_dir, "STOP_ALL_REAL.txt")
    with open(stop_txt, "w", encoding="utf-8") as f:
        f.write(
            "STOP_ALL_REAL\n"
            f"detector={detector}\n"
            f"note=Detector predicted only REAL in baseline round0. Orchestrator+loop skipped.\n"
            f"fake={n_fake}, real={n_real}\n"
        )

    # json reason (useful for resume/debug)
    stop_json = os.path.join(orch_dir, "stop_reason.json")
    with open(stop_json, "w", encoding="utf-8") as f:
        json.dump({
            "detector": detector,
            "reason": "all_real_in_baseline0",
            "fake": n_fake,
            "real": n_real,
            "round_dir": os.path.abspath(round0_dir),
            "detector_round_path": os.path.abspath(detector_round_path),
            "metrics_round_path": os.path.abspath(metrics_round_path),
            "generated_path": os.path.abspath(generated_path),
            "metrics_master_path": os.path.abspath(metrics_master_path),
            "note": "No ACTIVE_FAKE available. Baseline completed and pipeline should proceed to next detector."
        }, f, indent=2, ensure_ascii=False)

    # create empty orchestrator outputs so other tools that "require files" won't break in resume/analysis
    # decisions CSV (empty but with header)
    decisions_path = os.path.join(orch_dir, "orch_decisions_with_reasons.csv")
    if not os.path.exists(decisions_path):
        pd.DataFrame(columns=["ID", "chosen_feature", "reason", "detector", "round"]).to_csv(
            decisions_path, index=False, encoding="utf-8"
        )

    # guides jsonl (empty file is OK)
    guides_path = os.path.join(orch_dir, "orch_guides.jsonl")
    if not os.path.exists(guides_path):
        with open(guides_path, "w", encoding="utf-8") as f:
            f.write("")

    # Also drop a meta marker, since meta-model is not meaningful when all REAL
    meta_skip = os.path.join(meta_frozen_dir, "meta_model_skipped.json")
    if not os.path.exists(meta_skip):
        with open(meta_skip, "w", encoding="utf-8") as f:
            json.dump({
                "detector_name": detector,
                "round_dir": os.path.abspath(round0_dir),
                "target": "label",
                "reason": "all_real_in_baseline0",
                "note": "Meta-model + SHAP skipped because all samples were predicted REAL."
            }, f, indent=2, ensure_ascii=False)


# ---------------- Parser / Runner ----------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Round 0 baseline for ONE detector (no generation)")

    parser.add_argument("--nomedataset", type=str, required=True, help="Path dataset CSV con colonna ID")
    parser.add_argument("--nomedetector", type=str, default=os.getenv("DETECTOR", "valhalla"), help="Nome detector")
    parser.add_argument("--resume", action="store_true", help="Resume (default True)")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume")
    parser.add_argument("--se-multiplier", type=float, default=1.0, help="SE multiplier for SGL CV selection rule (default 1.0)")
    parser.add_argument("--output_root", type=str, default="Output", help="Output root (default: Output)")
    parser.add_argument("--invert-direction", action="store_true",
                        help="exp2 ablation: flip direction in the round-0 decisions (drives round 1).")
    parser.add_argument("--anti-feature", action="store_true",
                        help="exp3 ablation: least-important feature in the round-0 decisions (drives round 1).")

    return parser


def _run(args) -> int:
    # resolve resume (default True)
    RESUME = True
    if args.no_resume:
        RESUME = False
    elif args.resume:
        RESUME = True

    ROUND = 0
    ROUND_TAG = f"round_{ROUND:03d}"

    DATASET_PATH = args.nomedataset
    DETECTOR = args.nomedetector
    SE_MULTIPLIER = float(getattr(args, "se_multiplier", 1.0))
    INVERT_DIRECTION = bool(getattr(args, "invert_direction", False))
    ANTI_FEATURE = bool(getattr(args, "anti_feature", False))
    if INVERT_DIRECTION or ANTI_FEATURE:
        print(f"[BASELINE0][CONFIG] ABLATION invert_direction={INVERT_DIRECTION} "
              f"anti_feature={ANTI_FEATURE} (applied to round-0 decisions)")

    os.environ["DATASET_PATH"] = DATASET_PATH
    os.environ["DETECTOR"] = DETECTOR

    OUTPUT_ROOT = getattr(args, "output_root", "Output") or "Output"
    ROUNDS_DIR = os.path.join(OUTPUT_ROOT, "rounds")
    ROUND0_DIR = os.path.join(ROUNDS_DIR, ROUND_TAG)

    # IMPORTANT: round0/generation must be GENERATION-ONLY
    GEN_DIR = os.path.join(ROUND0_DIR, "generation")

    # per-detector dirs
    DET_DIR = os.path.join(ROUND0_DIR, "detectors", DETECTOR)
    ORCH_DIR = os.path.join(ROUND0_DIR, "orchestrator", DETECTOR)
    META_FROZEN_DIR = os.path.join(OUTPUT_ROOT, "meta_frozen", DETECTOR)

    # files
    GENERATED_PATH = os.path.join(GEN_DIR, "Generated_Fake_News.csv")

    METRICS_CSV_NAME = "agent_metrics.csv"
    METRICS_MASTER_PATH = os.path.join(GEN_DIR, METRICS_CSV_NAME)

    DETECTOR_PREFIX = "detector_"
    DETECTOR_OUT_PATH = os.path.join(DET_DIR, f"{DETECTOR_PREFIX}{DETECTOR}.csv")

    ORCH_DECISIONS_PATH = os.path.join(ORCH_DIR, "orch_decisions_with_reasons.csv")
    ORCH_GUIDES_PATH = os.path.join(ORCH_DIR, "orch_guides.jsonl")

    _ensure_dir(ROUND0_DIR)
    _ensure_dir(DET_DIR)
    _ensure_dir(ORCH_DIR)
    _ensure_dir(META_FROZEN_DIR)

    print(f"[INFO] Torch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"[INFO] ROUND0 baseline (detector-only) -> {ROUND_TAG}")
    print(f"[INFO] Dataset:  {DATASET_PATH}")
    print(f"[INFO] Detector: {DETECTOR}")
    print(f"[INFO] RESUME:   {RESUME}")
    print(f"[INFO] GEN_DIR:  {GEN_DIR}")
    print(f"[INFO] DET_DIR:  {DET_DIR}")
    print(f"[INFO] ORCH_DIR: {ORCH_DIR}")
    print(f"[INFO] META_FROZEN_DIR: {META_FROZEN_DIR}")

    # preconditions: generation must exist
    _require_file(GENERATED_PATH, "Missing round0 generation output (run `python -m pipeline gen0 ...` first)")
    _require_file(METRICS_MASTER_PATH, "Missing round0 metrics (run `python -m pipeline gen0 ...` first)")

    # load generated
    df_generated = _normalize_id(_load_csv(GENERATED_PATH))
    truth_ids = set(df_generated["ID"].astype(str).tolist())

    # 1) Baseline coherence audit + detector/meta/orchestrator rebuild policy
    coeffs_path = os.path.join(META_FROZEN_DIR, "meta_model_coefficients.csv")
    shap_path = os.path.join(META_FROZEN_DIR, "shap_attributions_per_id.csv")
    stats_path = os.path.join(META_FROZEN_DIR, "feature_baseline_stats.csv")
    pipe_path = os.path.join(META_FROZEN_DIR, "meta_model_pipe.pkl")

    audit = _collect_round0_baseline_audit(
        detector=DETECTOR,
        truth_ids=truth_ids,
        metrics_round_path=METRICS_MASTER_PATH,
        detector_round_path=DETECTOR_OUT_PATH,
        meta_frozen_dir=META_FROZEN_DIR,
        orch_decisions_path=ORCH_DECISIONS_PATH,
        round0_dir=ROUND0_DIR,
        metrics_csv_name=METRICS_CSV_NAME,
        detector_prefix=DETECTOR_PREFIX,
    )
    rebuild_reasons = [] if not RESUME else _baseline_rebuild_reasons(audit)
    _write_json(os.path.join(META_FROZEN_DIR, "baseline_round0_audit.json"), {
        "detector": DETECTOR,
        "rebuild_reasons": rebuild_reasons,
        "meta_ready": audit.get("meta_ready", False),
        "n_truth_ids": len(audit.get("truth_ids", [])),
        "n_metrics_ids": len(audit.get("metrics_ids", [])),
        "n_detector_ids": len(audit.get("detector_ids", [])),
        "n_join_ids": len(audit.get("join_ids", [])),
        "n_shap_ids": len(audit.get("shap_ids", [])),
        "n_orch_ids": len(audit.get("orch_ids", [])),
        "n_fake_truth_ids": len(audit.get("fake_truth_ids", [])),
    })

    if rebuild_reasons or (not RESUME):
        if rebuild_reasons:
            print(f"[BASELINE0][REBUILD] detector-specific baseline reset -> reasons={rebuild_reasons}")
        _reset_detector_specific_baseline(DET_DIR, ORCH_DIR, META_FROZEN_DIR, DETECTOR_OUT_PATH)

    detector_ready = os.path.exists(DETECTOR_OUT_PATH) and os.path.exists(DETECTOR_OUT_PATH) and not rebuild_reasons and RESUME
    if detector_ready:
        print(f"[DETECTOR] Found -> {DETECTOR_OUT_PATH}")
        df_det = _normalize_id(_load_csv(DETECTOR_OUT_PATH))
    else:
        print(f"[DETECTOR] Running detector '{DETECTOR}' -> {DETECTOR_OUT_PATH}")
        df_det = run_detector_on_df(detector=DETECTOR, df=df_generated, device=None)
        df_det = _normalize_id(df_det)
        _save_csv(df_det, DETECTOR_OUT_PATH)

    n_fake, n_real = _count_labels(df_det)
    print(f"[DETECTOR] FAKE={n_fake} | REAL={n_real}")

    if n_fake == 0 and n_real > 0:
        print(f"[BASELINE0][ALL-REAL] Detector '{DETECTOR}' predicted only REAL in round0.")
        print("[BASELINE0][ALL-REAL] Skipping meta-model, SHAP, orchestrator, and loop for this detector.")
        _write_all_real_markers(
            detector=DETECTOR,
            orch_dir=ORCH_DIR,
            meta_frozen_dir=META_FROZEN_DIR,
            round0_dir=ROUND0_DIR,
            detector_round_path=DETECTOR_OUT_PATH,
            metrics_round_path=METRICS_MASTER_PATH,
            generated_path=GENERATED_PATH,
            metrics_master_path=METRICS_MASTER_PATH,
            n_fake=n_fake,
            n_real=n_real,
        )

        print("\n[DONE] BASELINE ROUND0 (DETECTOR-ONLY) COMPLETE — ALL REAL")
        print(f"[OUT] Generated:        {GENERATED_PATH}")
        print(f"[OUT] Metrics:          {METRICS_MASTER_PATH}")
        print(f"[OUT] Detector:         {DETECTOR_OUT_PATH}")
        print(f"[OUT] Orch dir:         {ORCH_DIR}")
        print(f"[OUT] Meta frozen dir:  {META_FROZEN_DIR}")
        print(f"[STATUS] FAKE={n_fake} | REAL={n_real}")
        return 0

    meta_ready = all(os.path.exists(pth) for pth in [coeffs_path, shap_path, stats_path, pipe_path]) and RESUME and not rebuild_reasons
    if meta_ready:
        print("[META] Found existing frozen baseline meta artifacts -> skip")
    else:
        print("[META] Training FROZEN baseline meta-model -> Output/meta_frozen/<DETECTOR>/")
        train_result = train_meta_model_for_detector(
            detector_name=DETECTOR,
            round_dir=ROUND0_DIR,
            target="label",
            metrics_csv_name=METRICS_CSV_NAME,
            detector_csv_prefix=DETECTOR_PREFIX,
            output_dir=META_FROZEN_DIR,
            se_multiplier=SE_MULTIPLIER,
        )
        if train_result is None:
            stop_path = os.path.join(ORCH_DIR, "STOP_INSUFFICIENT_FAKE.txt")
            _ensure_dir(os.path.dirname(stop_path))
            with open(stop_path, "w", encoding="utf-8") as f:
                f.write(f"n_fake={n_fake} < threshold={MIN_ACTIVE_FAKE}\n")
            print(f"[STOP] Insufficient FAKE -> {stop_path}. Loop will not start.")
            return 0
        if os.path.exists(pipe_path):
            compute_shap_with_frozen_baseline(
                baseline_pipe_path=pipe_path,
                round_dir=ROUND0_DIR,
                detector_name=DETECTOR,
                out_dir=META_FROZEN_DIR,
                metrics_csv_name=METRICS_CSV_NAME,
                detector_csv_prefix=DETECTOR_PREFIX,
                target="label",
                only_ids=None,
                append_if_exists=False,
            )

    skipped_path = os.path.join(META_FROZEN_DIR, "meta_model_skipped.json")
    if os.path.exists(skipped_path):
        print(f"[META][SKIP] Found meta_model_skipped.json -> {skipped_path}")
        print("[META][SKIP] Skipping orchestrator because meta artifacts are not available.")
        _write_all_real_markers(
            detector=DETECTOR,
            orch_dir=ORCH_DIR,
            meta_frozen_dir=META_FROZEN_DIR,
            round0_dir=ROUND0_DIR,
            detector_round_path=DETECTOR_OUT_PATH,
            metrics_round_path=METRICS_MASTER_PATH,
            generated_path=GENERATED_PATH,
            metrics_master_path=METRICS_MASTER_PATH,
            n_fake=n_fake,
            n_real=n_real,
        )
        return 0

    _require_file(coeffs_path, "Missing meta-model coefficients")
    _require_file(shap_path, "Missing SHAP per-ID attributions")
    _require_file(stats_path, "Missing baseline feature stats")
    _require_file(pipe_path, "Missing frozen meta-model pipe")

    # PUNTO 4 — snapshot REAL articles into retraining buffer (frozen).
    try:
        _buf_dir = _retrain_buffer_dir(os.path.abspath(OUTPUT_ROOT), DETECTOR)
        n_real_buf = build_real_baseline(
            round0_metrics_csv=METRICS_MASTER_PATH,
            round0_detector_csv=DETECTOR_OUT_PATH,
            buffer_dir=_buf_dir,
        )
        print(f"[BUFFER] real_baseline snapshot: n={n_real_buf} -> {_buf_dir}")
    except Exception as _e_buf:
        print(f"[BUFFER][WARN] build_real_baseline failed: {_e_buf}")

    orch_ready = os.path.exists(ORCH_DECISIONS_PATH) and os.path.exists(ORCH_GUIDES_PATH) and RESUME and not rebuild_reasons
    if orch_ready:
        print("[ORCH] Found existing decisions -> skip")
    else:
        print("[ORCH] Building baseline decisions (FAKE only)")
        df_decisions = run_orchestrator_round(
            detector_name=DETECTOR,
            round_dir=ROUND0_DIR,
            detector_csv_prefix=DETECTOR_PREFIX,
            filter_fake=True,
            meta_dir=META_FROZEN_DIR,
            invert_direction=INVERT_DIRECTION,
            anti_feature=ANTI_FEATURE,
        )
        print(f"[ORCH] Decisions rows={0 if df_decisions is None else len(df_decisions)}")

    _require_file(ORCH_DECISIONS_PATH, "Missing orchestrator decisions")
    _require_file(ORCH_GUIDES_PATH, "Missing orchestrator guides")

    print("\n[DONE] BASELINE ROUND0 (DETECTOR-ONLY) COMPLETE")
    print(f"[OUT] Generated:        {GENERATED_PATH}")
    print(f"[OUT] Metrics:          {METRICS_MASTER_PATH}")
    print(f"[OUT] Detector:         {DETECTOR_OUT_PATH}")
    print(f"[OUT] Meta frozen:      {META_FROZEN_DIR}")
    print(f"[OUT] Orch dec:         {ORCH_DECISIONS_PATH}")
    print(f"[OUT] Orch guides:      {ORCH_GUIDES_PATH}")
    print(f"[STATUS] FAKE={n_fake} | REAL={n_real}")
    return 0


class Round0DetectorBaselinePipeline:
    def __init__(self, dataset_path: str | None = None, detector: str | None = None,
                 resume: bool = True, se_multiplier: float = 1.0,
                 invert_direction: bool = False, anti_feature: bool = False,
                 output_root: str = "Output"):
        self.dataset_path = dataset_path
        self.detector = detector
        self.resume = bool(resume)
        self.se_multiplier = float(se_multiplier)
        self.invert_direction = bool(invert_direction)
        self.anti_feature = bool(anti_feature)
        self.output_root = output_root or "Output"

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
        args.resume = bool(self.resume)
        args.no_resume = not bool(self.resume)
        args.se_multiplier = self.se_multiplier
        args.invert_direction = bool(self.invert_direction)
        args.anti_feature = bool(self.anti_feature)
        args.output_root = self.output_root
        return _run(args)


def main(argv=None) -> int:
    return Round0DetectorBaselinePipeline().run(argv)


if __name__ == "__main__":
    raise SystemExit(main())