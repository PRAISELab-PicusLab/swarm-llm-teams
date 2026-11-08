import os
import shutil
import argparse
import pandas as pd
import torch

from .pipeline_multiagent import run_generation_pipeline
from .metrics_runner import compute_agent_metrics, write_baseline_references
from .detector_runner import run_detector_on_df
from .meta_model import train_meta_model_for_detector, compute_shap_with_frozen_baseline
from .orchestrator_baseline_subtasks import run_orchestrator_round
from .id_controller import ensure_sample_ids, compute_missing_ids


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

def _count_labels(det_df: pd.DataFrame):
    s = det_df["pred_label"].astype(str).str.upper()
    return int((s == "FAKE").sum()), int((s == "REAL").sum())

def _order_df_by_sample_ids(df: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    df2 = df.copy()
    df2["ID"] = df2["ID"].astype(str)
    ord_map = {sid: i for i, sid in enumerate(sample_ids)}
    df2["__ord__"] = df2["ID"].map(ord_map).fillna(10**12).astype(int)
    df2 = df2.sort_values("__ord__").drop(columns=["__ord__"])
    return df2

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

def _write_panel_parquet_round0(sample_ids: list[str], df_generated: pd.DataFrame, out_path: str) -> None:
    """
    Panel required by downstream loops:
      ID, title_round_r, text_round_r, status, source_round

    IMPORTANT:
    - Always write ONE ROW PER sample_id (even if generation missing).
    - Stable cohort size across rounds.
    """
    df_g = df_generated.copy()
    df_g = df_g[["ID", "title", "text"]].copy()
    df_g["ID"] = df_g["ID"].astype(str)

    g_map = df_g.set_index("ID")[["title", "text"]].to_dict(orient="index")

    titles, texts, status = [], [], []
    for sid in sample_ids:
        if sid in g_map:
            titles.append(str(g_map[sid].get("title", "")))
            texts.append(str(g_map[sid].get("text", "")))
            status.append("OK")
        else:
            titles.append("")
            texts.append("")
            status.append("MISSING")

    panel = pd.DataFrame({
        "ID": [str(x) for x in sample_ids],
        "title_round_r": titles,
        "text_round_r": texts,
        "status": status,
        "source_round": [0] * len(sample_ids),
    })

    _ensure_dir(os.path.dirname(out_path) or ".")
    panel.to_parquet(out_path, index=False)


# ---------------- Parser / Runner ----------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Round 0 baseline orchestrator")
    parser.add_argument("--nomedataset", type=str, required=True, help="Path dataset CSV con colonna ID")
    parser.add_argument("--nomedetector", type=str, default=os.getenv("DETECTOR", "valhalla"), help="Nome detector (override env DETECTOR)")

    # sample ids behavior
    parser.add_argument("--mode", type=str, default="keep", choices=["keep", "expand", "rebuild"], help="Mode sample ids")
    parser.add_argument("--desired_n", type=int, default=100, help="Numero ID sample (round0)")
    parser.add_argument("--random_state", type=int, default=123, help="Seed deterministic")

    # resume behavior
    parser.add_argument("--resume_pipeline", action="store_true", help="Resume pipeline generation (default True)")
    parser.add_argument("--no_resume_pipeline", action="store_true", help="Disable resume pipeline generation")
    parser.add_argument("--se-multiplier", type=float, default=1.0, help="SE multiplier for SGL CV selection rule (default 1.0)")
    return parser


def _run(args) -> int:
    # resolve flags (default True)
    resume_pipeline = True
    if args.no_resume_pipeline:
        resume_pipeline = False
    elif args.resume_pipeline:
        resume_pipeline = True

    ROUND = 0
    ROUND_TAG = f"round_{ROUND:03d}"

    DATASET_PATH = args.nomedataset
    DETECTOR = args.nomedetector
    ROUND0_SAMPLE_SIZE = int(args.desired_n)
    RANDOM_STATE = int(args.random_state)
    SE_MULTIPLIER = float(getattr(args, "se_multiplier", 1.0))

    # Set env so downstream modules can read if they need
    os.environ["DATASET_PATH"] = DATASET_PATH
    os.environ["DETECTOR"] = DETECTOR

    # sample stabile in dataset/
    SAMPLE_IDS_PATH = "dataset/dataset_sample_ids.csv"

    # OUTPUT STRUCTURE
    OUTPUT_ROOT = "Output"
    COHORT_DIR = os.path.join(OUTPUT_ROOT, "cohort")
    TRACKING_DIR = os.path.join(OUTPUT_ROOT, "tracking")
    ROUNDS_DIR = os.path.join(OUTPUT_ROOT, "rounds")

    ROUND0_DIR = os.path.join(ROUNDS_DIR, ROUND_TAG)
    GEN_DIR = os.path.join(ROUND0_DIR, "generation")
    DET_DIR = os.path.join(ROUND0_DIR, "detectors", DETECTOR)

    META_FROZEN_DIR = os.path.join(OUTPUT_ROOT, "meta_frozen", DETECTOR)
    ORCH_DIR = os.path.join(ROUND0_DIR, "orchestrator", DETECTOR)

    # files
    COHORT_ORIGINAL_PATH = os.path.join(COHORT_DIR, "cohort_original.csv")
    GENERATED_PATH = os.path.join(GEN_DIR, "Generated_Fake_News.csv")

    METRICS_CSV_NAME = "agent_metrics.csv"
    METRICS_MASTER_PATH = os.path.join(GEN_DIR, METRICS_CSV_NAME)

    DETECTOR_PREFIX = "detector_"
    DETECTOR_OUT_PATH = os.path.join(DET_DIR, f"{DETECTOR_PREFIX}{DETECTOR}.csv")

    ORCH_DECISIONS_PATH = os.path.join(ORCH_DIR, "orch_decisions_with_reasons.csv")
    ORCH_GUIDES_PATH = os.path.join(ORCH_DIR, "orch_guides.jsonl")

    PANEL_PARQUET_PATH = os.path.join(GEN_DIR, "panel_articles_round_r.parquet")

    # dirs
    _ensure_dir(COHORT_DIR)
    _ensure_dir(TRACKING_DIR)
    _ensure_dir(ROUND0_DIR)
    _ensure_dir(GEN_DIR)
    _ensure_dir(DET_DIR)
    _ensure_dir(META_FROZEN_DIR)
    _ensure_dir(ORCH_DIR)

    print(f"[INFO] Torch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"[INFO] ROUND0 baseline -> {ROUND_TAG}")
    print(f"[INFO] Dataset:  {DATASET_PATH}")
    print(f"[INFO] Detector: {DETECTOR}")
    print(f"[INFO] mode={args.mode} desired_n={ROUND0_SAMPLE_SIZE} seed={RANDOM_STATE}")
    print(f"[INFO] resume_pipeline={resume_pipeline}")
    print(f"[INFO] GEN_DIR: {GEN_DIR}")
    print(f"[INFO] DET_DIR: {DET_DIR}")
    print(f"[INFO] META_FROZEN_DIR: {META_FROZEN_DIR}")
    print(f"[INFO] ORCH_DIR: {ORCH_DIR}")

    # 0) ensure sample ids (stable, deterministic)
    sample_info = ensure_sample_ids(
        sample_ids_path=SAMPLE_IDS_PATH,
        dataset_path=DATASET_PATH,
        desired_n=ROUND0_SAMPLE_SIZE,
        mode=args.mode,
        random_state=RANDOM_STATE,
        recreate_if_missing=True,
        recreate_force=(args.mode == "rebuild"),
    )
    print(f"[ID] sample_ids mode={sample_info['mode']} sample_n={sample_info['sample_n']} added={sample_info['added_n']}")
    if sample_info.get("notes"):
        for n in sample_info["notes"]:
            print("[ID][NOTE]", n)

    # ✅ REBUILD = davvero da zero (pulizia outputs round0)
    if args.mode == "rebuild":
        print("[REBUILD] Cleaning previous round0 outputs (GEN/DET/ORCH + round copies) to start from scratch...")
        for p in (GEN_DIR, DET_DIR, ORCH_DIR):
            try:
                if os.path.exists(p):
                    shutil.rmtree(p)
                _ensure_dir(p)
            except Exception as e:
                raise RuntimeError(f"[REBUILD] Failed cleaning {p}: {e}")

    # 1) load dataset
    _require_file(DATASET_PATH, "Dataset not found")
    df_real = _normalize_id(_load_csv(DATASET_PATH))
    print(f"[DATASET] rows={len(df_real)} path={DATASET_PATH}")

    # load sample ids (and preserve order)
    _require_file(SAMPLE_IDS_PATH, "Sample IDs file not found")
    sample_ids_df = _load_csv(SAMPLE_IDS_PATH)
    if "ID" not in sample_ids_df.columns:
        raise ValueError(f"Sample IDs file must contain 'ID': {SAMPLE_IDS_PATH}")
    sample_ids_df["ID"] = sample_ids_df["ID"].astype(str)
    sample_ids = sample_ids_df["ID"].tolist()

    # Restrict to sample IDs (preserve order!)
    df_real_sample = df_real[df_real["ID"].isin(set(sample_ids))].copy()
    df_real_sample = _order_df_by_sample_ids(df_real_sample, sample_ids).reset_index(drop=True)
    print(f"[SAMPLE] IDs in sample file: {len(sample_ids)} | Found in dataset: {len(df_real_sample)}")

    # Save cohort_original.csv (stable cohort)
    if not os.path.exists(COHORT_ORIGINAL_PATH):
        _save_csv(df_real_sample, COHORT_ORIGINAL_PATH)
        print(f"[COHORT] saved -> {COHORT_ORIGINAL_PATH}")
    else:
        print(f"[COHORT] found -> {COHORT_ORIGINAL_PATH}")

    # 2) check missing in Generated_Fake_News (may not exist yet)
    generated_exists = os.path.exists(GENERATED_PATH) and os.path.getsize(GENERATED_PATH) > 0
    SKIPPED_PATH = os.path.join(GEN_DIR, "Skipped_IDs.csv")
    missing_ids, done_ids = compute_missing_ids(SAMPLE_IDS_PATH, GENERATED_PATH, skipped_csv_path=SKIPPED_PATH if resume_pipeline else None)
    print(f"[GEN-CHECK] generated_exists={generated_exists} done={len(done_ids)} missing={len(missing_ids)} (target={len(sample_ids)})")

    # 3) GENERATION GATING
    should_run_pipeline = False
    if len(missing_ids) == 0 and generated_exists:
        print("[PIPE] Round0: Generated already complete -> SKIP pipeline (NO BACKFILL, NO LLM).")
        should_run_pipeline = False
    else:
        if generated_exists and not resume_pipeline:
            print("[PIPE] Round0: Generated exists but resume_pipeline=False -> SKIP generation (will continue with next steps).")
            should_run_pipeline = False
        else:
            if len(missing_ids) > 0:
                print(f"[PIPE] Running pipeline to fill missing IDs ONLY: {len(missing_ids)} (resume_pipeline={resume_pipeline})")
            else:
                print(f"[PIPE] Running pipeline to create Generated_Fake_News (resume_pipeline={resume_pipeline})")
            should_run_pipeline = True

    if should_run_pipeline:
        process_only_missing = (args.mode == "keep" and resume_pipeline)

        run_generation_pipeline(
            df_input=df_real_sample,
            base_out_dir=GEN_DIR,
            sample_size=len(sample_ids),
            sample_ids_path=SAMPLE_IDS_PATH,
            resume=resume_pipeline,
            process_only_missing_round0=process_only_missing,
            df_prev=None,
            prev_round_dir=None,
            detector_name=None,
        )

    # If generation was skipped and file is missing, stop early with a clear message.
    if not (os.path.exists(GENERATED_PATH) and os.path.getsize(GENERATED_PATH) > 0):
        raise FileNotFoundError(
            f"[ROUND0] Generated_Fake_News.csv is missing at {GENERATED_PATH}. "
            f"If you intentionally skipped generation (resume_pipeline=False), you must provide this file."
        )

    df_generated = _normalize_id(_load_csv(GENERATED_PATH))
    print(f"[GEN] rows in Generated_Fake_News.csv: {len(df_generated)}")

    # Source of truth for downstream
    truth_ids = set(df_generated["ID"].astype(str).tolist())

    # re-check missing after pipeline/skip (ordered)
    missing_ids2, done_ids2 = compute_missing_ids(SAMPLE_IDS_PATH, GENERATED_PATH, skipped_csv_path=SKIPPED_PATH if resume_pipeline else None)
    print(f"[GEN-CHECK-POST] done={len(done_ids2)} missing={len(missing_ids2)}")
    if len(missing_ids2) > 0:
        print("[WARN] Still missing IDs after generation/skip. Downstream steps will run on available rows only.")
        print("[WARN] For resume runs, you can rerun later and it will process the remaining missing IDs.")

    # 3b) Panel parquet — ALWAYS 1 row per sample_id
    if not os.path.exists(PANEL_PARQUET_PATH):
        _write_panel_parquet_round0(sample_ids=sample_ids, df_generated=df_generated, out_path=PANEL_PARQUET_PATH)
        print(f"[PANEL] saved -> {PANEL_PARQUET_PATH}")
    else:
        print(f"[PANEL] found -> {PANEL_PARQUET_PATH}")

    # 3c) Baseline references — freeze SA outputs (KS_0, KN_0, title_orig) for later rounds.
    # Must happen BEFORE compute_agent_metrics so the round-0 metrics use the new
    # anchored-reference semantics on the very first run.
    BASELINE_REFS_PATH = os.path.join(OUTPUT_ROOT, "meta_frozen", "baseline_references.json")
    SA_CSV_PATH = os.path.join(GEN_DIR, "SemanticAnalyzer.csv")
    write_baseline_references(
        sa_csv_path=SA_CSV_PATH,
        df_orig=df_real_sample,
        out_path=BASELINE_REFS_PATH,
    )

    # 4) Metrics (compute once in GEN_DIR; then copy into DET_DIR and ROUND0_DIR)
    missing_metrics_ids = truth_ids - _ids_in_csv(METRICS_MASTER_PATH)
    RESIDUALIZER_PATH = os.path.join(OUTPUT_ROOT, "meta_frozen", "residualizer.json")
    BASELINE_REFS_PATH = os.path.join(OUTPUT_ROOT, "meta_frozen", "baseline_references.json")
    if not os.path.exists(METRICS_MASTER_PATH):
        print(f"[METRICS] Computing master metrics -> {METRICS_MASTER_PATH}")
        compute_agent_metrics(
            df_orig=df_real_sample,
            base_out_dir=GEN_DIR,
            ids_mode="csv_in_dataset",
            out_csv_name=METRICS_CSV_NAME,
            residualizer_path=RESIDUALIZER_PATH,
            baseline_references_path=BASELINE_REFS_PATH,
        )
    elif len(missing_metrics_ids) > 0:
        df_real_missing = df_real_sample[df_real_sample["ID"].astype(str).isin(missing_metrics_ids)].copy()
        print(f"[METRICS][FIX2] Backfill missing metrics IDs={len(missing_metrics_ids)} -> append")
        compute_agent_metrics(
            df_orig=df_real_missing,
            base_out_dir=GEN_DIR,
            ids_mode="csv_in_dataset",
            out_csv_name=METRICS_CSV_NAME,
            only_ids=sorted(list(missing_metrics_ids)),
            append_if_exists=True,
            residualizer_path=RESIDUALIZER_PATH,
            baseline_references_path=BASELINE_REFS_PATH,
        )
    else:
        print(f"[METRICS] Found master -> {METRICS_MASTER_PATH}")

    _require_file(METRICS_MASTER_PATH, "Missing metrics csv after metrics computation")

    # 5) Detector output (written in DET_DIR only — single canonical location)
    missing_detector_ids = truth_ids - _ids_in_csv(DETECTOR_OUT_PATH)
    if not os.path.exists(DETECTOR_OUT_PATH):
        print(f"[DETECTOR] Running detector '{DETECTOR}' -> {DETECTOR_OUT_PATH}")
        df_det = run_detector_on_df(detector=DETECTOR, df=df_generated, device=None)
        df_det = _normalize_id(df_det)
        _save_csv(df_det, DETECTOR_OUT_PATH)
    elif len(missing_detector_ids) > 0:
        print(f"[DETECTOR][FIX2] Backfill missing detector IDs={len(missing_detector_ids)} -> append")
        df_sub = df_generated[df_generated["ID"].astype(str).isin(missing_detector_ids)].copy()
        df_new = run_detector_on_df(detector=DETECTOR, df=df_sub, device=None)
        df_new = _normalize_id(df_new)
        _append_dedupe_by_id(DETECTOR_OUT_PATH, df_new)
        df_det = _normalize_id(_load_csv(DETECTOR_OUT_PATH))
    else:
        df_det = _normalize_id(_load_csv(DETECTOR_OUT_PATH))
        print(f"[DETECTOR] Found -> {DETECTOR_OUT_PATH}")

    n_fake, n_real = _count_labels(df_det)
    print(f"[DETECTOR] FAKE={n_fake} | REAL={n_real}")

    # 6) Meta-model baseline artifacts (FROZEN, per detector) — uses ROUND0_DIR as round_dir
    coeffs_path = os.path.join(META_FROZEN_DIR, "meta_model_coefficients.csv")
    shap_path = os.path.join(META_FROZEN_DIR, "shap_attributions_per_id.csv")
    stats_path = os.path.join(META_FROZEN_DIR, "feature_baseline_stats.csv")
    pipe_path = os.path.join(META_FROZEN_DIR, "meta_model_pipe.pkl")

    meta_ready = (os.path.exists(coeffs_path) and os.path.exists(shap_path) and os.path.exists(stats_path) and os.path.exists(pipe_path))
    if not meta_ready:
        print("[META] Training FROZEN baseline meta-model -> Output/meta_frozen/<DETECTOR>/")
        train_meta_model_for_detector(
            detector_name=DETECTOR,
            round_dir=ROUND0_DIR,                 # ✅ FIX: canonical round_dir
            target="label",
            metrics_csv_name=METRICS_CSV_NAME,    # will read GEN_DIR/agent_metrics.csv
            detector_csv_prefix=DETECTOR_PREFIX,  # will read ROUND0_DIR/detector_<detector>.csv
            output_dir=META_FROZEN_DIR,
            se_multiplier=SE_MULTIPLIER,
        )
    else:
        print("[META] Found existing frozen baseline meta artifacts -> skip")

    # Backfill SHAP for new IDs using frozen pipe (append)
    missing_shap_ids = truth_ids - _ids_in_csv(shap_path)
    if len(missing_shap_ids) > 0 and os.path.exists(pipe_path):
        print(f"[META][FIX2] Backfill missing SHAP IDs={len(missing_shap_ids)} -> append")
        compute_shap_with_frozen_baseline(
            baseline_pipe_path=pipe_path,
            round_dir=ROUND0_DIR,                 # ✅ FIX: canonical round_dir
            detector_name=DETECTOR,
            out_dir=META_FROZEN_DIR,
            metrics_csv_name=METRICS_CSV_NAME,
            detector_csv_prefix=DETECTOR_PREFIX,
            target="label",
            only_ids=sorted(list(missing_shap_ids)),
            append_if_exists=True,
        )

    _require_file(coeffs_path, "Missing meta-model coefficients")
    _require_file(shap_path, "Missing SHAP per-ID attributions")
    _require_file(stats_path, "Missing baseline feature stats")
    _require_file(pipe_path, "Missing frozen meta-model pipe")

    # 7) Orchestrator baseline (per detector, per round) — uses ROUND0_DIR as round_dir
    decisions_existing_ids = _ids_in_csv(ORCH_DECISIONS_PATH)
    fake_truth_ids = set(df_det[df_det["pred_label"].astype(str).str.upper() == "FAKE"]["ID"].astype(str).tolist()) & truth_ids
    missing_orch_ids = fake_truth_ids - decisions_existing_ids

    if not (os.path.exists(ORCH_DECISIONS_PATH) and os.path.exists(ORCH_GUIDES_PATH)):
        print("[ORCH] Building baseline decisions (FAKE only)")
        df_decisions = run_orchestrator_round(
            detector_name=DETECTOR,
            round_dir=ROUND0_DIR,                 # ✅ FIX: canonical round_dir
            detector_csv_prefix=DETECTOR_PREFIX,  # reads ROUND0_DIR/detector_<detector>.csv
            filter_fake=True,
            meta_dir=META_FROZEN_DIR,             # reads SHAP/coeffs from meta_frozen
        )
        print(f"[ORCH] Decisions rows={0 if df_decisions is None else len(df_decisions)}")
    elif len(missing_orch_ids) > 0:
        print(f"[ORCH][FIX2] Backfill missing decisions IDs={len(missing_orch_ids)} -> append")
        df_decisions = run_orchestrator_round(
            detector_name=DETECTOR,
            round_dir=ROUND0_DIR,                 # ✅ FIX: canonical round_dir
            detector_csv_prefix=DETECTOR_PREFIX,
            filter_fake=True,
            meta_dir=META_FROZEN_DIR,
            only_ids=sorted(list(missing_orch_ids)),
            append_if_exists=True,
        )
        print(f"[ORCH] Backfill decisions rows={0 if df_decisions is None else len(df_decisions)}")
    else:
        print("[ORCH] Found existing decisions -> skip")

    _require_file(ORCH_DECISIONS_PATH, "Missing orchestrator decisions")
    _require_file(ORCH_GUIDES_PATH, "Missing orchestrator guides")

    print("\n[DONE] BASELINE ROUND0 COMPLETE")
    print(f"[OUT] Cohort (original):  {COHORT_ORIGINAL_PATH}")
    print(f"[OUT] Sample IDs:         {SAMPLE_IDS_PATH}")
    print(f"[OUT] Generated:          {GENERATED_PATH}")
    print(f"[OUT] Panel parquet:      {PANEL_PARQUET_PATH}")
    print(f"[OUT] Metrics:            {METRICS_MASTER_PATH}")
    print(f"[OUT] Detector:           {DETECTOR_OUT_PATH}")
    print(f"[OUT] Meta-Model frozen:  {META_FROZEN_DIR}")
    print(f"[OUT] Orch dec:           {ORCH_DECISIONS_PATH}")
    print(f"[OUT] Orch guides:        {ORCH_GUIDES_PATH}")
    print(f"[STATUS] FAKE={n_fake} | REAL={n_real}")
    return 0


class BaselinePipeline:
    """Baseline (round 0) pipeline runner.

    You can run it either:
      - programmatically: BaselinePipeline(...).run()
      - via CLI: python run_orchestrator_round0.py --nomedataset ...
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        detector: str | None = None,
        mode: str = "keep",
        desired_n: int = 100,
        random_state: int = 123,
        resume: bool = True,
        se_multiplier: float = 1.0,
    ):
        self.dataset_path = dataset_path
        self.detector = detector
        self.mode = mode
        self.desired_n = int(desired_n)
        self.random_state = int(random_state)
        self.resume = bool(resume)
        self.se_multiplier = float(se_multiplier)

    def run(self, argv=None) -> int:
        # If argv is provided, behave like the original script.
        if argv is not None:
            parser = build_parser()
            args = parser.parse_args(argv)
            return _run(args)

        # Programmatic run (no argv): create a compatible args namespace.
        if not self.dataset_path:
            raise ValueError("dataset_path is required")
        if not self.detector:
            raise ValueError("detector is required")

        class _Args:
            pass

        args = _Args()
        args.nomedataset = self.dataset_path
        args.nomedetector = self.detector
        args.mode = self.mode
        args.desired_n = self.desired_n
        args.random_state = self.random_state
        args.resume_pipeline = bool(self.resume)
        args.no_resume_pipeline = not bool(self.resume)
        args.se_multiplier = self.se_multiplier
        return _run(args)


def main(argv=None) -> int:
    return BaselinePipeline().run(argv)


if __name__ == "__main__":
    raise SystemExit(main())