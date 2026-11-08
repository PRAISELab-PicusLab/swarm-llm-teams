import os
import shutil
import argparse
import pandas as pd
import torch

from .pipeline_multiagent import run_generation_pipeline
from .metrics_runner import compute_agent_metrics, write_baseline_references
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

def _order_df_by_sample_ids(df: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    df2 = df.copy()
    df2["ID"] = df2["ID"].astype(str)
    ord_map = {sid: i for i, sid in enumerate(sample_ids)}
    df2["__ord__"] = df2["ID"].map(ord_map).fillna(10**12).astype(int)
    df2 = df2.sort_values("__ord__").drop(columns=["__ord__"])
    return df2

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
    parser = argparse.ArgumentParser(description="Round 0 generation (isolated)")

    parser.add_argument("--nomedataset", type=str, required=True, help="Path dataset CSV con colonna ID")

    # sample ids behavior
    parser.add_argument("--mode", type=str, default="keep", choices=["keep", "expand", "rebuild"], help="Mode sample ids")
    parser.add_argument("--desired_n", type=int, default=100, help="Numero ID sample (round0)")
    parser.add_argument("--random_state", type=int, default=123, help="Seed deterministic")
    parser.add_argument("--output_root", type=str, default="Output", help="Output root (default: Output)")

    # resume behavior
    parser.add_argument("--resume_pipeline", action="store_true", help="Resume pipeline generation (default True)")
    parser.add_argument("--no_resume_pipeline", action="store_true", help="Disable resume pipeline generation")
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
    ROUND0_SAMPLE_SIZE = int(args.desired_n)
    RANDOM_STATE = int(args.random_state)

    # OUTPUT STRUCTURE
    OUTPUT_ROOT = getattr(args, "output_root", "Output") or "Output"
    COHORT_DIR = os.path.join(OUTPUT_ROOT, "cohort")
    TRACKING_DIR = os.path.join(OUTPUT_ROOT, "tracking")  # kept for compatibility
    ROUNDS_DIR = os.path.join(OUTPUT_ROOT, "rounds")

    ROUND0_DIR = os.path.join(ROUNDS_DIR, ROUND_TAG)
    GEN_DIR = os.path.join(ROUND0_DIR, "generation")

    # files (GEN ONLY)
    SAMPLE_IDS_PATH = os.path.join(COHORT_DIR, "cohort_sample_ids.csv")
    _SAMPLE_IDS_LEGACY_PATH = "dataset/dataset_sample_ids.csv"
    COHORT_ORIGINAL_PATH = os.path.join(COHORT_DIR, "cohort_original.csv")
    GENERATED_PATH = os.path.join(GEN_DIR, "Generated_Fake_News.csv")

    METRICS_CSV_NAME = "agent_metrics.csv"
    METRICS_MASTER_PATH = os.path.join(GEN_DIR, METRICS_CSV_NAME)

    PANEL_PARQUET_PATH = os.path.join(GEN_DIR, "panel_articles_round_r.parquet")

    # dirs (NOTE: we DO NOT create detectors/ or orchestrator/ here)
    _ensure_dir(COHORT_DIR)
    _ensure_dir(TRACKING_DIR)
    _ensure_dir(ROUND0_DIR)
    _ensure_dir(GEN_DIR)

    print(f"[INFO] Torch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"[INFO] ROUND0 generation (isolated) -> {ROUND_TAG}")
    print(f"[INFO] Dataset:  {DATASET_PATH}")
    print(f"[INFO] mode={args.mode} desired_n={ROUND0_SAMPLE_SIZE} seed={RANDOM_STATE}")
    print(f"[INFO] resume_pipeline={resume_pipeline}")
    print(f"[INFO] GEN_DIR: {GEN_DIR}")

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

    # ── Cleanup logic based on mode ──
    # rebuild: clean EVERYTHING (generation + downstream)
    # expand: keep generation, clean downstream
    # keep: nothing to clean

    def _clean_downstream(include_generation: bool):
        """Clean artifacts downstream of sample_ids change."""
        if include_generation:
            # 1. Generation round 0
            if os.path.exists(GEN_DIR):
                shutil.rmtree(GEN_DIR)
            _ensure_dir(GEN_DIR)
            print(f"[CLEANUP]   Removed {GEN_DIR}")
        # 2. Cohort (depends on sample IDs)
        if os.path.exists(COHORT_ORIGINAL_PATH):
            os.remove(COHORT_ORIGINAL_PATH)
            print(f"[CLEANUP]   Removed {COHORT_ORIGINAL_PATH}")
        # 3. Round 0 detector/orchestrator/shap (depend on generation)
        for subdir in ("detectors", "orchestrator", "orch", "shap"):
            p = os.path.join(ROUND0_DIR, subdir)
            if os.path.exists(p):
                shutil.rmtree(p)
                print(f"[CLEANUP]   Removed {p}")
        # 3b. Stray CSV copies at round_000 root (created by baseline0)
        if os.path.exists(ROUND0_DIR):
            for f in os.listdir(ROUND0_DIR):
                if f.endswith(".csv") and os.path.isfile(os.path.join(ROUND0_DIR, f)):
                    fp = os.path.join(ROUND0_DIR, f)
                    os.remove(fp)
                    print(f"[CLEANUP]   Removed {fp}")
        # 4. Meta frozen (depends on round 0 detector + metrics)
        meta_frozen_dir = os.path.join(OUTPUT_ROOT, "meta_frozen")
        if os.path.exists(meta_frozen_dir):
            shutil.rmtree(meta_frozen_dir)
            print(f"[CLEANUP]   Removed {meta_frozen_dir}")
        # 5. Tracking / ledger (depends on meta frozen + orchestrator)
        if os.path.exists(TRACKING_DIR):
            shutil.rmtree(TRACKING_DIR)
            _ensure_dir(TRACKING_DIR)
            print(f"[CLEANUP]   Cleaned {TRACKING_DIR}")
        # 6. Rounds > 0 (depend on everything above)
        if os.path.exists(ROUNDS_DIR):
            for entry in os.listdir(ROUNDS_DIR):
                if entry.startswith("round_") and entry != ROUND_TAG:
                    p = os.path.join(ROUNDS_DIR, entry)
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                        print(f"[CLEANUP]   Removed {p}")

    if args.mode == "rebuild":
        print("[REBUILD] Cleaning ALL outputs (generation + downstream) to start from scratch...")
        try:
            _clean_downstream(include_generation=True)
        except Exception as e:
            raise RuntimeError(f"[REBUILD] Failed cleaning: {e}")

    elif args.mode == "expand":
        print(f"[EXPAND] Cleaning downstream artifacts (keeping generation)...")
        try:
            _clean_downstream(include_generation=False)
        except Exception as e:
            raise RuntimeError(f"[EXPAND] Failed cleaning downstream: {e}")

    # 1) load dataset
    _require_file(DATASET_PATH, "Dataset not found")
    df_real = _normalize_id(_load_csv(DATASET_PATH))
    print(f"[DATASET] rows={len(df_real)} path={DATASET_PATH}")

    # load sample ids (and preserve order)
    if not os.path.exists(SAMPLE_IDS_PATH) and os.path.exists(_SAMPLE_IDS_LEGACY_PATH):
        print(f"[WARN] cohort_sample_ids.csv not found at {SAMPLE_IDS_PATH} "
              f"-- falling back to legacy path {_SAMPLE_IDS_LEGACY_PATH}")
        SAMPLE_IDS_PATH = _SAMPLE_IDS_LEGACY_PATH
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

    if not (os.path.exists(GENERATED_PATH) and os.path.getsize(GENERATED_PATH) > 0):
        raise FileNotFoundError(
            f"[ROUND0] Generated_Fake_News.csv is missing at {GENERATED_PATH}. "
            f"If you intentionally skipped generation (resume_pipeline=False), you must provide this file."
        )

    df_generated = _normalize_id(_load_csv(GENERATED_PATH))
    print(f"[GEN] rows in Generated_Fake_News.csv: {len(df_generated)}")

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

    # 4) Metrics (compute in GEN_DIR)
    RESIDUALIZER_PATH = os.path.join(OUTPUT_ROOT, "meta_frozen", "residualizer.json")
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
    else:
        print(f"[METRICS] Found master -> {METRICS_MASTER_PATH}")

    _require_file(METRICS_MASTER_PATH, "Missing metrics csv after metrics computation")

    print("\n[DONE] ROUND0 GENERATION COMPLETE (ISOLATED)")
    print(f"[OUT] Cohort (original): {COHORT_ORIGINAL_PATH}")
    print(f"[OUT] Sample IDs:        {SAMPLE_IDS_PATH}")
    print(f"[OUT] Generated:         {GENERATED_PATH}")
    print(f"[OUT] Panel parquet:     {PANEL_PARQUET_PATH}")
    print(f"[OUT] Metrics (master):  {METRICS_MASTER_PATH}")
    return 0


class Round0GenerationPipeline:
    def __init__(
        self,
        dataset_path: str | None = None,
        mode: str = "keep",
        desired_n: int = 100,
        random_state: int = 123,
        resume: bool = True,
        output_root: str = "Output",
    ):
        self.dataset_path = dataset_path
        self.mode = mode
        self.desired_n = int(desired_n)
        self.random_state = int(random_state)
        self.resume = bool(resume)
        self.output_root = output_root or "Output"

    def run(self, argv=None) -> int:
        if argv is not None:
            parser = build_parser()
            args = parser.parse_args(argv)
            return _run(args)

        if not self.dataset_path:
            raise ValueError("dataset_path is required")

        class _Args:
            pass

        args = _Args()
        args.nomedataset = self.dataset_path
        args.mode = self.mode
        args.desired_n = self.desired_n
        args.random_state = self.random_state
        args.resume_pipeline = bool(self.resume)
        args.no_resume_pipeline = not bool(self.resume)
        args.output_root = self.output_root
        return _run(args)


def main(argv=None) -> int:
    return Round0GenerationPipeline().run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
