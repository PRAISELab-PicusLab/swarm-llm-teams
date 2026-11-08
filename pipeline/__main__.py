from __future__ import annotations

import argparse

from .pipeline_runner import PipelineRunner, PipelineConfig
from .constants import MIN_ACTIVE_FAKE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Main pipeline commands."
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------------
    # Round 0
    # ------------------------------------------------------------------

    pg = sub.add_parser(
        "gen0",
        help="Round 0: generation ONLY (isolated)."
    )

    pg.add_argument("--dataset", type=str, required=True)
    pg.add_argument("--mode", type=str, default="keep",
                    choices=["keep", "expand", "rebuild"])
    pg.add_argument("--desired-n", type=int, default=100)
    pg.add_argument("--random-state", type=int, default=123)
    pg.add_argument("--resume", action="store_true")
    pg.add_argument("--no-resume", action="store_true")

    # ------------------------------------------------------------------
    # Baseline round 0
    # ------------------------------------------------------------------

    pb = sub.add_parser(
        "baseline0",
        help="Round 0: baseline for ONE detector (NO generation)."
    )

    pb.add_argument("--dataset", type=str, required=True)
    pb.add_argument("--detector", type=str, default="valhalla")
    pb.add_argument("--resume", action="store_true")
    pb.add_argument("--no-resume", action="store_true")

    pb.add_argument(
        "--se-multiplier",
        type=float,
        default=1.0,
        help="SE multiplier for SGL CV selection rule"
    )

    pb.add_argument("--invert-direction", action="store_true",
                    help="exp2 ablation: flip direction in round-0 decisions (drives round 1).")
    pb.add_argument("--anti-feature", action="store_true",
                    help="exp3 ablation: least-important feature in round-0 decisions (drives round 1).")

    # ------------------------------------------------------------------
    # Backward compatibility alias
    # ------------------------------------------------------------------

    p0 = sub.add_parser(
        "baseline",
        help="Alias of baseline0 (NO generation)."
    )

    p0.add_argument("--dataset", type=str, required=True)
    p0.add_argument("--detector", type=str, default="valhalla")
    p0.add_argument("--resume", action="store_true")
    p0.add_argument("--no-resume", action="store_true")

    p0.add_argument(
        "--se-multiplier",
        type=float,
        default=1.0,
        help="SE multiplier for SGL CV selection rule"
    )

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    pl = sub.add_parser(
        "loop",
        help="Run iterative rounds > 0 (ONE detector)."
    )

    pl.add_argument("--dataset", type=str, required=True)
    pl.add_argument("--detector", type=str, default="valhalla")

    pl.add_argument(
        "--max-round",
        type=int,
        default=None,
        help=(
            "Max round to run. "
            "None or 0 = no limit. "
            "-1 = skip loop."
        )
    )

    pl.add_argument("--resume", action="store_true")
    pl.add_argument("--no-resume", action="store_true")

    pl.add_argument(
        "--min-active-fake",
        type=int,
        default=MIN_ACTIVE_FAKE,
        help=(
            "Minimum ACTIVE_FAKE per round to keep iterating "
            f"(default: {MIN_ACTIVE_FAKE}). Use a smaller value for test-mini cohorts."
        )
    )

    pl.add_argument(
        "--no-drift-retrain",
        action="store_true",
        help=(
            "Disable drift monitoring AND SGL retrain entirely. The loop still "
            "scores/edits articles and tracks REAL flips."
        )
    )

    pl.add_argument("--invert-direction", action="store_true",
                    help="exp2 ablation: same chosen feature, FLIP edit direction (INCREASE<->DECREASE).")
    pl.add_argument("--anti-feature", action="store_true",
                    help="exp3 ablation: pick LEAST important feature (min |SHAP|), normal direction.")

    # ------------------------------------------------------------------
    # Loop multi-detector
    # ------------------------------------------------------------------

    pl2 = sub.add_parser(
        "loop-detectors",
        help=(
            "Run iterative loop for MULTIPLE detectors sequentially "
            "(requires baseline0 already done for each)."
        ),
    )

    pl2.add_argument("--dataset", type=str, required=True)

    pl2.add_argument(
        "--detectors",
        type=str,
        default="roberta,valhalla",
        help="Comma-separated detector list"
    )

    pl2.add_argument(
        "--max-round",
        type=int,
        default=None,
        help=(
            "Max round per detector. "
            "None or 0 = no limit. "
            "-1 = skip loop."
        )
    )

    pl2.add_argument("--resume", action="store_true")
    pl2.add_argument("--no-resume", action="store_true")

    pl2.add_argument(
        "--min-active-fake",
        type=int,
        default=MIN_ACTIVE_FAKE,
        help=(
            "Minimum ACTIVE_FAKE per round to keep iterating "
            f"(default: {MIN_ACTIVE_FAKE}). Use a smaller value for test-mini cohorts."
        )
    )

    pl2.add_argument(
        "--no-drift-retrain",
        action="store_true",
        help=(
            "Disable drift monitoring AND SGL retrain entirely for every detector's loop."
        )
    )

    pl2.add_argument("--invert-direction", action="store_true",
                     help="exp2 ablation: same chosen feature, FLIP edit direction (INCREASE<->DECREASE).")
    pl2.add_argument("--anti-feature", action="store_true",
                     help="exp3 ablation: pick LEAST important feature (min |SHAP|), normal direction.")

    # ------------------------------------------------------------------
    # Run all one detector
    # ------------------------------------------------------------------

    pr = sub.add_parser(
        "runall",
        help="ONE detector: gen0 -> baseline0 -> loop."
    )

    pr.add_argument("--dataset", type=str, required=True)
    pr.add_argument("--detector", type=str, default="valhalla")
    pr.add_argument(
        "--output-root",
        type=str,
        default="Output",
        help="Root output dir (e.g. Output_ATN / Output_Isot for parallel jobs)."
    )

    pr.add_argument(
        "--mode",
        type=str,
        default="keep",
        choices=["keep", "expand", "rebuild"]
    )

    pr.add_argument("--desired-n", type=int, default=100)
    pr.add_argument("--random-state", type=int, default=123)

    pr.add_argument(
        "--max-round",
        type=int,
        default=None,
        help=(
            "Max round to run. "
            "None or 0 = no limit. "
            "-1 = skip loop."
        )
    )

    pr.add_argument("--resume", action="store_true")
    pr.add_argument("--no-resume", action="store_true")

    pr.add_argument(
        "--se-multiplier",
        type=float,
        default=1.0,
        help="SE multiplier for SGL CV selection rule"
    )

    pr.add_argument(
        "--min-active-fake",
        type=int,
        default=MIN_ACTIVE_FAKE,
        help=(
            "Min ACTIVE_FAKE per round to keep iterating "
            f"(default: {MIN_ACTIVE_FAKE}). Use a smaller value for test-mini cohorts."
        )
    )

    pr.add_argument(
        "--no-drift-retrain",
        action="store_true",
        help=(
            "Disable drift monitoring AND SGL retrain entirely for the loop stage."
        )
    )

    pr.add_argument("--invert-direction", action="store_true",
                    help="exp2 ablation: same chosen feature, FLIP edit direction (INCREASE<->DECREASE).")
    pr.add_argument("--anti-feature", action="store_true",
                    help="exp3 ablation: pick LEAST important feature (min |SHAP|), normal direction.")

    # ------------------------------------------------------------------
    # Run all detectors
    # ------------------------------------------------------------------

    pm = sub.add_parser(
        "runall-detectors",
        help=(
            "MULTI detector: gen0 once, then baseline0+loop "
            "sequentially for each detector."
        )
    )

    pm.add_argument("--dataset", type=str, required=True)

    pm.add_argument(
        "--output-root",
        type=str,
        default="Output",
        help="Root output dir (e.g. Output_ATN / Output_Isot for parallel jobs)."
    )

    pm.add_argument(
        "--detectors",
        type=str,
        default="bart,bert,roberta,valhalla",
        help="Comma-separated detectors"
    )

    pm.add_argument(
        "--mode",
        type=str,
        default="keep",
        choices=["keep", "expand", "rebuild"]
    )

    pm.add_argument("--desired-n", type=int, default=100)
    pm.add_argument("--random-state", type=int, default=123)

    pm.add_argument(
        "--max-round",
        type=int,
        default=None,
        help=(
            "Max round per detector. "
            "None or 0 = no limit. "
            "-1 = skip loop."
        )
    )

    pm.add_argument("--resume", action="store_true")
    pm.add_argument("--no-resume", action="store_true")

    pm.add_argument(
        "--se-multiplier",
        type=float,
        default=1.0,
        help="SE multiplier for SGL CV selection rule"
    )

    pm.add_argument(
        "--min-active-fake",
        type=int,
        default=MIN_ACTIVE_FAKE,
        help=(
            "Min ACTIVE_FAKE per round to keep iterating "
            f"(default: {MIN_ACTIVE_FAKE}). Use a smaller value for test-mini cohorts."
        )
    )

    pm.add_argument(
        "--no-drift-retrain",
        action="store_true",
        help=(
            "Disable drift monitoring AND SGL retrain entirely for every detector's loop."
        )
    )

    pm.add_argument("--invert-direction", action="store_true",
                    help="exp2 ablation: same chosen feature, FLIP edit direction (INCREASE<->DECREASE).")
    pm.add_argument("--anti-feature", action="store_true",
                    help="exp3 ablation: pick LEAST important feature (min |SHAP|), normal direction.")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    prq1 = sub.add_parser(
        "rq1",
        help="RQ1 analysis: fooling rate."
    )

    prq1.add_argument("--output_root", type=str, default="Output")

    prq2 = sub.add_parser(
        "rq2",
        help="RQ2 analysis: semantic distance."
    )

    prq2.add_argument("--output_root", type=str, default="Output")

    prq2.add_argument(
        "--embedding_model",
        type=str,
        default="all-MiniLM-L6-v2"
    )

    prq3 = sub.add_parser(
        "rq3",
        help="RQ3 analysis: structural features."
    )

    prq3.add_argument("--output_root", type=str, default="Output")

    # ------------------------------------------------------------------
    # Legacy
    # ------------------------------------------------------------------

    plg = sub.add_parser(
        "baseline-legacy",
        help=(
            "LEGACY: round0 generation+detector+meta+orchestrator "
            "in one shot."
        )
    )

    plg.add_argument("--dataset", type=str, required=True)
    plg.add_argument("--detector", type=str, default="valhalla")

    plg.add_argument(
        "--mode",
        type=str,
        default="keep",
        choices=["keep", "expand", "rebuild"]
    )

    plg.add_argument("--desired-n", type=int, default=100)
    plg.add_argument("--random-state", type=int, default=123)

    plg.add_argument("--resume", action="store_true")
    plg.add_argument("--no-resume", action="store_true")

    plg.add_argument(
        "--se-multiplier",
        type=float,
        default=1.0,
        help="SE multiplier for SGL CV selection rule"
    )

    plg.add_argument(
        "--min-active-fake",
        type=int,
        default=MIN_ACTIVE_FAKE,
        help=(
            "Minimum ACTIVE_FAKE per round to keep iterating "
            f"(default: {MIN_ACTIVE_FAKE}). Use a smaller value for test-mini cohorts."
        )
    )

    plg.add_argument(
        "--max-round",
        type=int,
        default=None,
        help=(
            "Max round to run. "
            "None or 0 = no limit. "
            "-1 = skip loop."
        )
    )

    return p


def main(argv=None) -> int:

    args = build_parser().parse_args(argv)

    # ------------------------------------------------------------------
    # Resume policy
    # Default = True unless explicitly disabled
    # ------------------------------------------------------------------

    resume = not bool(getattr(args, "no_resume", False))

    if bool(getattr(args, "resume", False)):
        resume = True

    # ------------------------------------------------------------------
    # gen0
    # ------------------------------------------------------------------

    if args.cmd == "gen0":

        cfg = PipelineConfig(
            dataset_path=args.dataset,
            detector="none",
            output_root="Output",
            desired_n=int(args.desired_n),
            random_state=int(args.random_state),
            max_round=None,
            resume=resume,
            mode=args.mode,
        )

        return int(PipelineRunner(cfg).run_gen0())

    # ------------------------------------------------------------------
    # baseline0 / baseline
    # ------------------------------------------------------------------

    if args.cmd in ("baseline0", "baseline"):

        cfg = PipelineConfig(
            dataset_path=args.dataset,
            detector=args.detector,
            output_root="Output",
            resume=resume,
            se_multiplier=getattr(args, "se_multiplier", 1.0),
        )

        return int(PipelineRunner(cfg).run_baseline0())

    # ------------------------------------------------------------------
    # loop
    # ------------------------------------------------------------------

    if args.cmd == "loop":

        if args.max_round == -1:
            print("[LOOP] Skipped because max_round == -1")
            return 0

        cfg = PipelineConfig(
            dataset_path=args.dataset,
            detector=args.detector,
            output_root="Output",
            max_round=args.max_round,
            resume=resume,
            min_active_fake=getattr(args, "min_active_fake", MIN_ACTIVE_FAKE),
            enable_drift_retrain=not getattr(args, "no_drift_retrain", False),
            invert_direction=getattr(args, "invert_direction", False),
            anti_feature=getattr(args, "anti_feature", False),
        )

        return int(PipelineRunner(cfg).run_loop())

    # ------------------------------------------------------------------
    # loop-detectors
    # ------------------------------------------------------------------

    if args.cmd == "loop-detectors":

        if args.max_round == -1:
            print("[LOOP-DETECTORS] Skipped because max_round == -1")
            return 0

        detectors = [
            d.strip()
            for d in args.detectors.split(",")
            if d.strip()
        ]

        for det in detectors:

            print(
                f"\n{'=' * 80}\n"
                f"[LOOP-DETECTORS] Running loop for detector={det}\n"
                f"{'=' * 80}"
            )

            cfg = PipelineConfig(
                dataset_path=args.dataset,
                detector=det,
                output_root="Output",
                max_round=args.max_round,
                resume=resume,
                min_active_fake=getattr(args, "min_active_fake", MIN_ACTIVE_FAKE),
                enable_drift_retrain=not getattr(args, "no_drift_retrain", False),
                invert_direction=getattr(args, "invert_direction", False),
                anti_feature=getattr(args, "anti_feature", False),
            )

            PipelineRunner(cfg).run_loop()

        return 0

    # ------------------------------------------------------------------
    # runall
    # ------------------------------------------------------------------

    if args.cmd == "runall":

        cfg = PipelineConfig(
            dataset_path=args.dataset,
            detector=args.detector,
            output_root=getattr(args, "output_root", "Output"),
            desired_n=int(args.desired_n),
            random_state=int(args.random_state),
            max_round=args.max_round,
            resume=resume,
            mode=args.mode,
            se_multiplier=getattr(args, "se_multiplier", 1.0),
            min_active_fake=getattr(args, "min_active_fake", MIN_ACTIVE_FAKE),
            enable_drift_retrain=not getattr(args, "no_drift_retrain", False),
            invert_direction=getattr(args, "invert_direction", False),
            anti_feature=getattr(args, "anti_feature", False),
        )

        PipelineRunner(cfg).run_all_one_detector()

        return 0

    # ------------------------------------------------------------------
    # runall-detectors
    # ------------------------------------------------------------------

    if args.cmd == "runall-detectors":

        cfg = PipelineConfig(
            dataset_path=args.dataset,
            detector="none",
            output_root=getattr(args, "output_root", "Output"),
            desired_n=int(args.desired_n),
            random_state=int(args.random_state),
            max_round=args.max_round,
            resume=resume,
            mode=args.mode,
            detectors=args.detectors,
            se_multiplier=getattr(args, "se_multiplier", 1.0),
            min_active_fake=getattr(args, "min_active_fake", MIN_ACTIVE_FAKE),
            enable_drift_retrain=not getattr(args, "no_drift_retrain", False),
            invert_direction=getattr(args, "invert_direction", False),
            anti_feature=getattr(args, "anti_feature", False),
        )

        PipelineRunner(cfg).run_all_detectors()

        return 0

    # ------------------------------------------------------------------
    # RQ1
    # ------------------------------------------------------------------

    if args.cmd == "rq1":

        from .rq1 import run as run_rq1

        return run_rq1(
            output_root=args.output_root
        )

    # ------------------------------------------------------------------
    # RQ2
    # ------------------------------------------------------------------

    if args.cmd == "rq2":

        from .rq2 import run as run_rq2

        return run_rq2(
            output_root=args.output_root,
            embedding_model=args.embedding_model
        )

    # ------------------------------------------------------------------
    # RQ3
    # ------------------------------------------------------------------

    if args.cmd == "rq3":

        from .rq3 import run as run_rq3

        return run_rq3(
            output_root=args.output_root
        )

    # ------------------------------------------------------------------
    # baseline legacy
    # ------------------------------------------------------------------

    if args.cmd == "baseline-legacy":

        cfg = PipelineConfig(
            dataset_path=args.dataset,
            detector=args.detector,
            output_root="Output",
            desired_n=int(args.desired_n),
            random_state=int(args.random_state),
            max_round=args.max_round,
            resume=resume,
            mode=args.mode,
            se_multiplier=getattr(args, "se_multiplier", 1.0),
            min_active_fake=getattr(args, "min_active_fake", MIN_ACTIVE_FAKE),
        )

        return int(PipelineRunner(cfg).run_baseline_legacy())

    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())