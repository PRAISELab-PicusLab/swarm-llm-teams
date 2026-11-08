import argparse
from .run_loop_rounds import LoopPipeline

DEFAULT_DETECTORS = ["roberta", "bert", "bart", "valhalla"]

def build_parser():
    p = argparse.ArgumentParser(description="Run loop pipeline sequentially for multiple detectors (round0 shared).")
    p.add_argument("--nomedataset", required=True, help="Path to dataset CSV")
    p.add_argument("--output_root", default="Output", help="Output root folder")
    p.add_argument("--max_round", type=int, default=0, help="Max rounds (0 means run until stop condition)")
    p.add_argument("--detectors", default=",".join(DEFAULT_DETECTORS),
                   help="Comma-separated list of detectors (default: roberta,bert,bart,valhalla)")
    p.add_argument("--no_resume", action="store_true", help="Disable resume (recompute even if files exist)")
    return p

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    resume = not bool(args.no_resume)

    # Run sequentially. Round0 generation is shared and will be reused via resume logic.
    for det in detectors:
        print("\n" + "#"*110)
        print(f"[RUN-ALL] Detector = {det}")
        print("#"*110)
        LoopPipeline(
            dataset_path=args.nomedataset,
            detector=det,
            output_root=args.output_root,
            max_round=args.max_round,
            resume=resume,
        ).run(argv=None)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
