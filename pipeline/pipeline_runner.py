from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List

from .run_round0_generation import Round0GenerationPipeline
from .run_round0_detector_baseline import Round0DetectorBaselinePipeline
from .run_orchestrator_round0 import BaselinePipeline  # legacy (all-in-one)
from .run_loop_rounds import LoopPipeline
from .constants import MIN_ACTIVE_FAKE


@dataclass
class PipelineConfig:
    dataset_path: Optional[str] = None
    detector: str = "valhalla"
    output_root: str = "Output"
    desired_n: int = 100
    random_state: int = 123
    # Max round for the loop stage.
    # - None or 0: no max-round limit (loop stops by its internal criteria)
    # - < 0: skip loop stage entirely
    max_round: Optional[int] = None
    resume: bool = True
    mode: str = "keep"  # gen0 sampling mode
    detectors: Optional[str] = None  # comma-separated list for multi-detector run
    se_multiplier: float = 1.0  # SE multiplier for SGL CV selection rule
    # Single uniform statistical-power lower bound (pipeline.constants): applied at
    # the round-0 SGL gate, every loop round, DDM inference, and the retrain gates.
    # Override via CLI --min-active-fake for small "test mini" cohorts.
    min_active_fake: int = MIN_ACTIVE_FAKE
    # When False, the loop skips drift monitoring AND SGL retrain entirely.
    enable_drift_retrain: bool = True
    # Framework-validity ablations (exp2/exp3), forwarded to the loop orchestrator.
    invert_direction: bool = False  # exp2: flip edit direction, same feature
    anti_feature: bool = False      # exp3: pick least-important feature, normal direction


class PipelineRunner:
    """Facade runner that composes gen0, baseline0, loop, and analysis with a uniform API."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    # -------------------------
    # Internal helpers
    # -------------------------
    def _stop_all_real_marker(self, detector: str) -> str:
        """
        Marker written by baseline0 when a detector predicts FAKE=0 (ALL REAL).
        In that case loop must be skipped (no meta-model pipe exists and no ACTIVE_FAKE to process).
        """
        return os.path.join(
            self.config.output_root,
            "rounds",
            "round_000",
            "orchestrator",
            detector,
            "STOP_ALL_REAL.txt",
        )

    def _stop_insufficient_fake_marker(self, detector: str) -> str:
        """
        Marker written by baseline0 when n_fake < MIN_ACTIVE_FAKE.
        SGL cannot be trained meaningfully; loop must be skipped.
        """
        return os.path.join(
            self.config.output_root,
            "rounds",
            "round_000",
            "orchestrator",
            detector,
            "STOP_INSUFFICIENT_FAKE.txt",
        )

    # -------------------------
    # Steps
    # -------------------------
    def run_gen0(self) -> int:
        """Run round 0 GENERATION only (isolated)."""
        if not self.config.dataset_path:
            raise ValueError("dataset_path is required for gen0")
        gp = Round0GenerationPipeline(
            dataset_path=self.config.dataset_path,
            mode=self.config.mode,
            desired_n=self.config.desired_n,
            random_state=self.config.random_state,
            resume=self.config.resume,
            output_root=self.config.output_root,
        )
        return int(gp.run(argv=None))

    def run_baseline0(self) -> int:
        """Run round 0 baseline for ONE detector (no generation)."""
        if not self.config.dataset_path:
            raise ValueError("dataset_path is required for baseline0")
        bp = Round0DetectorBaselinePipeline(
            dataset_path=self.config.dataset_path,
            detector=self.config.detector,
            resume=self.config.resume,
            se_multiplier=self.config.se_multiplier,
            invert_direction=self.config.invert_direction,
            anti_feature=self.config.anti_feature,
            output_root=self.config.output_root,
        )
        return int(bp.run(argv=None))

    def run_loop(self) -> int:
        """Run rounds > 0 up to max_round (for ONE detector)."""
        if not self.config.dataset_path:
            raise ValueError("dataset_path is required for loop")

        max_round = 0 if self.config.max_round is None else int(self.config.max_round)
        if max_round < 0:
            return 0

        lp = LoopPipeline(
            dataset_path=self.config.dataset_path,
            detector=self.config.detector,
            output_root=self.config.output_root,
            max_round=max_round,
            resume=self.config.resume,
            min_active_fake=self.config.min_active_fake,
            enable_drift_retrain=self.config.enable_drift_retrain,
            invert_direction=self.config.invert_direction,
            anti_feature=self.config.anti_feature,
        )
        return int(lp.run(argv=None))

    # -------------------------
    # Analysis (RQ1, RQ2, RQ3)
    # -------------------------
    def run_rq1(self) -> int:
        from .rq1 import run as run_rq1
        return run_rq1(output_root=self.config.output_root)

    def run_rq2(self, embedding_model: str = "all-MiniLM-L6-v2", epsilon: float = 0.10) -> int:
        from .rq2 import run as run_rq2
        return run_rq2(output_root=self.config.output_root, embedding_model=embedding_model, epsilon=epsilon)

    def run_rq3(self) -> int:
        from .rq3 import run as run_rq3
        return run_rq3(output_root=self.config.output_root)

    # -------------------------
    # High-level orchestration
    # -------------------------
    def run_all_one_detector(self) -> dict:
        """gen0 -> baseline0 -> loop (ONE detector)."""
        codes = {}
        codes["gen0"] = int(self.run_gen0())
        codes["baseline0"] = int(self.run_baseline0())

        # skip loop if baseline0 wrote STOP_ALL_REAL or STOP_INSUFFICIENT_FAKE
        if os.path.exists(self._stop_all_real_marker(self.config.detector)):
            print(f"[RUNNER] ALL_REAL for detector={self.config.detector} -> skipping loop.")
            codes["loop"] = 0
        elif os.path.exists(self._stop_insufficient_fake_marker(self.config.detector)):
            print(f"[RUNNER] INSUFFICIENT_FAKE for detector={self.config.detector} -> skipping loop.")
            codes["loop"] = 0
        else:
            codes["loop"] = int(self.run_loop())

        return codes

    def run_all_detectors(self, detectors: Optional[List[str]] = None) -> dict:
        """One command: gen0 ONCE, then (baseline0 -> loop) sequentially for each detector."""
        if detectors is None:
            if self.config.detectors:
                detectors = [d.strip() for d in self.config.detectors.split(",") if d.strip()]
            else:
                detectors = [self.config.detector]

        codes = {"gen0": int(self.run_gen0()), "detectors": {}}

        for det in detectors:
            self.config.detector = det
            det_codes = {}

            det_codes["baseline0"] = int(self.run_baseline0())

            # skip loop if baseline0 wrote STOP_ALL_REAL or STOP_INSUFFICIENT_FAKE
            if os.path.exists(self._stop_all_real_marker(det)):
                print(f"[RUNNER] ALL_REAL for detector={det} -> skipping loop, moving to next detector.")
                det_codes["loop"] = 0
            elif os.path.exists(self._stop_insufficient_fake_marker(det)):
                print(f"[RUNNER] INSUFFICIENT_FAKE for detector={det} -> skipping loop, moving to next detector.")
                det_codes["loop"] = 0
            else:
                det_codes["loop"] = int(self.run_loop())

            codes["detectors"][det] = det_codes

        return codes

    # ---------------- Legacy ----------------
    def run_baseline_legacy(self) -> int:
        """LEGACY: round0 generation + detector + meta + orchestrator in one shot."""
        if not self.config.dataset_path:
            raise ValueError("dataset_path is required for baseline legacy")
        bp = BaselinePipeline(
            dataset_path=self.config.dataset_path,
            detector=self.config.detector,
            mode=self.config.mode,
            desired_n=self.config.desired_n,
            random_state=self.config.random_state,
            resume=self.config.resume,
            se_multiplier=self.config.se_multiplier,
        )
        return int(bp.run(argv=None))
