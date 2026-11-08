"""Shared numeric constants for the pipeline.

Single source of truth for the minimum-active-FAKE lower bound, applied
uniformly across every stage: round-0 baseline (SGL training gate), every
iterative loop round, DDM surrogate-fidelity inference, and the SGL retrain
buffer gates.
"""

MIN_ACTIVE_FAKE = 40
