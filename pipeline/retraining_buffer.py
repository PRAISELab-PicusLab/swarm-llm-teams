"""PUNTO 4 — Retraining SGL su buffer cumulativo.

Provides:
- build_real_baseline:       snapshot REAL articles from round 0 (frozen).
- update_fake_accumulated:   upsert FAKE from each round (last_round wins).
- build_balanced_training_set: real_baseline + (subsampled) fake_accumulated.
- do_retrain_and_reload:     archive frozen, retrain SGL in-place, recompute
                             SHAP, log beta-sign flips, update feature_meta.

Storage layout:
    Output/retraining_buffer/<detector>/
        real_baseline.parquet
        fake_accumulated.parquet
        buffer_stats.json
        retrain_log.csv
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from .metrics_runner import METRICS_REGISTRY
from .meta_model import (
    train_meta_model_for_detector,
    compute_shap_with_frozen_baseline,
)
from .constants import MIN_ACTIVE_FAKE

FEATURE_COLS = list(METRICS_REGISTRY.keys())
# Both retrain gates share the single uniform lower bound (pipeline.constants):
# the pre-balance buffer must hold ≥ MIN_ACTIVE_FAKE FAKE rows, and the
# post-balance training set must hold ≥ MIN_ACTIVE_FAKE total samples.
RETRAIN_MIN_FAKE = MIN_ACTIVE_FAKE
MIN_TOTAL_SAMPLES = MIN_ACTIVE_FAKE


def _buffer_dir(output_root: str, detector: str) -> str:
    return os.path.join(output_root, "retraining_buffer", detector)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ─────────────────────────────────────────────
# Real baseline (one-shot, frozen)
# ─────────────────────────────────────────────
def build_real_baseline(round0_metrics_csv: str, round0_detector_csv: str,
                        buffer_dir: str) -> int:
    """Snapshot REAL articles from round 0. Called once at end of baseline0."""
    _ensure_dir(buffer_dir)
    out = os.path.join(buffer_dir, "real_baseline.parquet")
    if os.path.exists(out):
        try:
            n = len(pd.read_parquet(out))
            print(f"[BUFFER] real_baseline.parquet already exists (n={n}) — skip")
            return n
        except Exception:
            pass  # corrupt → rebuild

    if not os.path.exists(round0_metrics_csv):
        print(f"[BUFFER][WARN] missing round0 metrics csv: {round0_metrics_csv}")
        return 0
    if not os.path.exists(round0_detector_csv):
        print(f"[BUFFER][WARN] missing round0 detector csv: {round0_detector_csv}")
        return 0

    dfm = pd.read_csv(round0_metrics_csv, encoding="utf-8")
    dfd = pd.read_csv(round0_detector_csv, encoding="utf-8")
    dfm["ID"] = dfm["ID"].astype(str)
    dfd["ID"] = dfd["ID"].astype(str)

    feat_present = [c for c in FEATURE_COLS if c in dfm.columns]
    df = dfm[["ID"] + feat_present].merge(
        dfd[["ID", "pred_label", "confidence"]], on="ID", how="inner"
    )
    real = df[df["pred_label"].astype(str).str.upper() == "REAL"].copy()
    real["label"] = 1
    real["source_round"] = 0
    real.to_parquet(out, index=False)
    print(f"[BUFFER] built real_baseline.parquet n_real={len(real)} cols={feat_present}")
    return len(real)


# ─────────────────────────────────────────────
# Fake accumulated (per-round upsert)
# ─────────────────────────────────────────────
def update_fake_accumulated(buffer_dir: str, current_round: int,
                            round_metrics_csv: str,
                            round_detector_csv: str) -> int:
    """Upsert FAKE entries from the current round into fake_accumulated.parquet."""
    _ensure_dir(buffer_dir)
    if not os.path.exists(round_metrics_csv) or not os.path.exists(round_detector_csv):
        return _safe_count(os.path.join(buffer_dir, "fake_accumulated.parquet"))

    try:
        dfm = pd.read_csv(round_metrics_csv, encoding="utf-8")
        dfd = pd.read_csv(round_detector_csv, encoding="utf-8")
    except Exception as e:
        print(f"[BUFFER][WARN] could not read round csvs: {e}")
        return _safe_count(os.path.join(buffer_dir, "fake_accumulated.parquet"))

    dfm["ID"] = dfm["ID"].astype(str)
    dfd["ID"] = dfd["ID"].astype(str)
    feat_present = [c for c in FEATURE_COLS if c in dfm.columns]
    df = dfm[["ID"] + feat_present].merge(
        dfd[["ID", "pred_label", "confidence"]], on="ID", how="inner"
    )
    fake = df[df["pred_label"].astype(str).str.upper() == "FAKE"].copy()
    if fake.empty:
        return _safe_count(os.path.join(buffer_dir, "fake_accumulated.parquet"))
    fake["label"] = 0
    fake["last_seen_round"] = int(current_round)
    fake["source_round"] = int(current_round)

    out = os.path.join(buffer_dir, "fake_accumulated.parquet")
    if os.path.exists(out):
        try:
            prev = pd.read_parquet(out)
            merged = pd.concat([prev, fake], ignore_index=True)
            merged = merged.sort_values("last_seen_round").drop_duplicates(
                subset=["ID"], keep="last"
            )
            merged.to_parquet(out, index=False)
            return len(merged)
        except Exception as e:
            print(f"[BUFFER][WARN] could not merge with existing parquet: {e} — overwriting")
    fake.to_parquet(out, index=False)
    return len(fake)


def _safe_count(parquet_path: str) -> int:
    if not os.path.exists(parquet_path):
        return 0
    try:
        return len(pd.read_parquet(parquet_path))
    except Exception:
        return 0


def append_done_real_post_flip(buffer_dir: str, article_id: str,
                               metrics_csv_path: str,
                               confidence: float,
                               current_round: int) -> int:
    # §1.3 — when an ID flips ACTIVE_FAKE→DONE_REAL the detector now calls it REAL.
    # The post-flip metrics are exactly the "REAL adversarial" signal we want the
    # retrained surrogate to see, so we move the ID from fake_accumulated to a
    # dedicated done_real_post_flip parquet (label=1).
    _ensure_dir(buffer_dir)
    if not os.path.exists(metrics_csv_path):
        return _safe_count(os.path.join(buffer_dir, "done_real_post_flip.parquet"))
    try:
        dfm = pd.read_csv(metrics_csv_path, encoding="utf-8")
    except Exception:
        return _safe_count(os.path.join(buffer_dir, "done_real_post_flip.parquet"))
    dfm["ID"] = dfm["ID"].astype(str)
    row = dfm[dfm["ID"] == str(article_id)]
    if row.empty:
        return _safe_count(os.path.join(buffer_dir, "done_real_post_flip.parquet"))

    feat_present = [c for c in FEATURE_COLS if c in row.columns]
    rec = row[["ID"] + feat_present].iloc[0:1].copy()
    rec["pred_label"] = "REAL"
    rec["confidence"] = float(confidence) if confidence is not None and not pd.isna(confidence) else np.nan
    rec["label"] = 1
    rec["source_round"] = int(current_round)

    out = os.path.join(buffer_dir, "done_real_post_flip.parquet")
    if os.path.exists(out):
        try:
            prev = pd.read_parquet(out)
            merged = pd.concat([prev, rec], ignore_index=True)
            merged = merged.sort_values("source_round").drop_duplicates(
                subset=["ID"], keep="last"
            )
            merged.to_parquet(out, index=False)
            n = len(merged)
        except Exception:
            rec.to_parquet(out, index=False)
            n = len(rec)
    else:
        rec.to_parquet(out, index=False)
        n = len(rec)

    fake_path = os.path.join(buffer_dir, "fake_accumulated.parquet")
    if os.path.exists(fake_path):
        try:
            fake_df = pd.read_parquet(fake_path)
            fake_df["ID"] = fake_df["ID"].astype(str)
            before = len(fake_df)
            fake_df = fake_df[fake_df["ID"] != str(article_id)]
            if len(fake_df) < before:
                fake_df.to_parquet(fake_path, index=False)
        except Exception:
            pass

    return n


# ─────────────────────────────────────────────
# Balanced training set
# ─────────────────────────────────────────────
def build_balanced_training_set(buffer_dir: str, ratio: float = 2.0,
                                random_state: int = 42) -> pd.DataFrame:
    real_path = os.path.join(buffer_dir, "real_baseline.parquet")
    fake_path = os.path.join(buffer_dir, "fake_accumulated.parquet")
    done_path = os.path.join(buffer_dir, "done_real_post_flip.parquet")
    if not os.path.exists(real_path):
        return pd.DataFrame()
    real = pd.read_parquet(real_path)
    if os.path.exists(done_path):
        try:
            done = pd.read_parquet(done_path)
            if not done.empty:
                real = pd.concat([real, done], ignore_index=True)
        except Exception as e:
            print(f"[BUFFER][WARN] could not load done_real_post_flip: {e}")
    if not os.path.exists(fake_path):
        return real
    fake = pd.read_parquet(fake_path)
    if len(real) == 0:
        return fake
    if len(fake) > ratio * len(real):
        n_keep = int(ratio * len(real))
        n_recent = int(0.6 * n_keep)
        n_random = max(0, n_keep - n_recent)
        fake_sorted = fake.sort_values("last_seen_round", ascending=False)
        recent = fake_sorted.head(n_recent)
        rest = fake_sorted.iloc[n_recent:]
        if n_random > 0 and len(rest) > 0:
            rest = rest.sample(n=min(n_random, len(rest)), random_state=random_state)
        else:
            rest = rest.iloc[0:0]
        fake = pd.concat([recent, rest], ignore_index=True)
    return pd.concat([real, fake], ignore_index=True)


# ─────────────────────────────────────────────
# Retrain trigger
# ─────────────────────────────────────────────
def _compute_beta_sign_changes(old_coeffs: pd.DataFrame,
                               new_coeffs: pd.DataFrame) -> List[Tuple[str, float, float]]:
    if old_coeffs.empty or new_coeffs.empty:
        return []
    # Try common column names
    feat_col = "feature" if "feature" in old_coeffs.columns else old_coeffs.columns[0]
    beta_col = None
    for cand in ("beta", "coefficient", "coef", "weight"):
        if cand in old_coeffs.columns and cand in new_coeffs.columns:
            beta_col = cand
            break
    if beta_col is None:
        return []
    merged = old_coeffs[[feat_col, beta_col]].merge(
        new_coeffs[[feat_col, beta_col]],
        on=feat_col,
        suffixes=("_old", "_new"),
    )
    bo = f"{beta_col}_old"
    bn = f"{beta_col}_new"
    flipped = merged[
        (merged[bo] * merged[bn] < 0)
        & (merged[bo].abs() > 1e-8)
        & (merged[bn].abs() > 1e-8)
    ]
    return [(str(r[feat_col]), float(r[bo]), float(r[bn])) for _, r in flipped.iterrows()]


def _stale_trajectory_for_flipped(output_root: str, detector: str,
                                  current_round: int, flipped_features: List[str]) -> None:
    """Mark feature trajectories as stale where beta sign flipped.

    Looks for `Output/trajectory/<det>/trajectory.json`. No-op if missing.
    """
    if not flipped_features:
        return
    traj_path = os.path.join(output_root, "trajectory", detector, "trajectory.json")
    if not os.path.exists(traj_path):
        return
    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            traj = json.load(f)
    except Exception as e:
        print(f"[RETRAIN][WARN] could not read trajectory.json: {e}")
        return
    by_id = traj.get("by_id", {}) if isinstance(traj, dict) else {}
    n_marked = 0
    for _id, id_rec in by_id.items():
        feats = id_rec.get("features", {}) if isinstance(id_rec, dict) else {}
        for feat in flipped_features:
            if feat in feats and isinstance(feats[feat], dict):
                feats[feat]["stale_before_round"] = int(current_round)
                feats[feat]["beta_flipped_at_round"] = int(current_round)
                n_marked += 1
    try:
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(traj, f, ensure_ascii=False, indent=2)
        print(f"[RETRAIN] marked {n_marked} feature-trajectory entries as stale")
    except Exception as e:
        print(f"[RETRAIN][WARN] could not write trajectory.json: {e}")


_ARCHIVE_FILES = (
    "meta_model_pipe.pkl",
    "meta_model_coefficients.csv",
    "feature_meta.json",
    "shap_attributions_per_id.csv",
    "shap_values.npy",
    "X_all_std.npy",
    "beta_stability.csv",
    "selected_features.csv",
)


def do_retrain_and_reload(output_root: str, detector: str,
                          current_round: int) -> dict:
    """Archive current frozen model, retrain SGL on cumulative buffer, recompute SHAP."""
    buffer_dir = _buffer_dir(output_root, detector)
    meta_frozen = os.path.join(output_root, "meta_frozen", detector)
    archive = os.path.join(meta_frozen, "archive", f"round_{current_round:03d}")

    # 1) Archive current frozen artifacts
    _ensure_dir(archive)
    for fname in _ARCHIVE_FILES:
        src = os.path.join(meta_frozen, fname)
        if os.path.exists(src):
            try:
                shutil.copy2(src, os.path.join(archive, fname))
            except Exception as e:
                print(f"[RETRAIN][WARN] archive {fname} failed: {e}")

    # 2) Snapshot old coefficients
    old_coeffs_path = os.path.join(meta_frozen, "meta_model_coefficients.csv")
    old_coeffs = (
        pd.read_csv(old_coeffs_path, encoding="utf-8") if os.path.exists(old_coeffs_path) else pd.DataFrame()
    )

    # 3) Build training set
    n_fake_pre_balance = _safe_count(os.path.join(buffer_dir, "fake_accumulated.parquet"))
    if n_fake_pre_balance < RETRAIN_MIN_FAKE:
        print(f"[RETRAIN][SKIP] n_fake_pre_balance={n_fake_pre_balance} < {RETRAIN_MIN_FAKE} — buffer too small")
        return {
            "status": "SKIPPED_INSUFFICIENT_FAKE",
            "n_fake_pre_balance": n_fake_pre_balance,
        }

    df_train = build_balanced_training_set(buffer_dir)
    n_real = int((df_train.get("label", pd.Series(dtype=int)) == 1).sum()) if not df_train.empty else 0
    n_fake = int((df_train.get("label", pd.Series(dtype=int)) == 0).sum()) if not df_train.empty else 0
    n_total = n_real + n_fake
    if n_total < MIN_TOTAL_SAMPLES:
        print(
            f"[RETRAIN][SKIP] n_total_post_balance={n_total} < {MIN_TOTAL_SAMPLES} "
            f"(n_real={n_real}, n_fake_pre={n_fake_pre_balance}, n_fake_post={n_fake})"
        )
        return {
            "status": "SKIPPED_INSUFFICIENT_BALANCED",
            "n_real": n_real,
            "n_fake_pre_balance": n_fake_pre_balance,
            "n_fake_post_balance": n_fake,
            "n_total": n_total,
        }

    # 4) Materialize synthetic round dir for train_meta_model_for_detector
    _tmp_round_dir = os.path.join(buffer_dir, "_retrain_input", f"round_{current_round:03d}")
    _tmp_gen_det = os.path.join(_tmp_round_dir, "generation", detector)
    _tmp_det_dir = os.path.join(_tmp_round_dir, "detectors", detector)
    _ensure_dir(_tmp_gen_det)
    _ensure_dir(_tmp_det_dir)

    feat_present = [c for c in FEATURE_COLS if c in df_train.columns]
    df_train_metrics = df_train[["ID"] + feat_present].copy()
    df_train_metrics.to_csv(
        os.path.join(_tmp_gen_det, "agent_metrics.csv"), index=False
    )
    df_train_det = df_train[["ID", "confidence"]].copy()
    df_train_det["pred_label"] = df_train["label"].map({0: "FAKE", 1: "REAL"})
    df_train_det.to_csv(
        os.path.join(_tmp_det_dir, f"detector_{detector}.csv"), index=False
    )

    # 5) Retrain in-place into meta_frozen
    _skipped_marker = os.path.join(meta_frozen, "meta_model_skipped.json")
    try:
        if os.path.exists(_skipped_marker):
            os.remove(_skipped_marker)
    except Exception:
        pass
    try:
        train_meta_model_for_detector(
            detector_name=detector,
            round_dir=_tmp_round_dir,
            target="label",
            output_dir=meta_frozen,
        )
    except Exception as e:
        print(f"[RETRAIN][ERROR] training failed: {e}")
        return {"status": "FAILED_TRAINING", "error": str(e),
                "n_real": n_real, "n_fake": n_fake}
    if os.path.exists(_skipped_marker):
        print(f"[RETRAIN][SKIP] train_meta_model_for_detector emitted "
              f"meta_model_skipped.json — old model preserved")
        return {"status": "SKIPPED_BY_TRAINER", "n_real": n_real, "n_fake": n_fake}

    # 6) Recompute SHAP against new pipe
    try:
        compute_shap_with_frozen_baseline(
            baseline_pipe_path=os.path.join(meta_frozen, "meta_model_pipe.pkl"),
            round_dir=_tmp_round_dir,
            detector_name=detector,
            out_dir=meta_frozen,
            target="label",
        )
    except Exception as e:
        print(f"[RETRAIN][WARN] SHAP recompute failed: {e}")

    # 7) Compute beta-sign flips
    new_coeffs = pd.read_csv(old_coeffs_path, encoding="utf-8") if os.path.exists(old_coeffs_path) else pd.DataFrame()
    flipped = _compute_beta_sign_changes(old_coeffs, new_coeffs)
    for feat, old_b, new_b in flipped:
        print(f"[RETRAIN][BETA-FLIP] {feat}: {old_b:.4f} → {new_b:.4f}")

    # 8) Update feature_meta.json retrain_history
    fmeta_path = os.path.join(meta_frozen, "feature_meta.json")
    fmeta = {}
    if os.path.exists(fmeta_path):
        try:
            with open(fmeta_path, "r", encoding="utf-8") as f:
                fmeta = json.load(f)
        except Exception:
            fmeta = {}
    fmeta.setdefault("retrain_history", []).append({
        "round": int(current_round),
        "timestamp": datetime.utcnow().isoformat(),
        "n_real": n_real,
        "n_fake": n_fake,
        "beta_sign_changes": [
            {"feature": f, "old": o, "new": n} for f, o, n in flipped
        ],
    })
    try:
        with open(fmeta_path, "w", encoding="utf-8") as f:
            json.dump(fmeta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[RETRAIN][WARN] could not write feature_meta.json: {e}")

    # 9) Mark trajectories stale for flipped features
    _stale_trajectory_for_flipped(
        output_root, detector, current_round, [f for f, _, _ in flipped]
    )

    # 10) Append retrain_log.csv
    log_path = os.path.join(buffer_dir, "retrain_log.csv")
    log_row = pd.DataFrame([{
        "round": int(current_round),
        "timestamp": datetime.utcnow().isoformat(),
        "n_real": n_real,
        "n_fake": n_fake,
        "n_beta_flipped": len(flipped),
        "flipped_features": ";".join(f for f, _, _ in flipped),
    }])
    if os.path.exists(log_path):
        try:
            existing = pd.read_csv(log_path, encoding="utf-8")
            log_row = pd.concat([existing, log_row], ignore_index=True)
        except Exception:
            pass
    log_row.to_csv(log_path, index=False)

    return {
        "status": "OK",
        "n_real": n_real,
        "n_fake": n_fake,
        "beta_sign_changes": flipped,
    }
