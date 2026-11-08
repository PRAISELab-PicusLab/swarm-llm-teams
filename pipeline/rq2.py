"""
RQ2 — Does the manipulated text remain semantically faithful to the original?

Metrics computed per ID (original vs generated at round it became REAL):
  - Semantic distance: 1 − cosine_similarity(emb_original, emb_generated)
  - TTR (Type-Token Ratio): unique_words / total_words
  - Word count
  - Sentence count

Statistical tests:
  1) Wilcoxon signed-rank (paired, per detector):
     For TTR, word_count, sentence_count:
       H0: no systematic difference between original and generated
       H1: the manipulation altered the property
     These are paired tests (same ID, two conditions).

  2) Spearman correlation (per detector + aggregated):
     "Does semantic fidelity degrade as more iterative rounds are needed?"

Plots (per detector):
  RQ2A — Box plot of semantic distance grouped by became_real_round
  RQ2B — ECDF of semantic distance per became_real_round

Usage:
  python -m pipeline rq2 --output_root Output
  python pipeline/rq2.py --output_root Output
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def discover_detectors(output_root: str) -> list[str]:
    det_dir = os.path.join(output_root, "rounds", "round_000", "detectors")
    if not os.path.isdir(det_dir):
        return []
    detectors = []
    for d in sorted(os.listdir(det_dir)):
        csv_path = os.path.join(det_dir, d, f"detector_{d}.csv")
        if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 10:
            detectors.append(d)
    return detectors


def discover_rounds_for_detector(output_root: str, detector: str) -> list[int]:
    rounds = [0]
    for d in sorted(glob.glob(os.path.join(output_root, "rounds", "round_*"))):
        try:
            r = int(os.path.basename(d).split("_")[1])
        except Exception:
            continue
        if r == 0:
            continue
        det_csv = os.path.join(d, "detectors", detector, f"detector_{detector}.csv")
        if os.path.isfile(det_csv) and os.path.getsize(det_csv) > 10:
            rounds.append(r)
    return sorted(rounds)


def load_detector_csv(output_root: str, detector: str, r: int) -> pd.DataFrame:
    if r == 0:
        p = os.path.join(output_root, "rounds", "round_000", "detectors", detector, f"detector_{detector}.csv")
    else:
        p = os.path.join(output_root, "rounds", f"round_{r:03d}", "detectors", detector, f"detector_{detector}.csv")
    df = pd.read_csv(p, encoding="utf-8")
    df["ID"] = df["ID"].astype(str)
    return df


def load_generated_texts(output_root: str, detector: str, r: int) -> dict[str, str]:
    """Load generated texts for a round/detector. Returns {ID: text}."""
    if r == 0:
        p = os.path.join(output_root, "rounds", "round_000", "generation", "Generated_Fake_News.csv")
    else:
        p = os.path.join(output_root, "rounds", f"round_{r:03d}", "generation", detector, "Generated_Fake_News.csv")
    if not os.path.isfile(p):
        return {}
    df = pd.read_csv(p, encoding="utf-8")
    df["ID"] = df["ID"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    return dict(zip(df["ID"], df["text"]))


def load_cohort(output_root: str) -> dict[str, str]:
    p = os.path.join(output_root, "cohort", "cohort_original.csv")
    df = pd.read_csv(p, encoding="utf-8")
    df["ID"] = df["ID"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    return dict(zip(df["ID"], df["text"]))


def load_ledger(output_root: str, detector: str) -> pd.DataFrame | None:
    p = os.path.join(output_root, "tracking", detector, "baseline_fakes_ledger.csv")
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p, encoding="utf-8")
    df["ID"] = df["ID"].astype(str)
    return df


def get_became_real_map(output_root: str, detector: str) -> dict[str, int]:
    """Returns {ID: round_when_became_REAL}. IDs REAL at round 0 -> 0."""
    result: dict[str, int] = {}
    det0 = load_detector_csv(output_root, detector, 0)
    for _, row in det0.iterrows():
        if str(row["pred_label"]).upper() == "REAL":
            result[str(row["ID"])] = 0
    ledger = load_ledger(output_root, detector)
    if ledger is not None and "became_real_round" in ledger.columns:
        for _, row in ledger.iterrows():
            br = row.get("became_real_round")
            if pd.notna(br):
                result[str(row["ID"])] = int(float(br))
    return result


def ecdf(arr: np.ndarray):
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


# ═══════════════════════════════════════════
# Text property functions
# ═══════════════════════════════════════════
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercased tokenization."""
    return text.lower().split()


def compute_ttr(text: str) -> float:
    """Type-Token Ratio: unique words / total words."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_word_count(text: str) -> int:
    return len(_tokenize(text))


def compute_sentence_count(text: str) -> int:
    if not text.strip():
        return 0
    sents = _SENT_SPLIT_RE.split(text.strip())
    return len([s for s in sents if len(s.strip()) >= 3])


# ═══════════════════════════════════════════
# Compute distances and text properties
# ═══════════════════════════════════════════
def compute_real_distances_and_properties(
    output_root: str,
    detector: str,
    orig_map: dict[str, str],
    became_real_map: dict[str, int],
    model,
) -> pd.DataFrame:
    """
    For each ID that became REAL, compute:
      - Semantic distance (embedding-based)
      - TTR, word_count, sentence_count (on original and generated)

    Returns DataFrame with all metrics.
    """
    if not became_real_map:
        return pd.DataFrame()

    ids_by_round: dict[int, list[str]] = {}
    for _id, r in became_real_map.items():
        ids_by_round.setdefault(r, []).append(_id)

    gen_r0 = load_generated_texts(output_root, detector, 0)
    rows = []

    for r, ids in sorted(ids_by_round.items()):
        gen_r = gen_r0 if r == 0 else load_generated_texts(output_root, detector, r)

        valid_ids = [i for i in ids
                     if i in orig_map and orig_map[i].strip()
                     and i in gen_r and gen_r[i].strip()
                     and i in gen_r0 and gen_r0[i].strip()]
        if not valid_ids:
            continue

        # Semantic distance (embedding-based)
        orig_texts = [orig_map[i][:12000] for i in valid_ids]
        orig_emb = model.encode(orig_texts, normalize_embeddings=True, show_progress_bar=False)

        real_texts = [gen_r[i][:12000] for i in valid_ids]
        real_emb = model.encode(real_texts, normalize_embeddings=True, show_progress_bar=False)
        dist_real = (1.0 - np.sum(orig_emb * real_emb, axis=1)).astype(float)

        if r == 0:
            dist_r0 = dist_real
        else:
            r0_texts = [gen_r0[i][:12000] for i in valid_ids]
            r0_emb = model.encode(r0_texts, normalize_embeddings=True, show_progress_bar=False)
            dist_r0 = (1.0 - np.sum(orig_emb * r0_emb, axis=1)).astype(float)

        # Text properties
        for i, _id in enumerate(valid_ids):
            orig_text = orig_map[_id]
            gen_text = gen_r[_id]

            rows.append({
                "ID": _id,
                "became_real_round": r,
                "distance_at_real": float(dist_real[i]),
                "distance_at_r0": float(dist_r0[i]),
                "ttr_original": compute_ttr(orig_text),
                "ttr_generated": compute_ttr(gen_text),
                "wc_original": compute_word_count(orig_text),
                "wc_generated": compute_word_count(gen_text),
                "sc_original": compute_sentence_count(orig_text),
                "sc_generated": compute_sentence_count(gen_text),
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════
# Statistical tests
# ═══════════════════════════════════════════
def wilcoxon_paired_test(
    original: np.ndarray,
    generated: np.ndarray,
    metric_name: str,
    alpha: float = 0.05,
) -> dict:
    """
    Wilcoxon signed-rank test (paired, two-sided).
    H0: no systematic difference between original and generated
    H1: the manipulation altered the property
    """
    n = len(original)
    if n < 10:
        return {
            "metric": metric_name,
            "n": n,
            "median_original": float(np.median(original)) if n > 0 else None,
            "median_generated": float(np.median(generated)) if n > 0 else None,
            "median_diff": None,
            "statistic": None,
            "p_value": None,
            "significant": False,
            "effect_size_r": None,
            "interpretation": f"Not applicable (n={n}, need at least 10 paired observations)",
        }

    diff = generated - original
    median_diff = float(np.median(diff))

    # Remove zero differences (Wilcoxon ignores them)
    nonzero = diff[diff != 0]
    if len(nonzero) < 5:
        return {
            "metric": metric_name,
            "n": n,
            "median_original": float(np.median(original)),
            "median_generated": float(np.median(generated)),
            "median_diff": median_diff,
            "statistic": None,
            "p_value": None,
            "significant": False,
            "effect_size_r": None,
            "interpretation": f"Not applicable — too few non-zero differences ({len(nonzero)})",
        }

    stat, p_val = stats.wilcoxon(original, generated, alternative="two-sided")
    stat = float(stat)
    p_val = float(p_val)

    # Effect size: r = Z / sqrt(N)
    z_approx = float(stats.norm.ppf(1.0 - p_val / 2.0)) if p_val < 1.0 else 0.0
    r_effect = z_approx / np.sqrt(n)

    significant = p_val < alpha

    if median_diff > 0:
        direction = "generated HIGHER"
    elif median_diff < 0:
        direction = "generated LOWER"
    else:
        direction = "no clear direction"

    if significant:
        interp = (
            f"SIGNIFICANT DIFFERENCE — {direction} — "
            f"median_orig={np.median(original):.4f}, "
            f"median_gen={np.median(generated):.4f}, "
            f"median_diff={median_diff:+.4f}, "
            f"W={stat:.1f}, p={p_val:.6f}, r={r_effect:.3f}"
        )
    else:
        interp = (
            f"NO SIGNIFICANT DIFFERENCE — "
            f"median_orig={np.median(original):.4f}, "
            f"median_gen={np.median(generated):.4f}, "
            f"median_diff={median_diff:+.4f}, "
            f"W={stat:.1f}, p={p_val:.6f}, r={r_effect:.3f}"
        )

    return {
        "metric": metric_name,
        "n": n,
        "median_original": round(float(np.median(original)), 6),
        "median_generated": round(float(np.median(generated)), 6),
        "median_diff": round(median_diff, 6),
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "significant": significant,
        "effect_size_r": round(r_effect, 4),
        "interpretation": interp,
    }


def spearman_test(rounds: np.ndarray, distances: np.ndarray) -> dict:
    """
    Spearman rank correlation: became_real_round vs distance_at_real.
    H0: rho = 0  (no monotonic association between round and distance)
    H1 two-sided: rho != 0
    H1 one-sided: rho > 0  (more rounds -> higher distance, i.e. degradation)
    """
    n = len(rounds)
    n_unique_rounds = len(np.unique(rounds))

    if n < 3:
        return {
            "n": n, "n_unique_rounds": n_unique_rounds,
            "rho": None, "p_value_two_sided": None, "p_value_one_sided": None,
            "significant_two_sided": False, "significant_one_sided": False,
            "interpretation": f"Not applicable (n={n}, need at least 3 observations)",
        }
    if n_unique_rounds < 2:
        return {
            "n": n, "n_unique_rounds": n_unique_rounds,
            "rho": None, "p_value_two_sided": None, "p_value_one_sided": None,
            "significant_two_sided": False, "significant_one_sided": False,
            "interpretation": f"Not applicable (only 1 distinct round value, n={n})",
        }

    rho, p_two = stats.spearmanr(rounds, distances)
    rho = float(rho)
    p_two = float(p_two)

    p_one = p_two / 2 if rho > 0 else 1.0 - p_two / 2

    sig_two = p_two < 0.05
    sig_one = p_one < 0.05

    if sig_two:
        direction = "MORE rounds -> HIGHER distance" if rho > 0 else "MORE rounds -> LOWER distance"
        interp = (
            f"SIGNIFICANT — {direction} — "
            f"rho={rho:.4f}, n={n}, "
            f"p-value(two-sided)={p_two:.6f}, p-value(one-sided)={p_one:.6f}"
        )
    else:
        interp = (
            f"NO SIGNIFICANT ASSOCIATION — "
            f"rho={rho:.4f}, n={n}, "
            f"p-value(two-sided)={p_two:.6f}, p-value(one-sided)={p_one:.6f}"
        )

    return {
        "n": n,
        "n_unique_rounds": n_unique_rounds,
        "rho": round(rho, 6),
        "p_value_two_sided": round(p_two, 6),
        "p_value_one_sided": round(p_one, 6),
        "significant_two_sided": sig_two,
        "significant_one_sided": sig_one,
        "interpretation": interp,
    }


# ═══════════════════════════════════════════
# Run tests per detector
# ═══════════════════════════════════════════
def run_statistical_tests(df: pd.DataFrame, detector: str, plots_dir: str) -> None:
    """Run Wilcoxon paired tests + Spearman per-detector. Save CSV and print summary."""
    test_rows = []

    # --- Wilcoxon signed-rank: TTR ---
    w_ttr = wilcoxon_paired_test(
        df["ttr_original"].to_numpy(),
        df["ttr_generated"].to_numpy(),
        metric_name="TTR (Type-Token Ratio)",
    )
    test_rows.append({
        "detector": detector,
        "test": "Wilcoxon_paired_TTR",
        **{k: v for k, v in w_ttr.items() if k != "metric"},
    })

    # --- Wilcoxon signed-rank: Word count ---
    w_wc = wilcoxon_paired_test(
        df["wc_original"].to_numpy().astype(float),
        df["wc_generated"].to_numpy().astype(float),
        metric_name="Word Count",
    )
    test_rows.append({
        "detector": detector,
        "test": "Wilcoxon_paired_WordCount",
        **{k: v for k, v in w_wc.items() if k != "metric"},
    })

    # --- Wilcoxon signed-rank: Sentence count ---
    w_sc = wilcoxon_paired_test(
        df["sc_original"].to_numpy().astype(float),
        df["sc_generated"].to_numpy().astype(float),
        metric_name="Sentence Count",
    )
    test_rows.append({
        "detector": detector,
        "test": "Wilcoxon_paired_SentenceCount",
        **{k: v for k, v in w_sc.items() if k != "metric"},
    })

    # --- Spearman per-detector ---
    sp = spearman_test(
        df["became_real_round"].to_numpy(),
        df["distance_at_real"].to_numpy(),
    )
    test_rows.append({
        "detector": detector,
        "test": "Spearman_round_vs_distance",
        "n": sp["n"],
        "median_original": None,
        "median_generated": None,
        "median_diff": None,
        "statistic": sp["rho"],
        "p_value": sp["p_value_two_sided"],
        "significant": sp["significant_two_sided"],
        "effect_size_r": None,
        "interpretation": sp["interpretation"],
    })

    df_tests = pd.DataFrame(test_rows)
    fname = f"RQ2_statistical_tests_{detector}.csv"
    df_tests.to_csv(os.path.join(plots_dir, fname), index=False)
    print(f"  [STATS] Saved {fname}")

    # Print summary
    print(f"\n  ┌─── Statistical tests — {detector} ───")
    for _, row in df_tests.iterrows():
        print(f"  │ {row['test']}:")
        print(f"  │   {row['interpretation']}")
    print(f"  └{'─' * 60}")


# ═══════════════════════════════════════════
# Spearman aggregated across all detectors
# ═══════════════════════════════════════════
def run_aggregated_spearman(all_distance_dfs: dict[str, pd.DataFrame], plots_dir: str) -> None:
    """
    Pool distance data from all detectors with >1 distinct round
    and run a single Spearman test on the combined sample.
    Also includes per-detector results for reference.
    """
    parts = []
    for det, df in all_distance_dfs.items():
        if df["became_real_round"].nunique() >= 2:
            tmp = df[["ID", "became_real_round", "distance_at_real"]].copy()
            tmp["detector"] = det
            parts.append(tmp)

    rows = []

    if parts:
        combined = pd.concat(parts, ignore_index=True)
        sp = spearman_test(
            combined["became_real_round"].to_numpy(),
            combined["distance_at_real"].to_numpy(),
        )
        rows.append({
            "scope": "aggregated_all_detectors",
            "detectors_included": ", ".join(sorted(combined["detector"].unique())),
            "n": sp["n"],
            "n_unique_rounds": sp["n_unique_rounds"],
            "rho": sp["rho"],
            "p_value_two_sided": sp["p_value_two_sided"],
            "p_value_one_sided": sp["p_value_one_sided"],
            "significant_two_sided": sp["significant_two_sided"],
            "significant_one_sided": sp["significant_one_sided"],
            "interpretation": sp["interpretation"],
        })
    else:
        print("\n  [SPEARMAN AGG] No detectors with >1 round — skipping aggregated test")

    for det, df in all_distance_dfs.items():
        sp_det = spearman_test(
            df["became_real_round"].to_numpy(),
            df["distance_at_real"].to_numpy(),
        )
        rows.append({
            "scope": f"per_detector_{det}",
            "detectors_included": det,
            "n": sp_det["n"],
            "n_unique_rounds": sp_det["n_unique_rounds"],
            "rho": sp_det["rho"],
            "p_value_two_sided": sp_det["p_value_two_sided"],
            "p_value_one_sided": sp_det["p_value_one_sided"],
            "significant_two_sided": sp_det["significant_two_sided"],
            "significant_one_sided": sp_det["significant_one_sided"],
            "interpretation": sp_det["interpretation"],
        })

    if rows:
        df_sp = pd.DataFrame(rows)
        fname = "RQ2_spearman_round_vs_distance.csv"
        df_sp.to_csv(os.path.join(plots_dir, fname), index=False)
        print(f"\n  [SPEARMAN AGG] Saved {fname}")

        print(f"\n  ┌─── Spearman: round vs semantic distance ───")
        for _, row in df_sp.iterrows():
            print(f"  │ [{row['scope']}]:")
            print(f"  │   {row['interpretation']}")
        print(f"  └{'─' * 60}")


# ═══════════════════════════════════════════
# RQ2-A  Box plot
# ═══════════════════════════════════════════
def plot_rq2a(df: pd.DataFrame, detector: str, plots_dir: str) -> None:
    rounds_sorted = sorted(df["became_real_round"].unique())
    if not rounds_sorted:
        print(f"  [RQ2A] {detector}: no data -> skip")
        return

    data = []
    labels = []
    info_lines = []
    for r in rounds_sorted:
        sub = df[df["became_real_round"] == r]
        data.append(sub["distance_at_real"].to_numpy())
        labels.append(f"R{r}")
        info_lines.append(f"R{r}: n={len(sub)}")

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(rounds_sorted), 2)))
    fig, ax = plt.subplots(figsize=(max(6, len(rounds_sorted) * 2), 6))
    bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=True)

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Semantic distance (1 − cos_sim)", fontsize=12)
    ax.set_xlabel("Round became REAL", fontsize=12)

    info_text = "IDs became REAL:\n" + "\n".join(info_lines)
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                                  edgecolor="black", alpha=0.9))

    plt.tight_layout()
    fname = f"RQ2A_boxplot_distance_{detector}.png"
    plt.savefig(os.path.join(plots_dir, fname), dpi=220)
    plt.close(fig)
    print(f"  [RQ2A] Saved {fname}")


# ═══════════════════════════════════════════
# RQ2-B  ECDF
# ═══════════════════════════════════════════
def plot_rq2b(df: pd.DataFrame, detector: str, plots_dir: str) -> None:
    rounds_sorted = sorted(df["became_real_round"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rounds_sorted), 2)))

    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False

    for i, r in enumerate(rounds_sorted):
        sub = df[df["became_real_round"] == r]
        if sub.empty:
            continue
        has_data = True
        x, y = ecdf(sub["distance_at_real"].to_numpy())
        ax.plot(x, y, color=colors[i], linewidth=2, label=f"Round {r} (n={len(sub)})")

    if not has_data:
        ax.text(0.5, 0.5, "No IDs became REAL", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)

    ax.set_xlabel("Semantic distance (1 − cos_sim)", fontsize=12)
    ax.set_ylabel("ECDF", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    fname = f"RQ2B_ecdf_distance_{detector}.png"
    plt.savefig(os.path.join(plots_dir, fname), dpi=220)
    plt.close(fig)
    print(f"  [RQ2B] Saved {fname}")


# ═══════════════════════════════════════════
# Round-0 shared distances
# The round-0 generation is UNIQUE and shared across all detectors, so the
# distance (generated fake vs original REAL) is identical for every detector.
# It is therefore computed ONCE and plotted separately (not per detector).
# ═══════════════════════════════════════════
def compute_round0_distances(output_root: str, orig_map: dict[str, str], model) -> pd.DataFrame:
    gen0 = load_generated_texts(output_root, "", 0)  # detector ignored for round 0
    ids = [i for i in gen0
           if i in orig_map and orig_map[i].strip() and gen0[i].strip()]
    if not ids:
        return pd.DataFrame()
    orig_emb = model.encode([orig_map[i][:12000] for i in ids],
                            normalize_embeddings=True, show_progress_bar=False)
    gen_emb = model.encode([gen0[i][:12000] for i in ids],
                           normalize_embeddings=True, show_progress_bar=False)
    dist = (1.0 - np.sum(orig_emb * gen_emb, axis=1)).astype(float)
    return pd.DataFrame({"ID": ids, "distance_round0": dist})


def plot_round0(df: pd.DataFrame, plots_dir: str) -> None:
    d = df["distance_round0"].to_numpy()
    # boxplot
    fig, ax = plt.subplots(figsize=(5, 6))
    bp = ax.boxplot([d], vert=True, patch_artist=True, showfliers=True)
    bp["boxes"][0].set_facecolor("#66c2a5"); bp["boxes"][0].set_alpha(0.7)
    ax.set_xticklabels([f"gen0 (n={len(d)})"])
    ax.set_ylabel("Semantic distance (1 - cos_sim)")
    ax.set_title("Round 0 - generated fake vs original (shared, all detectors)")
    ax.text(0.98, 0.98, f"median={np.median(d):.4f}\nmean={np.mean(d):.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "RQ2_round0_boxplot.png"), dpi=220); plt.close(fig)
    # ecdf
    fig, ax = plt.subplots(figsize=(9, 6))
    x, y = ecdf(d); ax.plot(x, y, color="#3288bd", linewidth=2)
    ax.set_xlabel("Semantic distance (1 - cos_sim)"); ax.set_ylabel("ECDF")
    ax.set_title("Round 0 - ECDF generated fake vs original (shared)")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "RQ2_round0_ecdf.png"), dpi=220); plt.close(fig)
    # histogram
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(d, bins=40, color="#fc8d62", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Semantic distance (1 - cos_sim)"); ax.set_ylabel("count")
    ax.set_title("Round 0 - distribution generated fake vs original (shared)")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "RQ2_round0_histogram.png"), dpi=220); plt.close(fig)
    print("  [RQ2 round0] Saved RQ2_round0_boxplot / ecdf / histogram .png")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
def run(output_root: str = "Output", embedding_model: str = "all-MiniLM-L6-v2") -> int:
    # Avoid UnicodeEncodeError when stdout is a non-UTF8 console/pipe (e.g. Windows cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    plots_dir = os.path.join(output_root, "analysis_plots", "rq2")
    ensure_dir(plots_dir)

    detectors = discover_detectors(output_root)
    if not detectors:
        print("[RQ2] No detectors found.")
        return 1
    print(f"[RQ2] Detectors found: {detectors}")

    orig_map = load_cohort(output_root)
    print(f"[RQ2] Original texts loaded: {len(orig_map)}")

    print(f"[RQ2] Loading embedding model: {embedding_model}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedding_model)

    # ---- 1) ROUND-0 SHARED (generation is unique -> one plot for all detectors) ----
    print("\n[RQ2] ===== Round 0 (shared: generated fake vs original) =====")
    df0 = compute_round0_distances(output_root, orig_map, model)
    if not df0.empty:
        df0.to_csv(os.path.join(plots_dir, "RQ2_round0_generation_distances.csv"), index=False)
        print(f"  Saved RQ2_round0_generation_distances.csv ({len(df0)} rows) "
              f"median={df0['distance_round0'].median():.4f}")
        plot_round0(df0, plots_dir)
    else:
        print("  [RQ2 round0] no data")

    # ---- 2) PER LOOP-DETECTOR (rounds >= 1: adversarial edits) ----
    # Only detectors whose loop actually flipped IDs at round >= 1 get per-detector
    # plots. STOP detectors (no loop) are fully covered by the shared round-0 plot.
    all_distance_dfs: dict[str, pd.DataFrame] = {}
    for det in detectors:
        became_real_map = get_became_real_map(output_root, det)
        loop_map = {i: r for i, r in became_real_map.items() if r >= 1}
        if not loop_map:
            print(f"\n[RQ2] {det}: no loop flips (round>=1) -> skip per-detector (covered by round-0 shared)")
            continue
        print(f"\n[RQ2] ===== Detector (loop): {det} - {len(loop_map)} IDs flipped in loop =====")
        df = compute_real_distances_and_properties(output_root, det, orig_map, loop_map, model)
        if df.empty:
            print(f"  [RQ2] {det}: no distance data -> skip")
            continue
        df.to_csv(os.path.join(plots_dir, f"RQ2_loop_distances_{det}.csv"), index=False)
        print(f"  Saved RQ2_loop_distances_{det}.csv ({len(df)} rows)")
        all_distance_dfs[det] = df
        plot_rq2a(df, det, plots_dir)
        plot_rq2b(df, det, plots_dir)
        run_statistical_tests(df, det, plots_dir)

    if all_distance_dfs:
        run_aggregated_spearman(all_distance_dfs, plots_dir)

    print(f"\n[RQ2] All outputs saved in: {plots_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RQ2 — Semantic distance analysis.")
    p.add_argument("--output_root", type=str, default="Output")
    p.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(run(output_root=args.output_root, embedding_model=args.embedding_model))
