"""
RQ3 — Which structural components most influence classification?

Analysis:
  RQ3A — Feature assignments by the orchestrator across all detectors
         (heatmap + normalized stacked bars for cross-detector comparison)
  RQ3B — Decisive feature (the last feature manipulated before an ID
         becomes REAL) across all detectors
  RQ3C — Per-round feature evolution per detector (heatmap detail)

The key question: do all detectors share the same vulnerability signals,
or do they respond to different structural manipulations?

Usage:
  python -m pipeline rq3 --output_root Output
  python pipeline/rq3.py --output_root Output
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


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


def is_all_real(output_root: str, detector: str) -> bool:
    return os.path.exists(
        os.path.join(output_root, "rounds", "round_000", "orchestrator", detector, "STOP_ALL_REAL.txt")
    )


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
        orch_csv = os.path.join(d, "orchestrator", detector, "orch_decisions_with_reasons.csv")
        if (os.path.isfile(det_csv) and os.path.getsize(det_csv) > 10) or \
           (os.path.isfile(orch_csv) and os.path.getsize(orch_csv) > 10):
            rounds.append(r)
    return sorted(rounds)


def load_orch_decisions(output_root: str, detector: str, r: int) -> pd.DataFrame | None:
    p = os.path.join(output_root, "rounds", f"round_{r:03d}", "orchestrator", detector,
                     "orch_decisions_with_reasons.csv")
    if not os.path.isfile(p):
        p = os.path.join(output_root, "rounds", f"round_{r:03d}", "orch", detector,
                         "orch_decisions_with_reasons.csv")
    if not os.path.isfile(p) or os.path.getsize(p) < 10:
        return None
    df = pd.read_csv(p, encoding="utf-8")
    if "ID" not in df.columns or "chosen_feature" not in df.columns:
        return None
    df["ID"] = df["ID"].astype(str)
    df["chosen_feature"] = df["chosen_feature"].fillna("").astype(str)
    return df


def load_ledger(output_root: str, detector: str) -> pd.DataFrame | None:
    p = os.path.join(output_root, "tracking", detector, "baseline_fakes_ledger.csv")
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p, encoding="utf-8")
    df["ID"] = df["ID"].astype(str)
    return df


def load_detector_round0(output_root: str, detector: str) -> pd.DataFrame:
    p = os.path.join(output_root, "rounds", "round_000", "detectors", detector, f"detector_{detector}.csv")
    df = pd.read_csv(p, encoding="utf-8")
    df["ID"] = df["ID"].astype(str)
    return df


def json_list_to_list(x) -> list:
    if pd.isna(x) or x is None:
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        result = json.loads(s)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    return []


# ═══════════════════════════════════════════
# Data collection
# ═══════════════════════════════════════════
def collect_all_assignments(output_root: str, detectors: list[str]) -> pd.DataFrame:
    """Collect ALL orchestrator feature assignments across all detectors and rounds.
    Returns: [detector, round, feature, ID]"""
    rows = []
    for det in detectors:
        if is_all_real(output_root, det):
            continue
        rounds = discover_rounds_for_detector(output_root, det)
        for r in rounds:
            orch = load_orch_decisions(output_root, det, r)
            if orch is None or orch.empty:
                continue
            for _, row in orch.iterrows():
                feat = str(row["chosen_feature"]).strip()
                if feat:
                    rows.append({"detector": det, "round": r, "feature": feat, "ID": str(row["ID"])})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["detector", "round", "feature", "ID"])


def collect_techniques(output_root: str, detectors: list[str]) -> pd.DataFrame:
    """Aggregate NaM `technique_used` across all rounds and detectors.

    Joins with the ledger to attach the outcome (status, became_real_round).
    Returns: [detector, ID, round_idx, technique_used, status, became_real_round]
    """
    rows = []
    for det in detectors:
        if is_all_real(output_root, det):
            continue
        for round_dir in sorted(glob.glob(os.path.join(output_root, "rounds", "round_*"))):
            try:
                r = int(os.path.basename(round_dir).split("_")[1])
            except Exception:
                continue
            candidates = [
                os.path.join(round_dir, "generation", det, "NarrativeModifier.csv"),
                os.path.join(round_dir, "generation", "NarrativeModifier.csv"),
            ]
            nam_csv = next((p for p in candidates if os.path.isfile(p) and os.path.getsize(p) > 10), None)
            if nam_csv is None:
                continue
            try:
                df = pd.read_csv(nam_csv, encoding="utf-8")
            except Exception:
                continue
            if "technique_used" not in df.columns or "ID" not in df.columns:
                continue
            df = df[["ID", "technique_used"]].copy()
            df["ID"] = df["ID"].astype(str)
            df["technique_used"] = df["technique_used"].fillna("").astype(str)
            df["detector"] = det
            df["round_idx"] = r
            rows.append(df)
        # nothing else to do per detector
    if not rows:
        return pd.DataFrame(columns=["detector", "ID", "round_idx", "technique_used", "status", "became_real_round"])
    df_all = pd.concat(rows, ignore_index=True)
    # Join with ledger for outcome
    ledger_frames = []
    for det in detectors:
        led = load_ledger(output_root, det)
        if led is None:
            continue
        keep_cols = [c for c in ["ID", "status", "became_real_round"] if c in led.columns]
        if "ID" not in keep_cols:
            continue
        sub = led[keep_cols].copy()
        sub["detector"] = det
        ledger_frames.append(sub)
    if ledger_frames:
        led_all = pd.concat(ledger_frames, ignore_index=True)
        df_all = df_all.merge(led_all, on=["detector", "ID"], how="left")
    return df_all


def collect_decisive_features(output_root: str, detectors: list[str]) -> pd.DataFrame:
    """For each ID that became REAL (round > 0), find the decisive feature."""
    rows = []
    for det in detectors:
        if is_all_real(output_root, det):
            continue
        ledger = load_ledger(output_root, det)
        if ledger is None:
            continue
        became_real = ledger[ledger["became_real_round"].notna()].copy()
        if became_real.empty:
            continue
        for _, row in became_real.iterrows():
            _id = str(row["ID"])
            br_round = int(float(row["became_real_round"]))
            # Legacy: "used_features" column was removed from the ledger (Fix 6).
            # This returns an empty list when the column is absent, and the code
            # falls through to the orch_decisions fallback below.
            used = json_list_to_list(row.get("used_features", None))
            if used:
                decisive_feat = used[-1]
            else:
                decisive_feat = None
                if br_round > 0:
                    orch_prev = load_orch_decisions(output_root, det, br_round - 1)
                    if orch_prev is not None:
                        match = orch_prev[orch_prev["ID"] == _id]
                        if not match.empty:
                            decisive_feat = str(match.iloc[0]["chosen_feature"]).strip()
                if not decisive_feat:
                    decisive_feat = "unknown"
            rows.append({"detector": det, "ID": _id, "became_real_round": br_round, "feature": decisive_feat})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["detector", "ID", "became_real_round", "feature"])


# ═══════════════════════════════════════════
# RQ3A — Cross-detector heatmap (total assignments)
# ═══════════════════════════════════════════
def plot_rq3a_heatmap(assignments: pd.DataFrame, plots_dir: str) -> None:
    """Heatmap: rows = features, columns = detectors. Cells = total assignment count."""
    if assignments.empty:
        print("  [RQ3A] No assignment data -> skip heatmap")
        return

    pivot = assignments.groupby(["detector", "feature"]).size().reset_index(name="count")
    matrix = pivot.pivot_table(index="feature", columns="detector", values="count", fill_value=0)

    all_features = sorted(matrix.index)
    all_detectors = sorted(matrix.columns)
    matrix = matrix.loc[all_features, all_detectors]

    fig, ax = plt.subplots(figsize=(max(6, len(all_detectors) * 2.5), max(4, len(all_features) * 1.2)))

    data = matrix.values.astype(float)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto",
                   norm=mcolors.Normalize(vmin=0, vmax=data.max() if data.max() > 0 else 1))

    # Annotate each cell with count
    for i in range(len(all_features)):
        for j in range(len(all_detectors)):
            val = int(data[i, j])
            color = "white" if val > data.max() * 0.6 else "black"
            ax.text(j, i, str(val) if val > 0 else "—",
                    ha="center", va="center", fontsize=13, fontweight="bold", color=color)

    ax.set_xticks(range(len(all_detectors)))
    ax.set_xticklabels(all_detectors, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(all_features)))
    ax.set_yticklabels(all_features, fontsize=10)
    ax.set_xlabel("Detector", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Total assignments (all rounds)", fontsize=10)

    plt.tight_layout()
    fname = "RQ3A_heatmap_features_vs_detectors.png"
    plt.savefig(os.path.join(plots_dir, fname), dpi=220)
    plt.close(fig)
    print(f"  [RQ3A] Saved {fname}")


# ═══════════════════════════════════════════
# RQ3A — Normalized stacked bars (% per detector)
# ═══════════════════════════════════════════
def plot_rq3a_normalized_bars(assignments: pd.DataFrame, plots_dir: str) -> None:
    """Horizontal stacked bars: one bar per detector, showing % of each feature."""
    if assignments.empty:
        print("  [RQ3A] No assignment data -> skip normalized bars")
        return

    pivot = assignments.groupby(["detector", "feature"]).size().reset_index(name="count")

    # Compute percentages per detector
    totals = pivot.groupby("detector")["count"].sum().to_dict()
    pivot["pct"] = pivot.apply(lambda r: r["count"] / totals[r["detector"]] * 100, axis=1)

    all_detectors = sorted(pivot["detector"].unique())
    all_features = sorted(pivot["feature"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(all_features), 2)))

    fig, ax = plt.subplots(figsize=(12, max(3, len(all_detectors) * 1.5)))

    y = np.arange(len(all_detectors))
    bar_height = 0.55
    lefts = np.zeros(len(all_detectors))

    for j, feat in enumerate(all_features):
        vals = []
        counts = []
        for det in all_detectors:
            sub = pivot[(pivot["detector"] == det) & (pivot["feature"] == feat)]
            vals.append(float(sub["pct"].sum()) if not sub.empty else 0.0)
            counts.append(int(sub["count"].sum()) if not sub.empty else 0)

        bars = ax.barh(y, vals, bar_height, left=lefts, label=feat,
                       color=colors[j % len(colors)], edgecolor="black", linewidth=0.5)

        # Annotate inside bars with count and %
        for k, (bar, pct_val, cnt) in enumerate(zip(bars, vals, counts)):
            if pct_val > 8:  # only annotate if wide enough
                cx = lefts[k] + pct_val / 2
                ax.text(cx, y[k], f"{cnt}\n({pct_val:.0f}%)",
                        ha="center", va="center", fontsize=9, fontweight="bold")

        lefts += np.array(vals)

    # Add total count at the end of each bar
    for k, det in enumerate(all_detectors):
        ax.text(lefts[k] + 1.5, y[k], f"n={totals[det]}",
                ha="left", va="center", fontsize=10, fontstyle="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(all_detectors, fontsize=11, fontweight="bold")
    ax.set_xlabel("Feature assignment distribution (%)", fontsize=12)
    ax.set_xlim(0, 115)
    ax.legend(loc="upper right", fontsize=9, title="Feature", title_fontsize=10)

    plt.tight_layout()
    fname = "RQ3A_normalized_bars_features_per_detector.png"
    plt.savefig(os.path.join(plots_dir, fname), dpi=220)
    plt.close(fig)
    print(f"  [RQ3A] Saved {fname}")


# ═══════════════════════════════════════════
# RQ3B — Decisive feature heatmap
# ═══════════════════════════════════════════
def plot_rq3b_heatmap(decisive: pd.DataFrame, plots_dir: str) -> None:
    """Heatmap: rows = features, columns = detectors. Cells = decisive feature count."""
    if decisive.empty:
        print("  [RQ3B] No decisive feature data -> skip heatmap")
        return

    pivot = decisive.groupby(["detector", "feature"]).size().reset_index(name="count")
    matrix = pivot.pivot_table(index="feature", columns="detector", values="count", fill_value=0)

    all_features = sorted(matrix.index)
    all_detectors = sorted(matrix.columns)
    matrix = matrix.loc[all_features, all_detectors]

    fig, ax = plt.subplots(figsize=(max(6, len(all_detectors) * 2.5), max(4, len(all_features) * 1.2)))

    data = matrix.values.astype(float)
    im = ax.imshow(data, cmap="PuBuGn", aspect="auto",
                   norm=mcolors.Normalize(vmin=0, vmax=data.max() if data.max() > 0 else 1))

    for i in range(len(all_features)):
        for j in range(len(all_detectors)):
            val = int(data[i, j])
            color = "white" if val > data.max() * 0.6 else "black"
            ax.text(j, i, str(val) if val > 0 else "—",
                    ha="center", va="center", fontsize=13, fontweight="bold", color=color)

    ax.set_xticks(range(len(all_detectors)))
    ax.set_xticklabels(all_detectors, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(all_features)))
    ax.set_yticklabels(all_features, fontsize=10)
    ax.set_xlabel("Detector", fontsize=12)
    ax.set_ylabel("Decisive feature", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Count (IDs tipped to REAL)", fontsize=10)

    plt.tight_layout()
    fname = "RQ3B_heatmap_decisive_features.png"
    plt.savefig(os.path.join(plots_dir, fname), dpi=220)
    plt.close(fig)
    print(f"  [RQ3B] Saved {fname}")


# ═══════════════════════════════════════════
# RQ3B — Decisive feature normalized bars
# ═══════════════════════════════════════════
def plot_rq3b_normalized_bars(decisive: pd.DataFrame, plots_dir: str) -> None:
    """Horizontal stacked bars for decisive features, one per detector."""
    if decisive.empty:
        print("  [RQ3B] No decisive feature data -> skip normalized bars")
        return

    pivot = decisive.groupby(["detector", "feature"]).size().reset_index(name="count")
    totals = pivot.groupby("detector")["count"].sum().to_dict()
    pivot["pct"] = pivot.apply(lambda r: r["count"] / totals[r["detector"]] * 100, axis=1)

    all_detectors = sorted(pivot["detector"].unique())
    all_features = sorted(pivot["feature"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(all_features), 2)))

    fig, ax = plt.subplots(figsize=(12, max(3, len(all_detectors) * 1.5)))

    y = np.arange(len(all_detectors))
    bar_height = 0.55
    lefts = np.zeros(len(all_detectors))

    for j, feat in enumerate(all_features):
        vals = []
        counts = []
        for det in all_detectors:
            sub = pivot[(pivot["detector"] == det) & (pivot["feature"] == feat)]
            vals.append(float(sub["pct"].sum()) if not sub.empty else 0.0)
            counts.append(int(sub["count"].sum()) if not sub.empty else 0)

        bars = ax.barh(y, vals, bar_height, left=lefts, label=feat,
                       color=colors[j % len(colors)], edgecolor="black", linewidth=0.5)

        for k, (bar, pct_val, cnt) in enumerate(zip(bars, vals, counts)):
            if pct_val > 12:
                cx = lefts[k] + pct_val / 2
                ax.text(cx, y[k], f"{cnt}\n({pct_val:.0f}%)",
                        ha="center", va="center", fontsize=9, fontweight="bold")

        lefts += np.array(vals)

    for k, det in enumerate(all_detectors):
        ax.text(lefts[k] + 1.5, y[k], f"n={totals[det]}",
                ha="left", va="center", fontsize=10, fontstyle="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(all_detectors, fontsize=11, fontweight="bold")
    ax.set_xlabel("Decisive feature distribution (%)", fontsize=12)
    ax.set_xlim(0, 115)
    ax.legend(loc="upper right", fontsize=9, title="Feature", title_fontsize=10)

    plt.tight_layout()
    fname = "RQ3B_normalized_bars_decisive_features.png"
    plt.savefig(os.path.join(plots_dir, fname), dpi=220)
    plt.close(fig)
    print(f"  [RQ3B] Saved {fname}")


# ═══════════════════════════════════════════
# RQ3C — Per-round evolution heatmap (all detectors)
# ═══════════════════════════════════════════
def plot_rq3c_round_evolution(assignments: pd.DataFrame, plots_dir: str) -> None:
    """Heatmap: rows = features, columns = detector × round. Shows how feature
    assignments shift across rounds for each detector."""
    if assignments.empty:
        print("  [RQ3C] No assignment data -> skip round evolution")
        return

    # Build column labels like "bart R0", "bart R1", "roberta R0", etc.
    assignments = assignments.copy()
    assignments["det_round"] = assignments["detector"] + " R" + assignments["round"].astype(str)

    pivot = assignments.groupby(["det_round", "feature"]).size().reset_index(name="count")
    matrix = pivot.pivot_table(index="feature", columns="det_round", values="count", fill_value=0)

    # Sort columns by detector then round
    col_order = []
    for det in sorted(assignments["detector"].unique()):
        for r in sorted(assignments[assignments["detector"] == det]["round"].unique()):
            label = f"{det} R{r}"
            if label in matrix.columns:
                col_order.append(label)

    all_features = sorted(matrix.index)
    matrix = matrix.loc[all_features, col_order]

    fig, ax = plt.subplots(figsize=(max(8, len(col_order) * 1.8), max(4, len(all_features) * 1.2)))

    data = matrix.values.astype(float)
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto",
                   norm=mcolors.Normalize(vmin=0, vmax=data.max() if data.max() > 0 else 1))

    for i in range(len(all_features)):
        for j in range(len(col_order)):
            val = int(data[i, j])
            color = "white" if val > data.max() * 0.55 else "black"
            ax.text(j, i, str(val) if val > 0 else "—",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, fontsize=9, rotation=35, ha="right", fontweight="bold")
    ax.set_yticks(range(len(all_features)))
    ax.set_yticklabels(all_features, fontsize=10)
    ax.set_ylabel("Feature", fontsize=12)

    # Add vertical separators between detectors
    detectors_in_data = sorted(assignments["detector"].unique())
    cum = 0
    for det in detectors_in_data[:-1]:
        n_cols = len([c for c in col_order if c.startswith(det + " ")])
        cum += n_cols
        ax.axvline(x=cum - 0.5, color="black", linewidth=1.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Assignment count", fontsize=10)

    plt.tight_layout()
    fname = "RQ3C_heatmap_round_evolution.png"
    plt.savefig(os.path.join(plots_dir, fname), dpi=220)
    plt.close(fig)
    print(f"  [RQ3C] Saved {fname}")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
def run(output_root: str = "Output") -> int:
    plots_dir = os.path.join(output_root, "analysis_plots", "rq3")
    ensure_dir(plots_dir)

    detectors = discover_detectors(output_root)
    if not detectors:
        print("[RQ3] Nessun detector trovato.")
        return 1

    print(f"[RQ3] Detector trovati: {detectors}")

    # ── Collect data ──
    assignments = collect_all_assignments(output_root, detectors)
    decisive = collect_decisive_features(output_root, detectors)
    techniques = collect_techniques(output_root, detectors)

    # NaM technique aggregation (PUNTO 5)
    if not techniques.empty:
        tech_nonempty = techniques[techniques["technique_used"].str.strip() != ""].copy()
        if not tech_nonempty.empty:
            tech_agg = (
                tech_nonempty.groupby(["detector", "technique_used"]).size()
                .reset_index(name="count")
            )
            tech_agg.to_csv(os.path.join(plots_dir, "RQ3D_techniques_by_detector.csv"), index=False)
            print(f"  Saved RQ3D_techniques_by_detector.csv ({len(tech_agg)} rows)")
            # Outcome split: flipped = became_real_round notna
            if "became_real_round" in tech_nonempty.columns:
                tech_nonempty["flipped"] = tech_nonempty["became_real_round"].notna()
                tech_out = (
                    tech_nonempty.groupby(["detector", "technique_used", "flipped"]).size()
                    .reset_index(name="count")
                )
                tech_out.to_csv(os.path.join(plots_dir, "RQ3D_techniques_outcome.csv"), index=False)
                print(f"  Saved RQ3D_techniques_outcome.csv ({len(tech_out)} rows)")
        else:
            print("  [RQ3D] technique_used column found but empty for all rows — skipping aggregation")
    else:
        print("  [RQ3D] No technique_used data available — skipping aggregation")

    # Save raw CSVs
    if not assignments.empty:
        agg = assignments.groupby(["detector", "round", "feature"]).size().reset_index(name="count")
        agg.to_csv(os.path.join(plots_dir, "RQ3A_all_assignments.csv"), index=False)
        print(f"  Saved RQ3A_all_assignments.csv ({len(agg)} rows)")

    if not decisive.empty:
        dec_agg = decisive.groupby(["detector", "feature"]).size().reset_index(name="count")
        dec_agg.to_csv(os.path.join(plots_dir, "RQ3B_decisive_features.csv"), index=False)
        print(f"  Saved RQ3B_decisive_features.csv ({len(dec_agg)} rows)")

    # ── RQ3A: feature assignments (cross-detector) ──
    print(f"\n[RQ3] ===== RQ3A: Feature assignments =====")
    plot_rq3a_heatmap(assignments, plots_dir)
    plot_rq3a_normalized_bars(assignments, plots_dir)

    # ── RQ3B: decisive features (cross-detector) ──
    print(f"\n[RQ3] ===== RQ3B: Decisive features =====")
    plot_rq3b_heatmap(decisive, plots_dir)
    plot_rq3b_normalized_bars(decisive, plots_dir)

    # ── RQ3C: per-round evolution (all detectors in one heatmap) ──
    print(f"\n[RQ3] ===== RQ3C: Round evolution =====")
    plot_rq3c_round_evolution(assignments, plots_dir)

    # ── Print summary ──
    print(f"\n  ┌─── RQ3 Summary ───")
    if not assignments.empty:
        for det in sorted(assignments["detector"].unique()):
            sub = assignments[assignments["detector"] == det]
            feat_counts = sub["feature"].value_counts()
            top = ", ".join([f"{f} ({c})" for f, c in feat_counts.items()])
            print(f"  │ {det}: {top}")
    if not decisive.empty:
        print(f"  │")
        print(f"  │ Decisive features (tipped ID to REAL):")
        for det in sorted(decisive["detector"].unique()):
            sub = decisive[decisive["detector"] == det]
            feat_counts = sub["feature"].value_counts()
            top = ", ".join([f"{f} ({c})" for f, c in feat_counts.items()])
            print(f"  │   {det}: {top}")
    print(f"  └{'─' * 55}")

    print(f"\n[RQ3] Tutti i grafici salvati in: {plots_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RQ3 — Structural features analysis.")
    p.add_argument("--output_root", type=str, default="Output")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(run(output_root=args.output_root))
