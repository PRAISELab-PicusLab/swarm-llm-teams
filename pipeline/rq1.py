"""
RQ1 — Il framework riesce a generare fake news capaci di ingannare il classificatore?

Grafici prodotti:
  RQ1A — Fooling rate globale per detector (barre, tutti i detector insieme)
  RQ1B — Fooling rate cumulativo per round (un grafico per detector)

Uso:
  python -m pipeline rq1 --output_root Output
  python pipeline/rq1.py --output_root Output
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
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
        if os.path.isfile(det_csv) and os.path.getsize(det_csv) > 10:
            rounds.append(r)
    return sorted(rounds)


def load_detector_round0(output_root: str, detector: str) -> pd.DataFrame:
    p = os.path.join(output_root, "rounds", "round_000", "detectors", detector, f"detector_{detector}.csv")
    return pd.read_csv(p, encoding="utf-8")


def load_ledger(output_root: str, detector: str) -> pd.DataFrame | None:
    p = os.path.join(output_root, "tracking", detector, "baseline_fakes_ledger.csv")
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p, encoding="utf-8")
    df["ID"] = df["ID"].astype(str)
    return df


# ═══════════════════════════════════════════
# RQ1-A  Fooling rate globale
# ═══════════════════════════════════════════
def compute_global_fooling(output_root: str, detectors: list[str]) -> pd.DataFrame:
    rows = []
    for det in detectors:
        det0 = load_detector_round0(output_root, det)
        n_total = len(det0)
        n_real_r0 = int((det0["pred_label"] == "REAL").sum())

        # ID che diventano REAL nei round successivi (dal ledger)
        n_became_real_later = 0
        ledger = load_ledger(output_root, det)
        if ledger is not None and "became_real_round" in ledger.columns:
            n_became_real_later = int(ledger["became_real_round"].notna().sum())

        n_fooled = n_real_r0 + n_became_real_later
        # Evita doppio conteggio: gli ID REAL al round 0 NON sono nel ledger
        # (il ledger contiene solo i baseline fakes)
        rate = n_fooled / n_total * 100 if n_total > 0 else 0.0

        rows.append({
            "detector": det,
            "n_total": n_total,
            "n_real_r0": n_real_r0,
            "n_became_real_later": n_became_real_later,
            "n_fooled": n_fooled,
            "fooling_rate_pct": round(rate, 2),
        })
    return pd.DataFrame(rows)


def plot_rq1a(df: pd.DataFrame, plots_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 2.2), 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(df), 2)))

    bars = ax.bar(df["detector"], df["fooling_rate_pct"], color=colors[: len(df)], edgecolor="black", width=0.55)

    for bar, row in zip(bars, df.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{row.fooling_rate_pct:.1f}%\n({row.n_fooled}/{row.n_total})",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Fooling rate (%)", fontsize=12)
    ax.set_xlabel("Detector", fontsize=12)
    ax.set_ylim(0, min(df["fooling_rate_pct"].max() + 12, 110))
    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "RQ1A_fooling_rate_global.png"), dpi=220)
    plt.close(fig)
    print(f"  [RQ1A] Saved RQ1A_fooling_rate_global.png")


# ═══════════════════════════════════════════
# RQ1-B  Fooling rate cumulativo per round
# ═══════════════════════════════════════════
def compute_cumulative_fooling(output_root: str, detector: str) -> pd.DataFrame:
    det0 = load_detector_round0(output_root, detector)
    n_total = len(det0)
    n_real_r0 = int((det0["pred_label"] == "REAL").sum())

    rounds = discover_rounds_for_detector(output_root, detector)
    ledger = load_ledger(output_root, detector)

    rows = []
    cumul = n_real_r0  # partenza: chi è già REAL al round 0

    for r in rounds:
        new_real = 0
        if r == 0:
            new_real = n_real_r0
            cumul_at_r = n_real_r0
        else:
            if ledger is not None and "became_real_round" in ledger.columns:
                new_real = int((ledger["became_real_round"] == float(r)).sum())
            cumul += new_real
            cumul_at_r = cumul

        rate = cumul_at_r / n_total * 100 if n_total > 0 else 0.0
        rows.append({
            "round": r,
            "new_real_this_round": new_real,
            "cumulative_fooled": cumul_at_r,
            "n_total": n_total,
            "fooling_rate_pct": round(rate, 2),
        })

    return pd.DataFrame(rows)


def plot_rq1b(df: pd.DataFrame, detector: str, plots_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 1.8), 6))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(df), 2)))

    bars = ax.bar(df["round"].astype(str), df["fooling_rate_pct"], color=colors[: len(df)], edgecolor="black", width=0.55)

    for bar, row in zip(bars, df.itertuples()):
        label = f"{row.fooling_rate_pct:.1f}%"
        if row.round == 0:
            note = f"+{row.new_real_this_round} REAL"
        else:
            note = f"+{row.new_real_this_round} new"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{label}\n({note})",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("Cumulative fooling rate (%)", fontsize=12)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylim(0, min(df["fooling_rate_pct"].max() + 15, 115))
    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    fname = f"RQ1B_fooling_rate_per_round_{detector}.png"
    plt.savefig(os.path.join(plots_dir, fname), dpi=220)
    plt.close(fig)
    print(f"  [RQ1B] Saved {fname}")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
def run(output_root: str = "Output") -> int:
    plots_dir = os.path.join(output_root, "analysis_plots", "rq1")
    ensure_dir(plots_dir)

    detectors = discover_detectors(output_root)
    if not detectors:
        print("[RQ1] Nessun detector trovato. Assicurati che Output/rounds/round_000/detectors/ contenga dati.")
        return 1

    print(f"[RQ1] Detector trovati: {detectors}")

    # ---------- RQ1-A ----------
    df_global = compute_global_fooling(output_root, detectors)
    df_global.to_csv(os.path.join(plots_dir, "RQ1A_fooling_rate_global.csv"), index=False)
    plot_rq1a(df_global, plots_dir)

    # ---------- RQ1-B ----------
    for det in detectors:
        df_cum = compute_cumulative_fooling(output_root, det)
        df_cum.to_csv(os.path.join(plots_dir, f"RQ1B_fooling_rate_per_round_{det}.csv"), index=False)
        plot_rq1b(df_cum, det, plots_dir)

    print(f"\n[RQ1] Tutti i grafici salvati in: {plots_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RQ1 — Fooling rate analysis.")
    p.add_argument("--output_root", type=str, default="Output")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(run(output_root=args.output_root))
