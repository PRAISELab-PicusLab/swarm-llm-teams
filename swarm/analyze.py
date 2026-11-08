"""Compare scenarios for one backbone.

Usage: python -m swarm.analyze Output_Swarm/llama3_1_8b
Writes <root>/summary_scenarios.csv and <root>/curve_by_round.csv.
Baseline row: phase-1 round-0 verdicts on the SAME articles the swarm played (no memory, no feedback).
"""
import glob
import json
import os
import sys

import pandas as pd


def load(root: str, s: int) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(root, f"s{s}", "rounds", "round_*", "results.csv")))
    return pd.concat([pd.read_csv(f, dtype={"article_id": str}) for f in files], ignore_index=True) if files else pd.DataFrame()


def main(root: str):
    rows, curves, played_ids = [], [], set()
    for s in range(5):
        d = load(root, s)
        if d.empty:
            continue
        state = json.load(open(os.path.join(root, f"s{s}", "state.json"), encoding="utf-8"))
        # One row per submitted article: scenario 4 has one row per chain member.
        art = d[d["label"].isin(["REAL", "FAKE"])].drop_duplicates(["round", "slot"])
        played_ids |= set(art["article_id"])
        posts = d[d["posted"] == True] if "posted" in d else pd.DataFrame()  # noqa: E712
        rows.append({
            "scenario": s, "rounds": state["round"], "end_reason": state.get("end_reason", "running"),
            "articles_submitted": len(art), "articles_fooled": int((art["label"] == "REAL").sum()),
            "fool_rate": round((art["label"] == "REAL").mean(), 3) if len(art) else None,
            "feedback_useful_rate": round(d["feedback_was_useful"].dropna().astype(bool).mean(), 3),
            "posts": len(posts),
            "posts_self_reported_untruthful": int((posts.get("post_is_truthful") == False).sum()) if len(posts) else 0,  # noqa: E712
            "s4_chain_rounds_as_team": round(d.loc[d["in_chain"] == True, "teamed"].mean(), 3) if s == 4 else None,  # noqa: E712
        })
        c = art.groupby("round")["label"].apply(lambda x: (x == "REAL").sum()).cumsum().rename("cum_fooled").reset_index()
        c["scenario"] = s
        curves.append(c)

    det = glob.glob(os.path.join(root, "phase1", "rounds", "round_000", "detectors", "*", "detector_*.csv"))
    if det:
        b = pd.read_csv(det[0], dtype={"ID": str})
        b = b[b["ID"].isin(played_ids)]
        rows.append({"scenario": "round0_no_memory", "articles_submitted": len(b),
                     "articles_fooled": int((b["pred_label"] == "REAL").sum()),
                     "fool_rate": round((b["pred_label"] == "REAL").mean(), 3) if len(b) else None})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(root, "summary_scenarios.csv"), index=False)
    if curves:
        pd.concat(curves).to_csv(os.path.join(root, "curve_by_round.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
