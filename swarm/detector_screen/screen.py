"""Screen candidate HF detectors for the swarm game (no LLM cost).

For each candidate, on the short-article subset (<= MAX_WORDS words) of the ATN cohort:
  spec  = % of ORIGINAL articles labelled REAL               (want >= 90%)
  fool  = % of round-0 generated fakes labelled REAL          (want ~30-70%: lives matter, room to learn)
  surrogate = L1 logistic on the 23 round-0 metrics -> detector label:
      auc (5-fold CV), top feature, top1 share of sum|beta|, TE share of sum|beta|  (want diffuse, not title-only)

Reuses load_model/score_batch from detector_probe_local/probe_local.py.
Usage: venv/Scripts/python.exe swarm/detector_screen/screen.py [--models a,b] [--limit N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "detector_probe_local"))
import probe_local as pl  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
ATN = os.path.join(ROOT, "Output_All_TheNews")
MAX_WORDS = 400

CANDIDATES = [
    "XSY/albert-base-v2-fakenews-discriminator",
    "dhruvpal/fake-news-bert",
    "GhostFaith/distilbert-fakenews",
    "haithamd/fake-news-bert",
    "Pulk17/Fake-News-Detection",
    "vinitsingare/distilbert_fake_news",
    "vikram71198/distilroberta-base-finetuned-fake-news-detection",
    "Lennywinks/fake-news-detector-english",
    "hadeel295/bert-fake-news-classifier",
    "Sumra81/fake-news-distilbert",
    "Giyaseddin/distilbert-base-cased-finetuned-fake-and-real-news-dataset",
    "shalev396/fake-news-detector",
    "therealcyberlord/fake-news-classification-distilbert",
    "winterForestStump/Roberta-fake-news-detector",
    "Pavan48/fake_news_detection_roberta",
    "zodumair/sifta-fake-news-detector-v2",
    "MYC007/Real-and-Fake-News-Detection",
    "Youssef-El-SaYed/Bert-fake-news-detector",
    "tomokaz777/distilbert-base-uncased-finetuned-fake-or-real-news",
    "Arko007/fake-news-liar-political",
    "RidzIn/Roberta_FakeNewsDetection",
    "adityaG04/fake-news-bert-finetuned",
    "prabhatraja445/fake-news-misinformation-distilbert-512",
    "safikhan/fake-news-small-bert",
    "anshy047/fake-news-detector-transformer",
    "ThomasTschinkel/fake-news-detector",
]
# Already-scored detectors from the paper, used as reference rows (gen0 labels from Output_All_TheNews).
REFERENCE = ["roberta", "valhalla"]


def load_data():
    orig = pd.read_csv(os.path.join(ATN, "cohort", "cohort_original.csv"), dtype={"ID": str})
    short = orig[orig["text"].astype(str).str.split().str.len() <= MAX_WORDS]
    gen0 = pd.read_csv(os.path.join(ROOT, "colab_detector_probe", "gen0_atn.csv"), dtype={"ID": str})
    feats = pd.read_csv(os.path.join(ATN, "rounds", "round_000", "generation", "agent_metrics.csv"), dtype={"ID": str})
    ids = set(short["ID"]) & set(gen0["ID"]) & set(feats["ID"])
    pick = lambda d: d[d["ID"].isin(ids)].sort_values("ID").reset_index(drop=True)
    return pick(short), pick(gen0), pick(feats)


def score(cfg, tok, model, real_ids, df):
    labels = []
    for s in range(0, len(df), cfg["batch_size"]):
        c = df.iloc[s:s + cfg["batch_size"]]
        pid, _, _ = pl.score_batch(cfg, tok, model, list(c["title"]), list(c["text"]), real_ids)
        labels += ["REAL" if p in real_ids else "FAKE" for p in pid]
    return pd.Series(labels, index=df["ID"])


def surrogate(feats, gen0_labels):
    X = feats.set_index("ID").loc[gen0_labels.index].fillna(0.0)
    y = (gen0_labels == "REAL").astype(int).values
    if min(y.sum(), len(y) - y.sum()) < 20:
        return dict(auc=np.nan, top_feature="", top1_share=np.nan, te_share=np.nan, n_nonzero=0)
    clf = LogisticRegressionCV(Cs=10, cv=5, penalty="l1", solver="liblinear", scoring="roc_auc",
                               class_weight="balanced", max_iter=2000)
    clf.fit(StandardScaler().fit_transform(X.values), y)
    beta = pd.Series(clf.coef_[0], index=X.columns)
    a = beta.abs()
    tot = a.sum() or 1.0
    top = a.idxmax()
    return dict(auc=round(float(clf.scores_[1].mean(axis=0).max()), 3),
                top_feature=f"{top} ({beta[top]:+.2f})",
                top1_share=round(a.max() / tot, 2),
                te_share=round(a[[c for c in a.index if c.startswith("TE_")]].sum() / tot, 2),
                n_nonzero=int((a > 1e-6).sum()))


def row(name, spec, gen0_labels, feats):
    fool = round(100 * (gen0_labels == "REAL").mean(), 1)
    r = dict(model=name, n=len(gen0_labels), spec_pct=spec, fool_pct=fool)
    r.update(surrogate(feats, gen0_labels))
    print(r, flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    short, gen0, feats = load_data()
    if args.limit:
        short, gen0, feats = short.head(args.limit), gen0.head(args.limit), feats.head(args.limit)
    print(f"short-article subset: {len(short)} IDs (<= {MAX_WORDS} words)")
    out_csv = os.path.join(OUT, "screen_results.csv")
    # Resume: keep rows already on disk, skip models already screened (a CUDA assert kills the
    # whole process, so the driver runs one model per process).
    rows = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) and not args.limit else []
    done = {r["model"] for r in rows}

    for det in REFERENCE:
        if f"[ref] {det}" in done:
            continue
        d = pd.read_csv(os.path.join(ATN, "rounds", "round_000", "detectors", det, f"detector_{det}.csv"),
                        dtype={"ID": str}).set_index("ID")["pred_label"]
        rows.append(row(f"[ref] {det}", 100.0, d.reindex(gen0["ID"]).dropna(), feats))

    models = args.models.split(",") if args.models else CANDIDATES
    for hf_id in [m for m in models if m not in done]:
        cfg = dict(name=hf_id.replace("/", "__"), hf_id=hf_id, arch="?", max_len=512, batch_size=16,
                   label_map=None, real_labels=None)
        try:
            tok, model, label_map, real_ids, _ = pl.load_model(cfg)
            spec_lab = score(cfg, tok, model, real_ids, short)
            gen_lab = score(cfg, tok, model, real_ids, gen0)
            spec = round(100 * (spec_lab == "REAL").mean(), 1)
            r = row(hf_id, spec, gen_lab, feats)
            r["label_map"] = str(label_map)
            r["real_ids"] = str(sorted(real_ids))
            rows.append(r)
            pd.DataFrame({"ID": gen_lab.index, "orig": spec_lab.values, "gen0": gen_lab.values}).to_csv(
                os.path.join(OUT, f"labels_{cfg['name']}.csv"), index=False)
        except Exception as e:  # one broken checkpoint must not stop the screen
            print(f"[skip] {hf_id}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            rows.append(dict(model=hf_id, error=f"{type(e).__name__}: {str(e)[:120]}"))
        finally:
            tok = model = None
            pl.free_gpu()
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"-> {out_csv}")


if __name__ == "__main__":
    main()
