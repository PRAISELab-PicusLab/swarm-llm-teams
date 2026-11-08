"""Detector verdict + dual SHAP feedback for one swarm round.

Reuses the pipeline: detector from DETECTOR_FUNCS, the 23 metrics from
compute_agent_metrics (round-0 semantics), and the frozen SGL surrogate trained by
`pipeline baseline0` (Output_root/meta_frozen/<detector>/).
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from pipeline.detector_runner import DETECTOR_FUNCS
from pipeline.metrics_runner import METRICS_REGISTRY, compute_agent_metrics, write_baseline_references

AGENT_CSVS = ["SemanticAnalyzer", "SalientSentenceEditor", "NumberModifier", "NarrativeModifier",
              "TitleEditor", "Generated_Fake_News"]


class Surrogate:
    def __init__(self, meta_dir: str):
        self.pipe = joblib.load(os.path.join(meta_dir, "meta_model_pipe.pkl"))
        with open(os.path.join(meta_dir, "feature_meta.json"), encoding="utf-8") as f:
            self.cols = json.load(f)["feature_cols"]
        self.coef = np.ravel(self.pipe.named_steps["clf"].coef_)
        # Background = round-0 training matrix, as in meta_model.py (shap Independent masker).
        self.bg = np.load(os.path.join(meta_dir, "X_all_std.npy")).mean(axis=0)

    def explain(self, feats: pd.DataFrame):
        X = feats[self.cols]
        Xs = self.pipe.named_steps["scale"].transform(self.pipe.named_steps["impute"].transform(X))
        # Linear model + independent masker: SHAP is exactly coef * (x - E[x]) in log-odds space.
        psi = self.coef * (Xs - self.bg)
        return self.pipe.predict_proba(X)[:, 1], psi


def _item(feature: str, psi: float, coef: float) -> dict:
    spec = METRICS_REGISTRY[feature]
    return {"feature": feature, "agent": spec.agent_id, "meaning": spec.description,
            "shap": round(float(psi), 3),
            "to_look_real": "increase" if coef > 0 else "decrease"}


def dual_shap(cols, coef, psi_row, k: int = 3):
    """Top-k features that pushed toward FAKE (hurt) and toward REAL (helped)."""
    order = np.argsort(psi_row)
    hurt = [_item(cols[i], psi_row[i], coef[i]) for i in order[:k] if psi_row[i] < 0]
    helped = [_item(cols[i], psi_row[i], coef[i]) for i in order[::-1][:k] if psi_row[i] > 0]
    return hurt, helped


def merge_unit_dirs(unit_dirs, out_dir: str) -> None:
    """Concatenate the per-unit pipeline CSVs so metrics load their models once per round."""
    os.makedirs(out_dir, exist_ok=True)
    for name in AGENT_CSVS:
        parts = [pd.read_csv(p, encoding="utf-8", dtype={"ID": str})
                 for p in (os.path.join(d, f"{name}.csv") for d in unit_dirs) if os.path.exists(p)]
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False, encoding="utf-8")


def score_round(gen_dir: str, df_orig: pd.DataFrame, detector: str, residualizer: str, surrogate: Surrogate) -> dict:
    """ID -> {label, confidence, title, p_real_surrogate, hurt, helped}. IDs the pipeline skipped are absent."""
    gen_csv = os.path.join(gen_dir, "Generated_Fake_News.csv")
    if not os.path.exists(gen_csv):
        return {}
    gen = pd.read_csv(gen_csv, encoding="utf-8", dtype={"ID": str}).drop_duplicates("ID", keep="last")
    out = {}
    for _, r in gen.iterrows():
        label, conf = DETECTOR_FUNCS[detector](str(r["title"]), str(r["text"]))
        out[r["ID"]] = {"label": label, "confidence": round(float(conf), 4), "title": str(r["title"])}
    if not out:
        return out

    orig = df_orig[df_orig["ID"].isin(out)].copy()
    refs = os.path.join(gen_dir, "baseline_references.json")
    write_baseline_references(sa_csv_path=os.path.join(gen_dir, "SemanticAnalyzer.csv"), df_orig=orig, out_path=refs)
    compute_agent_metrics(df_orig=orig, base_out_dir=gen_dir, residualizer_path=residualizer,
                          baseline_references_path=refs)
    feats = pd.read_csv(os.path.join(gen_dir, "agent_metrics.csv"), encoding="utf-8", dtype={"ID": str}).set_index("ID")
    feats = feats.loc[[i for i in out if i in feats.index]]
    p_real, psi = surrogate.explain(feats)
    for j, aid in enumerate(feats.index):
        hurt, helped = dual_shap(surrogate.cols, surrogate.coef, psi[j])
        out[aid].update(p_real_surrogate=round(float(p_real[j]), 3), hurt=hurt, helped=helped)
    return out
