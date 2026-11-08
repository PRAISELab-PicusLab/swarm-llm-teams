import inspect
from typing import Optional

import pandas as pd
from tqdm import tqdm
import torch

# NOTE (swarm, offline compute nodes): only the lennywinks detector is in hf_cache, and several
# of these modules load their HF model at import time -> commented out. Original: detector_runner.py.bak_pre_comment
# Existing detectors
# from detector.bart import detect_fake_news_bart
# from detector.fake_news_bert import detect_fake_news_bert
# from detector.roberta import detect_fake_news_roberta
# from detector.valhalla import detect_fake_news_valhalla
# from detector.longformer import detect_fake_news_longformer
# from detector.llama import detect_fake_news_llama
# Probed candidates (see detector_probe_local/): integrated after local GPU probe.
# from detector.longformer_awaitinf import detect_fake_news_longformer_awaitinf
# from detector.deberta_arko import detect_fake_news_deberta_arko
# from detector.roberta_welfake_akan import detect_fake_news_roberta_welfake_akan
# from detector.josu_distilbert import detect_fake_news_josu_distilbert
# from detector.bert_tiny import detect_fake_news_bert_tiny
from detector.lennywinks import detect_fake_news_lennywinks

DETECTOR_FUNCS = {
    # "bart":                  detect_fake_news_bart,
    # "bert":                  detect_fake_news_bert,
    # "roberta":               detect_fake_news_roberta,
    # "valhalla":              detect_fake_news_valhalla,
    # "longformer":            detect_fake_news_longformer,
    # "llama":                 detect_fake_news_llama,
    # "longformer_awaitinf":   detect_fake_news_longformer_awaitinf,
    # "deberta_arko":          detect_fake_news_deberta_arko,
    # "roberta_welfake_akan":  detect_fake_news_roberta_welfake_akan,
    # "josu_distilbert":       detect_fake_news_josu_distilbert,
    # "bert_tiny":             detect_fake_news_bert_tiny,
    "lennywinks":            detect_fake_news_lennywinks,
}

def _call_detector(fn, title: str, text: str, device: Optional[str]):
    """
    Call detector passing `device` only if supported.
    """
    try:
        sig = inspect.signature(fn)
        if "device" in sig.parameters:
            return fn(title, text, device=device)
        return fn(title, text)
    except TypeError:
        return fn(title, text)

def run_detector_on_df(
    detector: str,
    df: pd.DataFrame,
    device: Optional[str] = None,
    id_col: str = "ID",
    title_col: str = "title",
    text_col: str = "text",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Runs the chosen detector over a DataFrame.
    Returns columns: [ID, pred_label, confidence]
    """
    if detector not in DETECTOR_FUNCS:
        raise ValueError(
            f"Detector '{detector}' non supportato. "
            f"Supportati: {list(DETECTOR_FUNCS.keys())}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"[DETECTOR] Uso device: {device}")

    fn = DETECTOR_FUNCS[detector]

    results = []
    it = range(len(df))
    if verbose:
        it = tqdm(it, desc=f"Detector {detector} su DataFrame")

    for i in it:
        row = df.iloc[i]
        id_ = str(row.get(id_col, i))
        title = str(row.get(title_col, ""))
        text = str(row.get(text_col, ""))

        pred_label, confidence = _call_detector(fn, title, text, device)
        results.append((id_, pred_label, confidence))

    return pd.DataFrame(results, columns=["ID", "pred_label", "confidence"])


class DetectorRunner:
    """Class wrapper for running a detector over a DataFrame.

    Added for consistency with the class-based style used in preprocessing/inference.
    The underlying detection logic is unchanged.
    """

    def __init__(
        self,
        detector: str,
        device: Optional[str] = None,
        id_col: str = "ID",
        title_col: str = "title",
        text_col: str = "text",
        verbose: bool = True,
    ):
        self.detector = detector
        self.device = device
        self.id_col = id_col
        self.title_col = title_col
        self.text_col = text_col
        self.verbose = verbose

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return run_detector_on_df(
            detector=self.detector,
            df=df,
            device=self.device,
            id_col=self.id_col,
            title_col=self.title_col,
            text_col=self.text_col,
            verbose=self.verbose,
        )
