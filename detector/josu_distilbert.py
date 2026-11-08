from functools import lru_cache
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# DistilBERT fake-news detector (JosuMSC/fake-news-detector).
# config.id2label uses sentiment-style names {0: NEGATIVE, 1: POSITIVE}; the cohort
# probe shows REAL seeds -> id 1 (POSITIVE). We hardcode 0 = FAKE, 1 = REAL.
MODEL_NAME = "JosuMSC/fake-news-detector"
MAX_LEN = 512
_ID2LABEL = {0: "FAKE", 1: "REAL"}

_PRINTED = False


@lru_cache(maxsize=1)
def _load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def _map_label(pred_id: int) -> str:
    return _ID2LABEL.get(pred_id, "FAKE")


@torch.inference_mode()
def detect_fake_news_josu_distilbert(title: str, text: str, device: Optional[str] = None) -> Tuple[str, float]:
    global _PRINTED
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model = _load()
    model = model.to(device)

    input_str = f"{title or ''} {text or ''}".strip()

    enc = tokenizer(
        input_str,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    logits = model(**enc).logits.squeeze(0)
    probs = F.softmax(logits, dim=-1)

    pred_id = int(torch.argmax(probs).item())
    conf = float(probs[pred_id].item())
    pred_label = _map_label(pred_id)

    if not _PRINTED:
        _PRINTED = True
        print("[josu_distilbert] MODEL:", MODEL_NAME)
        print("[josu_distilbert] hardcoded id2label (sentiment names, IGNORED):", _ID2LABEL)
        print("[josu_distilbert] config.id2label:", getattr(model.config, "id2label", None))
        print("[josu_distilbert] example pred_id -> label:", pred_id, "->", pred_label)

    return pred_label, conf
