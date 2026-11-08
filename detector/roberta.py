import os
from functools import lru_cache
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Default = original public checkpoint. Set ROBERTA_MODEL_PATH to a local
# fine-tuned checkpoint dir (exp4 defense) to load the hardened model instead;
# the default keeps exp1/exp2/exp3 runs unchanged and works under HF_HUB_OFFLINE.
MODEL_NAME = os.environ.get("ROBERTA_MODEL_PATH", "hamzab/roberta-fake-news-classification")
MAX_LEN = 512

_PRINTED = False

@lru_cache(maxsize=1)
def _load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def _map_label(model, pred_id: int) -> str:
    # Use model config mapping if available
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and pred_id in id2label:
        raw = str(id2label[pred_id]).upper()
        # normalize common variants
        if "FAKE" in raw:
            return "FAKE"
        if "REAL" in raw or "TRUE" in raw:
            return "REAL"
        if raw in ("LABEL_0", "LABEL0"):
            # unknown -> fall back below
            pass
        if raw in ("LABEL_1", "LABEL1"):
            pass
        return raw
    # Fallback (only if config missing)
    return "FAKE" if pred_id == 0 else "REAL"

@torch.inference_mode()
def detect_fake_news_roberta(title: str, text: str, device: Optional[str] = None) -> Tuple[str, float]:
    global _PRINTED
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model = _load()
    model = model.to(device)

    # keep your format, but make it explicit + spaced to avoid token-glue issues
    input_str = f"<title> {title or ''} <content> {text or ''} <end>"

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
    pred_label = _map_label(model, pred_id)

    # print mapping once for debugging
    if not _PRINTED:
        _PRINTED = True
        print("[roberta] config.id2label:", getattr(model.config, "id2label", None))
        print("[roberta] config.label2id:", getattr(model.config, "label2id", None))
        print("[roberta] example pred_id -> label:", pred_id, "->", pred_label)

    return pred_label, conf

