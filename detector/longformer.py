import os
from functools import lru_cache
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ready-made fake-news Longformer (long-context). Model card (PushkarKumar/veritas_ai_new):
# base allenai/longformer-base-4096, trained on ISOT, title+text concatenated, max 1024 tokens.
MODEL_NAME = "PushkarKumar/veritas_ai_new"
MAX_LEN = 1024  # model card: "Max sequence length used: 1024 tokens" (faithful to fine-tuning)

# CRITICAL label convention: this checkpoint is 0 = REAL, 1 = FAKE (from the model card).
# It is INVERTED vs the other detectors here (roberta/GonzaloA-DeBERTa use 0 = FAKE).
# The checkpoint's config.json carries NO id2label, so we map explicitly and DELIBERATELY
# ignore model.config.id2label. Direction is verified empirically on the login node before
# any real run (see validation snippet / memory).
_ID2LABEL = {0: "REAL", 1: "FAKE"}

_PRINTED = False


@lru_cache(maxsize=1)
def _load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def _map_label(pred_id: int) -> str:
    # Hardcoded on purpose: config has no id2label AND the convention is reversed here.
    return _ID2LABEL.get(pred_id, "FAKE")


@torch.inference_mode()
def detect_fake_news_longformer(title: str, text: str, device: Optional[str] = None) -> Tuple[str, float]:
    global _PRINTED
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model = _load()
    model = model.to(device)

    # Model card: title and text concatenated into a single sequence (no special markers,
    # unlike roberta's <title>/<content> format — those would just be literal noise here).
    input_str = f"{title or ''} {text or ''}".strip()

    enc = tokenizer(
        input_str,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # Longformer classification needs global attention on the [CLS] token (position 0).
    # (LongformerForSequenceClassification auto-sets this if omitted, but we make it explicit.)
    global_attention_mask = torch.zeros_like(enc["input_ids"])
    global_attention_mask[:, 0] = 1
    enc["global_attention_mask"] = global_attention_mask

    logits = model(**enc).logits.squeeze(0)
    probs = F.softmax(logits, dim=-1)

    pred_id = int(torch.argmax(probs).item())
    conf = float(probs[pred_id].item())
    pred_label = _map_label(pred_id)

    if not _PRINTED:
        _PRINTED = True
        print("[longformer] MODEL:", MODEL_NAME)
        print("[longformer] hardcoded id2label (config IGNORED):", _ID2LABEL)
        print("[longformer] config.id2label (NOT used):", getattr(model.config, "id2label", None))
        print("[longformer] example pred_id -> label:", pred_id, "->", pred_label)

    return pred_label, conf
