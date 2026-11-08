import os
from functools import lru_cache
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Longformer-base-4096 fine-tuned for fake news (Awaitinf/longformer-base-fake_news).
# Distinct from detector/longformer.py (PushkarKumar/veritas_ai_new).
# config.id2label = {0: FAKE, 1: REAL} and is reliable (verified on cohort probe:
# 1500/1500 REAL seeds -> id 1). We still hardcode the map to be robust.
# Default = original public checkpoint. Set LONGFORMER_AWAITINF_MODEL_PATH to a
# local fine-tuned checkpoint dir (exp4 defense) to load the hardened model; the
# 0=FAKE/1=REAL convention is preserved by the fine-tuning script.
MODEL_NAME = os.environ.get("LONGFORMER_AWAITINF_MODEL_PATH", "Awaitinf/longformer-base-fake_news")
MAX_LEN = 1024
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
def detect_fake_news_longformer_awaitinf(title: str, text: str, device: Optional[str] = None) -> Tuple[str, float]:
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

    # Longformer needs global attention on the [CLS] token (position 0).
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
        print("[longformer_awaitinf] MODEL:", MODEL_NAME)
        print("[longformer_awaitinf] hardcoded id2label:", _ID2LABEL)
        print("[longformer_awaitinf] config.id2label:", getattr(model.config, "id2label", None))
        print("[longformer_awaitinf] example pred_id -> label:", pred_id, "->", pred_label)

    return pred_label, conf
