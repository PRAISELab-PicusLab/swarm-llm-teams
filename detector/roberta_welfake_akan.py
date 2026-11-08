from functools import lru_cache
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# RoBERTa fine-tuned on WELFake (akanbrown/AI-Fake_News_Detection_Roberta).
# config has GENERIC labels (LABEL_0/LABEL_1) so the map is hardcoded from the
# cohort probe: REAL seeds -> id 1 (1500/1500). Hence 0 = FAKE, 1 = REAL.
MODEL_NAME = "akanbrown/AI-Fake_News_Detection_Roberta"
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
def detect_fake_news_roberta_welfake_akan(title: str, text: str, device: Optional[str] = None) -> Tuple[str, float]:
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
        print("[roberta_welfake_akan] MODEL:", MODEL_NAME)
        print("[roberta_welfake_akan] hardcoded id2label (config generic, IGNORED):", _ID2LABEL)
        print("[roberta_welfake_akan] config.id2label:", getattr(model.config, "id2label", None))
        print("[roberta_welfake_akan] example pred_id -> label:", pred_id, "->", pred_label)

    return pred_label, conf
