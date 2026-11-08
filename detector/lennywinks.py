from functools import lru_cache
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# DistilBERT-cased fake-news detector (WELFake + WebzIO), chosen for the swarm
# hackathon by swarm/detector_screen: moderate round-0 fool rate + diffuse surrogate
# importance. config.id2label is explicit: {0: fake, 1: real}.
MODEL_NAME = "Lennywinks/fake-news-detector-english"
MAX_LEN = 512
_ID2LABEL = {0: "FAKE", 1: "REAL"}


@lru_cache(maxsize=1)
def _load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def detect_fake_news_lennywinks(title: str, text: str, device: Optional[str] = None) -> Tuple[str, float]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = _load()
    model = model.to(device)
    enc = tokenizer(f"{title or ''} {text or ''}".strip(), truncation=True, max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    probs = F.softmax(model(**enc).logits.squeeze(0), dim=-1)
    pred_id = int(torch.argmax(probs).item())
    return _ID2LABEL[pred_id], float(probs[pred_id].item())
