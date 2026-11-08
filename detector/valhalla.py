import os

from transformers import pipeline

# VALHALLA_MODEL_PATH lets exp4 point the detector at a hardened checkpoint
# (defense/finetune_detector.py, variant nli). Unset -> original public model.
MODEL_NAME = os.environ.get("VALHALLA_MODEL_PATH", "valhalla/distilbart-mnli-12-1")

# Inizializza la pipeline zero-shot classification con modello distillato
classifier = pipeline("zero-shot-classification", model=MODEL_NAME)


def detect_fake_news_valhalla(title: str, text: str):
    combined_text = title + "\n" + text
    candidate_labels = ["REAL", "FAKE"]
    result = classifier(combined_text, candidate_labels)
    predicted_label = result['labels'][0]
    confidence = result['scores'][0]
    return predicted_label, confidence
