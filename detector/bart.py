from transformers import pipeline
import pandas as pd

# Inizializza la pipeline zero-shot classification con modello NLI (ad es. bart-large-mnli)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def detect_fake_news_bart(title: str, text: str):
    # Concateno titolo e testo
    combined_text = title + "\n" + text

    # Definisco le candidate labels
    candidate_labels = ["REAL", "FAKE"]

    # Faccio inferenza
    result = classifier(combined_text, candidate_labels)

    # Estraggo label con la score più alta
    predicted_label = result['labels'][0]
    confidence = result['scores'][0]

    return predicted_label, confidence
