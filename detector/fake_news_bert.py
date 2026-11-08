from transformers import pipeline

MODEL = "jy46604790/Fake-News-Bert-Detect"
clf = pipeline("text-classification", model=MODEL, tokenizer=MODEL)


def detect_fake_news_bert(title, text):
    input_str = f"{title} {text}"
    result = clf(input_str, truncation=True)[0]

    # Mappatura etichetta
    label_map = {
        "LABEL_1": "REAL",
        "LABEL_0": "FAKE"
    }

    label = label_map.get(result["label"], result["label"])
    confidence = result['score']

    return label, confidence
