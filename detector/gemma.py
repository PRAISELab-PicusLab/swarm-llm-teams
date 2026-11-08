from agents import user_proxy, GemmaFakeNewsDetector
from check_json_format import check_llm_format


def detect_fake_news_gemma(title: str, text: str):
    prompt = f"Determine if the following is real or fake news:\n{title}\n{text}."

    user_proxy.initiate_chat(GemmaFakeNewsDetector, message=prompt)

    ok, final_answer = check_llm_format(GemmaFakeNewsDetector)
    if ok:
        return final_answer, 1
    return "FAKE", -1
