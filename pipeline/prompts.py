from .utils import read_from_file

# SEMANTIC ANALYZER #
semantic_analyzer_system_prompt = read_from_file("system_prompt.txt", "prompts/SemanticAnalyzer/")
check_semantic_analyzer_format_prompt = read_from_file("check_semantic_analyzer_format.txt", "prompts/SemanticAnalyzer/")

# SALIENT SENTENCE EDITOR #
salient_sentence_editor_system_prompt = read_from_file("system_prompt.txt", "prompts/SalientSentenceEditor/")
check_salient_sentence_editor_format_prompt = read_from_file("check_salient_sentence_editor_format.txt", "prompts/SalientSentenceEditor/")

# NUMBER MODIFIER #
number_modifier_system_prompt = read_from_file("system_prompt.txt", "prompts/NumberModifier/")
check_number_modifier_format_prompt = read_from_file("check_number_modifier_format.txt", "prompts/NumberModifier/")

# NARRATIVE MODIFIER #
narrative_modifier_system_prompt = read_from_file("system_prompt.txt", "prompts/NarrativeModifier/")
check_narrative_modifier_format_prompt = read_from_file("check_narrative_modifier_format.txt", "prompts/NarrativeModifier/")

# TITLE EDITOR #
title_editor_system_prompt = read_from_file("system_prompt.txt", "prompts/TitleEditor/")
check_title_editor_format_prompt = read_from_file("check_title_editor_format.txt", "prompts/TitleEditor/")

# LLM FAKE NEWS DETECTOR #
llm_fake_news_detector_system_prompt = read_from_file("system_prompt.txt", "prompts/LLMFakeNewsDetector/")
check_llm_fake_news_detector_format_prompt = read_from_file("check_llm_fake_news_detector_format.txt", "prompts/LLMFakeNewsDetector/")
