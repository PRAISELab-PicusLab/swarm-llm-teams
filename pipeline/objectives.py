from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class ObjectiveSpec:
    """
    Encapsulates a single pivot objective.

    Attributes
    ----------
    text : str
        Operational instruction injected verbatim into the agent payload as
        the "objective" field.  Must reference the exact payload field names
        the agent will receive so the LLM knows how to act without ambiguity.
    technique_hint : str
        Kept for backward compatibility with callers; always empty in the
        2026-05-20 revision (NaM may freely choose any of the 9 numbered
        techniques and must report `technique_used` in the JSON output).
    payload_fields_used : tuple
        Names of payload fields that the objective text references explicitly.
        Used by the orchestrator to verify the payload is complete before
        sending the request.
    """
    text: str
    technique_hint: str = ""
    payload_fields_used: tuple = ()


# ---------------------------------------------------------------------------
# Master registry — 2026-05-20 revision
# Key: (agent_id, metric_human_name, direction)
#
# Alignment with the metric is the contract.  Each objective references ONLY
# the fields the agent needs to act AND that match what the metric actually
# measures.  Round-0 anchors (original_article, original_sentence, KS_0,
# original_*_count, target_number, NUM2SENT_0) are the ground truth coming
# from the REAL article in the dataset; previous_* fields are the agent's
# own last move and exist only to break ties / track direction.
# ---------------------------------------------------------------------------
OBJECTIVES: Dict[Tuple[str, str, str], ObjectiveSpec] = {

    # =========================================================================
    # SA — Semantic Analyzer
    # Operando: current_article (post-NaM r-1).
    # Coverage features compare KS_r / KN_r vs the round-0 original_article.
    # Compression ratio features compare count(KS_r) / count(orig_round_0).
    # All reference anchors are round-0 (REAL article), never r-1.
    # =========================================================================

    ("SA", "Semantic Coverage", "INCREASE"): ObjectiveSpec(
        text=(
            "Re-extract key_sentences from current_article so they cover MORE of "
            "the content of original_article than previous_key_sentences did. "
            "original_article is the REAL source article (round 0); "
            "previous_key_sentences is your previous extraction — your new "
            "extraction must overlap MORE with the ideas in original_article "
            "than that baseline does. "
            "Copy sentences verbatim from current_article — do not paraphrase."
        ),
        payload_fields_used=("current_article", "original_article",
                             "previous_key_sentences"),
    ),

    ("SA", "Semantic Coverage", "DECREASE"): ObjectiveSpec(
        text=(
            "Re-extract key_sentences from current_article so they cover LESS of "
            "the content of original_article than previous_key_sentences did. "
            "original_article is the REAL source article (round 0); "
            "previous_key_sentences is your previous extraction — your new "
            "extraction must overlap LESS with the ideas in original_article "
            "than that baseline does. "
            "Copy sentences verbatim from current_article — do not paraphrase."
        ),
        payload_fields_used=("current_article", "original_article",
                             "previous_key_sentences"),
    ),

    ("SA", "Text Compression Ratio", "INCREASE"): ObjectiveSpec(
        text=(
            "Re-extract key_sentences from current_article. Extract MORE "
            "sentences than you did last round. "
            "previous_sentence_count is how many key_sentences you returned "
            "last round (not cumulative) — your new list must be STRICTLY "
            "LONGER than that, bounded above by original_sentence_count "
            "(total sentences in the REAL source article, round 0). "
            "Copy sentences verbatim from current_article, keep them "
            "informative and non-redundant."
        ),
        payload_fields_used=("current_article", "original_sentence_count",
                             "previous_sentence_count"),
    ),

    ("SA", "Text Compression Ratio", "DECREASE"): ObjectiveSpec(
        text=(
            "Re-extract key_sentences from current_article. Extract FEWER "
            "sentences than you did last round. "
            "previous_sentence_count is how many key_sentences you returned "
            "last round (not cumulative) — your new list must be STRICTLY "
            "SHORTER than that. original_sentence_count is the total number "
            "of sentences in the REAL source article (round 0) — for context "
            "only. Keep only the most essential sentences, verbatim from "
            "current_article."
        ),
        payload_fields_used=("current_article", "original_sentence_count",
                             "previous_sentence_count"),
    ),

    ("SA", "Numeric Coverage", "INCREASE"): ObjectiveSpec(
        text=(
            "Re-extract key_numbers from current_article so they recall MORE of "
            "the numeric facts in original_article than previous_key_numbers did. "
            "original_article is the REAL source article (round 0); "
            "previous_key_numbers is your previous extraction baseline. "
            "Write numbers in digits, avoid duplicates."
        ),
        payload_fields_used=("current_article", "original_article",
                             "previous_key_numbers"),
    ),

    ("SA", "Numeric Coverage", "DECREASE"): ObjectiveSpec(
        text=(
            "Re-extract key_numbers from current_article so they recall FEWER of "
            "the numeric facts in original_article than previous_key_numbers did. "
            "original_article is the REAL source article (round 0); "
            "previous_key_numbers is your previous extraction baseline. "
            "Write numbers in digits, avoid duplicates."
        ),
        payload_fields_used=("current_article", "original_article",
                             "previous_key_numbers"),
    ),

    ("SA", "Numeric Compression Ratio", "INCREASE"): ObjectiveSpec(
        text=(
            "Re-extract key_numbers from current_article. Extract MORE "
            "distinct numeric values than you did last round. "
            "previous_number_count is how many key_numbers you returned last "
            "round (not cumulative) — your new list must be STRICTLY LONGER "
            "than that, bounded above by original_number_count (total "
            "distinct numeric values in the REAL source article, round 0). "
            "Write numbers in digits, avoid duplicates."
        ),
        payload_fields_used=("current_article", "original_number_count",
                             "previous_number_count"),
    ),

    ("SA", "Numeric Compression Ratio", "DECREASE"): ObjectiveSpec(
        text=(
            "Re-extract key_numbers from current_article. Extract FEWER "
            "distinct numeric values than you did last round. "
            "previous_number_count is how many key_numbers you returned last "
            "round (not cumulative) — your new list must be STRICTLY SHORTER "
            "than that. original_number_count is the total in the REAL "
            "source article (round 0) — for context only. Keep only the most "
            "prominent numbers. Write numbers in digits, avoid duplicates."
        ),
        payload_fields_used=("current_article", "original_number_count",
                             "previous_number_count"),
    ),

    # =========================================================================
    # SSE — Salient Sentence Editor
    # Operando: current_sentence (= sentence in the current article that
    #   represents the agent's last modification, cosine-anchored to KS_0).
    # Reference (ground truth): original_sentence (= the round-0 REAL sentence
    #   matched via cosine to current_sentence). This is what the metric
    #   `SSE_semantic_drift` / `SSE_lexical_stability` use in anchored mode
    #   (metrics_runner.py:856-916).
    # =========================================================================

    ("SSE", "Semantic Drift", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_sentence so the new modified_sentence is "
            "semantically FARTHER from original_sentence. "
            "original_sentence is the matching sentence in the REAL article "
            "(round 0) — your output must express a meaning that is MORE "
            "different from original_sentence than current_sentence is. "
            "Note: current_sentence is your previous modification (or the "
            "sentence as it stands now). "
            "Keep the result coherent and plausible in journalistic context."
        ),
        payload_fields_used=("current_sentence", "original_sentence"),
    ),

    ("SSE", "Semantic Drift", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_sentence so the new modified_sentence is "
            "semantically CLOSER to original_sentence. "
            "original_sentence is the matching sentence in the REAL article "
            "(round 0) — your output must express a meaning that is LESS "
            "different from original_sentence than current_sentence is. "
            "Note: current_sentence is your previous modification (or the "
            "sentence as it stands now). "
            "Keep the result coherent and plausible."
        ),
        payload_fields_used=("current_sentence", "original_sentence"),
    ),

    ("SSE", "Lexical Stability", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_sentence so the new modified_sentence shares MORE "
            "words and surface structure with original_sentence. "
            "original_sentence is the matching sentence in the REAL article "
            "(round 0) — your output must have HIGHER lexical overlap "
            "(shared tokens, n-grams) with original_sentence than "
            "current_sentence has. "
            "Note: current_sentence is your previous modification. "
            "Keep the result coherent and plausible."
        ),
        payload_fields_used=("current_sentence", "original_sentence"),
    ),

    ("SSE", "Lexical Stability", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_sentence so the new modified_sentence shares FEWER "
            "words and surface structure with original_sentence. "
            "original_sentence is the matching sentence in the REAL article "
            "(round 0) — your output must have LOWER lexical overlap "
            "(use synonyms, rephrase, restructure) with original_sentence than "
            "current_sentence has. "
            "Note: current_sentence is your previous modification. "
            "Keep the result coherent and plausible."
        ),
        payload_fields_used=("current_sentence", "original_sentence"),
    ),

    ("SSE", "Fluency (Perplexity)", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_sentence so the new modified_sentence reads in a "
            "stylistically MORE complex and less predictable way. "
            "Use less common phrasing, longer or unusual syntactic structures, "
            "richer vocabulary. "
            "original_sentence is the matching sentence in the REAL article "
            "(round 0) for context — but here your goal is style, not "
            "similarity. "
            "Keep the sentence grammatically correct and coherent."
        ),
        payload_fields_used=("current_sentence", "original_sentence"),
    ),

    ("SSE", "Fluency (Perplexity)", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_sentence so the new modified_sentence reads MORE "
            "naturally and fluently. "
            "Use simpler, more common phrasing and standard syntactic "
            "structures. "
            "original_sentence is the matching sentence in the REAL article "
            "(round 0) for context — but here your goal is fluency, not "
            "similarity. "
            "Keep the sentence grammatically correct and coherent."
        ),
        payload_fields_used=("current_sentence", "original_sentence"),
    ),

    # =========================================================================
    # NM — Number Modifier
    # Operando: current_sentence (= sentence in the current article containing
    #   current_number, cosine-anchored to the round-0 source sentence).
    # target_number: the ORIGINAL numeric value extracted by SA at round 0
    #   (= the round-0 anchor — never changes across rounds for a given target).
    # current_number: the numeric token actually in current_sentence right now
    #   (= lineage[-1] = most recent NM modification, or = target_number on the
    #   first NM call). The agent edits current_number → modified_number.
    #
    # Metric NM_change_magnitude = mean( edit_distance(target_number,
    #   modified_number) ) — cumulative distance from the ORIGINAL round-0 value.
    # Metric NM_edit_success_rate = (# KN_0 modified successfully this round) /
    #   |KN_0| — denominator fixed at round-0 count.
    # =========================================================================

    ("NM", "Edit Success Rate", "INCREASE"): ObjectiveSpec(
        text=(
            "You are being invoked ONCE for ONE target_number. The pipeline "
            "calls NM separately for each round-0 key_number; the aggregate "
            "metric, edit_success_rate, is the fraction of those calls that "
            "produce a real edit (denominator = |KN_0|). "
            "target_number is the ORIGINAL round-0 value — the immutable "
            "identity of this target across rounds. "
            "current_number is the value actually present in current_sentence "
            "right now (= your most recent edit on this target, or "
            "= target_number on the first NM call). You edit current_number "
            "INSIDE current_sentence. "
            "Edit_success_rate INCREASE means: this call MUST contribute a "
            "real edit. Magnitude does not matter — even a small delta counts "
            "(e.g. current_number=100 → 101). The only mandatory condition "
            "is that modified_sentence contains a value DIFFERENT from "
            "target_number (the round-0 anchor). "
            "If current_number already equals a previous no-op of yours, "
            "produce a different value this time. "
            "Keep the result journalistically plausible."
        ),
        payload_fields_used=("current_sentence", "target_number",
                             "current_number"),
    ),

    ("NM", "Edit Success Rate", "DECREASE"): ObjectiveSpec(
        text=(
            "You are being invoked ONCE for ONE target_number. The pipeline "
            "calls NM separately for each round-0 key_number; the aggregate "
            "metric, edit_success_rate, is the fraction of those calls that "
            "produce a real edit (denominator = |KN_0|). "
            "target_number is the ORIGINAL round-0 value — the immutable "
            "identity of this target. current_number is the value present in "
            "current_sentence right now (= your most recent edit on this "
            "target). You edit current_number INSIDE current_sentence. "
            "Edit_success_rate DECREASE means: this call must NOT contribute "
            "a real edit. Write a modified_sentence whose numeric value is "
            "EQUAL or nearly identical to target_number (the round-0 anchor), "
            "so the row counts as a no-op (e.g. target_number=100, "
            "current_number=120 → 100, or 100.0). "
            "Keep the result journalistically plausible."
        ),
        payload_fields_used=("current_sentence", "target_number",
                             "current_number"),
    ),

    ("NM", "Change Magnitude", "INCREASE"): ObjectiveSpec(
        text=(
            "You are being invoked ONCE for ONE target_number. The pipeline "
            "calls NM separately for each round-0 key_number; the aggregate "
            "metric, change_magnitude, is the AVERAGE edit-distance "
            "(target_number → modified_number) across those calls. "
            "target_number is the ORIGINAL round-0 value — the immutable "
            "identity of this target. current_number is the value present in "
            "current_sentence right now (= your most recent edit on this "
            "target, may already differ from target_number). You edit "
            "current_number INSIDE current_sentence. "
            "Change_magnitude INCREASE means: your modified_number must be "
            "VERY DISTANT from target_number (the round-0 anchor) in numeric "
            "magnitude (e.g. target_number=100 → 500, 50 → 5000), and "
            "FARTHER from target_number than current_number currently is. "
            "Keep the result journalistically plausible (no obvious nonsense)."
        ),
        payload_fields_used=("current_sentence", "target_number",
                             "current_number"),
    ),

    ("NM", "Change Magnitude", "DECREASE"): ObjectiveSpec(
        text=(
            "You are being invoked ONCE for ONE target_number. The pipeline "
            "calls NM separately for each round-0 key_number; the aggregate "
            "metric, change_magnitude, is the AVERAGE edit-distance "
            "(target_number → modified_number) across those calls. "
            "target_number is the ORIGINAL round-0 value — the immutable "
            "identity of this target. current_number is the value present in "
            "current_sentence right now (= your most recent edit). You edit "
            "current_number INSIDE current_sentence. "
            "Change_magnitude DECREASE means: your modified_number must be "
            "VERY CLOSE to target_number (the round-0 anchor) in numeric "
            "magnitude (e.g. target_number=100, current_number=150 → 101, "
            "50 → 51), and CLOSER to target_number than current_number "
            "currently is. Keep the result journalistically plausible."
        ),
        payload_fields_used=("current_sentence", "target_number",
                             "current_number"),
    ),

    # =========================================================================
    # NaM — Narrative Modifier
    # Operando: current_article.
    # Reference: original_article (REAL round 0) for rewrite_distance;
    #   reference_key_sentences (KS_0) for contradiction_prob.
    # previous_article = NaM r-1 (the agent's own last move).
    # technique selection is FREE — the agent must report `technique_used` in
    # the JSON output but is not constrained by a hint.
    # =========================================================================

    ("NaM", "Global Rewrite Distance", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article so the new modified_article is FARTHER "
            "from original_article than current_article is. "
            "original_article is the REAL source article (round 0). "
            "current_article is the article as it stands now — it is also "
            "your previous NaM rewrite (NaM pivot starts from its own r-1 "
            "output). Your modified_article must move further away from "
            "original_article than current_article already does. "
            "Vary vocabulary, restructure sentences, reorder paragraphs. "
            "Do not change the core false claims already introduced. "
            "Keep the article fluent and superficially plausible. "
            "Choose freely any technique from the 9 listed in your system "
            "prompt and report the choice in technique_used."
        ),
        payload_fields_used=("current_article", "original_article"),
    ),

    ("NaM", "Global Rewrite Distance", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article so the new modified_article is CLOSER "
            "to original_article than current_article is. "
            "original_article is the REAL source article (round 0). "
            "current_article is the article as it stands now — it is also "
            "your previous NaM rewrite. Your modified_article must move "
            "back toward original_article more than current_article does. "
            "Reduce vocabulary variation, restore original sentence "
            "structures where possible. "
            "Keep the article coherent and plausible. "
            "Choose freely any technique from the 9 listed in your system "
            "prompt and report the choice in technique_used."
        ),
        payload_fields_used=("current_article", "original_article"),
    ),

    ("NaM", "Contradiction Probability", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article so the new modified_article contradicts "
            "reference_key_sentences MORE than current_article does. "
            "reference_key_sentences are the load-bearing claims of the REAL "
            "source article (round 0). current_article is the article as it "
            "stands now — it is also your previous NaM rewrite. "
            "Exaggerate, minimise, or invert claims so the new article "
            "clashes more sharply with reference_key_sentences without "
            "being obviously false. "
            "Keep the article superficially plausible. "
            "Choose freely any technique from the 9 listed in your system "
            "prompt and report the choice in technique_used."
        ),
        payload_fields_used=("current_article", "reference_key_sentences"),
    ),

    ("NaM", "Contradiction Probability", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article so the new modified_article contradicts "
            "reference_key_sentences LESS than current_article does. "
            "reference_key_sentences are the load-bearing claims of the REAL "
            "source article (round 0). current_article is the article as it "
            "stands now — it is also your previous NaM rewrite. "
            "Soften or remove statements that clash with reference_key_sentences; "
            "prefer neutral, factual phrasing where contradictions exist. "
            "Keep the article coherent and plausible. "
            "Choose freely any technique from the 9 listed in your system "
            "prompt and report the choice in technique_used."
        ),
        payload_fields_used=("current_article", "reference_key_sentences"),
    ),

    # =========================================================================
    # TE — Title Editor
    # Operando: current_title.
    # Reference: current_article (the article body — NLI premise).
    # NO round-0 anchor and NO previous-round reference: the metric
    # `TE_title_entailment` / `TE_title_contradiction_prob` is a pure NLI
    # between (article_r, title_r) of the SAME round — neither round 0 nor
    # r-1 ever enters the calculation.
    # current_title IS effectively the agent's last move (TE r-1 output, or
    # the unmodified title if TE was not pivot at r-1).
    # =========================================================================

    ("TE", "Title Entailment", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title so it is MORE directly supported by "
            "current_article (= higher entailment probability). "
            "current_title is the title as it stands now (your previous "
            "modification if TE was pivot before, otherwise the original "
            "title). Your new title must align more clearly with what "
            "current_article actually says. "
            "Keep it concise and headline-style."
        ),
        payload_fields_used=("current_title", "current_article"),
    ),

    ("TE", "Title Entailment", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title so it is LESS directly supported by "
            "current_article (= lower entailment probability). "
            "current_title is the title as it stands now (your previous "
            "modification if TE was pivot before, otherwise the original "
            "title). Introduce ambiguity or a slight mismatch with the "
            "content of current_article. "
            "Keep it plausible as a headline."
        ),
        payload_fields_used=("current_title", "current_article"),
    ),

    ("TE", "Title Contradiction Probability", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title so it is MORE contradictory with the "
            "content of current_article (= higher contradiction probability). "
            "current_title is the title as it stands now. Create a clear "
            "mismatch between what the title says and what current_article "
            "actually claims, while still looking like a plausible headline."
        ),
        payload_fields_used=("current_title", "current_article"),
    ),

    ("TE", "Title Contradiction Probability", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title so it is LESS contradictory with the "
            "content of current_article (= lower contradiction probability). "
            "current_title is the title as it stands now. Improve the "
            "consistency between the title and what current_article actually "
            "claims; keep a natural headline style."
        ),
        payload_fields_used=("current_title", "current_article"),
    ),

    # =========================================================================
    # TE — Style vector (TITLE). Operando: current_title. original_title is the
    # REAL source article's title (round 0), shown as the natural-style exemplar.
    # For Title Case / All-Caps / Exclamation-Question a deterministic actuator
    # in the pipeline enforces the requested move after the LLM edit; the text
    # below still instructs the LLM so its semantic phrasing stays coherent.
    # =========================================================================
    ("TE", "Title Case Ratio", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title in Title Case: capitalize the first letter of "
            "every significant word, like a stylized headline. Keep the wording "
            "and meaning identical to current_title — change only capitalization."
        ),
        payload_fields_used=("current_title",),
    ),
    ("TE", "Title Case Ratio", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title in sentence case: capitalize only the first "
            "word and genuine proper nouns (people, places, organizations), "
            "exactly like a standard newswire headline. original_title shows the "
            "source article's natural capitalization. Keep the wording and "
            "meaning identical — change only capitalization."
        ),
        payload_fields_used=("current_title", "original_title"),
    ),
    ("TE", "All-Caps Rate", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title introducing more emphatic ALL-CAPS words for "
            "emphasis. Keep it a plausible headline and do not change the meaning."
        ),
        payload_fields_used=("current_title",),
    ),
    ("TE", "All-Caps Rate", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title removing ALL-CAPS emphasis: render any fully "
            "capitalized words in normal mixed case, like standard newswire "
            "style (see original_title). Keep wording and meaning identical."
        ),
        payload_fields_used=("current_title", "original_title"),
    ),
    ("TE", "Exclamation/Question Rate", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title adding exclamation or question marks to make "
            "it more sensational. Keep it a plausible headline and do not change "
            "the factual content."
        ),
        payload_fields_used=("current_title",),
    ),
    ("TE", "Exclamation/Question Rate", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title removing exclamation and question marks, using "
            "a flat declarative newswire style like original_title. Keep wording "
            "and meaning identical."
        ),
        payload_fields_used=("current_title", "original_title"),
    ),
    ("TE", "Title Length", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title making it longer (more words) while staying a "
            "natural, plausible headline that matches current_article."
        ),
        payload_fields_used=("current_title", "current_article"),
    ),
    ("TE", "Title Length", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title making it shorter and more concise (fewer "
            "words), in the terse style of original_title, while still matching "
            "current_article."
        ),
        payload_fields_used=("current_title", "original_title", "current_article"),
    ),
    ("TE", "Proper-Noun Rate", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title so it names more specific people, places, or "
            "organizations drawn from current_article (more proper nouns), while "
            "staying a plausible headline."
        ),
        payload_fields_used=("current_title", "current_article"),
    ),
    ("TE", "Proper-Noun Rate", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_title reducing the density of named people, places, "
            "and organizations — prefer common nouns and general phrasing, like "
            "the style of original_title — while still matching current_article."
        ),
        payload_fields_used=("current_title", "original_title", "current_article"),
    ),

    # =========================================================================
    # NaM — Style vector (BODY). Operando: current_article. original_article is
    # the REAL source article (round 0), shown as the natural-style exemplar.
    # All edits must preserve the facts, claims, and overall meaning.
    # =========================================================================
    ("NaM", "Reading Grade Level", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article to raise its reading difficulty toward the "
            "style of original_article: use longer sentences and more complex, "
            "multi-syllable vocabulary. Preserve the facts, claims, and overall "
            "meaning of current_article."
        ),
        payload_fields_used=("current_article", "original_article"),
    ),
    ("NaM", "Reading Grade Level", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article to lower its reading difficulty: use shorter "
            "sentences and simpler words. Preserve the facts, claims, and overall "
            "meaning of current_article."
        ),
        payload_fields_used=("current_article", "original_article"),
    ),
    ("NaM", "Lexical Diversity", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article using a richer, more varied vocabulary "
            "(avoid repeating the same words), toward the lexical variety of "
            "original_article. Preserve the facts and meaning of current_article."
        ),
        payload_fields_used=("current_article", "original_article"),
    ),
    ("NaM", "Lexical Diversity", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article using a simpler, more repetitive vocabulary "
            "with fewer distinct words. Preserve the facts and meaning of "
            "current_article."
        ),
        payload_fields_used=("current_article", "original_article"),
    ),
    ("NaM", "Quotation Rate", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article incorporating more direct quotations (quoted "
            "statements attributed to sources), as a sourced news report would, "
            "in the style of original_article. Preserve the facts and meaning and "
            "do not invent implausible sources."
        ),
        payload_fields_used=("current_article", "original_article"),
    ),
    ("NaM", "Quotation Rate", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article reducing direct quotations, paraphrasing "
            "quoted statements into reported speech. Preserve the facts and "
            "meaning of current_article."
        ),
        payload_fields_used=("current_article", "original_article"),
    ),
    ("NaM", "Punctuation Rate", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article using richer punctuation (commas, "
            "semicolons, dashes, quotation marks) to structure the prose, toward "
            "the style of original_article. Preserve the facts and meaning."
        ),
        payload_fields_used=("current_article", "original_article"),
    ),
    ("NaM", "Punctuation Rate", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article using sparser, simpler punctuation and "
            "plainer sentence structure. Preserve the facts and meaning of "
            "current_article."
        ),
        payload_fields_used=("current_article", "original_article"),
    ),
    ("NaM", "Adjective/Adverb Density", "INCREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article using more descriptive adjectives and "
            "adverbs, making the prose more colorful and emphatic. Preserve the "
            "facts and meaning of current_article."
        ),
        payload_fields_used=("current_article", "original_article"),
    ),
    ("NaM", "Adjective/Adverb Density", "DECREASE"): ObjectiveSpec(
        text=(
            "Rewrite current_article reducing adjectives and adverbs to a sober, "
            "factual newswire tone like original_article — prefer plain nouns and "
            "verbs. Preserve the facts and meaning of current_article."
        ),
        payload_fields_used=("current_article", "original_article"),
    ),
}


# ---------------------------------------------------------------------------
# Backward-compatible helper used by orchestrator_iterative.py
# ---------------------------------------------------------------------------
def get_objective_text(agent: str, metric: str, direction: str) -> str:
    """Return the objective instruction string, or empty string if not found."""
    spec = OBJECTIVES.get((agent, metric, direction))
    return spec.text if spec is not None else ""
