"""Cases where a team's PRIVATE memory (analysis + lessons + guidance to its agents) says one thing and its PUBLIC post
(collective note / board post) says the opposite. Four detectors, all reported:
  A  NLI (textattack/roberta-base-MNLI, already in hf_cache): contradiction between a private sentence and a public one
  B  same item (technique/feature) judged effective in one text and harmful in the other (regex polarity)
  C  verdict claims in the post that contradict the real verdict (and what the private text said about it)
  D  overall judgement of the team's strategy: private negative vs public positive (and the reverse)
Output: JSON + Markdown report. Usage: python private_vs_public.py OUT_MD ROOT:LABEL:s1,s2.. [ROOT:LABEL:..]"""
import collections, glob, json, re, sys, itertools

OUT_MD = sys.argv[1]
import os
if os.path.exists(OUT_MD) or os.path.exists(OUT_MD + ".counts.json"):
    sys.exit(f"RIFIUTO: {OUT_MD} esiste gia' (non sovrascrivo mai i risultati)")
SPECS = [a.split(":") for a in sys.argv[2:]]
ITEMS = {"Loaded Language": r"loaded language", "Causal Oversimplification": r"causal oversimpl", "Whataboutism": r"whatabout",
         "Obfuscation": r"obfuscat", "proper-noun (NNP)": r"proper[- ]noun|\bnnp\b", "title length": r"title length",
         "title-case": r"title[- ]?case", "key numbers": r"key[_ ]?numbers?", "NLI contradiction": r"contradiction|\bnli\b",
         "Flesch-Kincaid": r"flesch", "quote marks": r"quote[- ]mark", "perplexity": r"perplex", "ROUGE": r"rouge"}
POS = re.compile(r"(was|were|is|are|been|proved)\s+(very\s+|particularly\s+)?(effective|successful|crucial|helpful|beneficial)|effective in making|"
                 r"crucial in making|helped|worked well|can be effective|(toward|towards) REAL|more believable", re.I)
NEG = re.compile(r"not\s+(been\s+)?(effective|successful|helpful|sufficient)|ineffective|too (obvious|explicit|noticeable)|did not work|hurt|"
                 r"(toward|towards) FAKE|contributed to the FAKE", re.I)
V_REAL = re.compile(r"(classified|labell?ed|verdict|judged|flagged)[^.]{0,25}\breal\b|\breal\b[^.]{0,12}(verdict|classification|label)", re.I)
V_FAKE = re.compile(r"(classified|labell?ed|verdict|judged|flagged)[^.]{0,25}\bfake\b|\bfake\b[^.]{0,12}(verdict|classification|label)", re.I)
STRAT_NEG = re.compile(r"(strategy|approach|agents?|team)[^.]{0,60}(not (been )?(effective|successful|working|yielding|sufficient)|failed|did not (work|follow|fool|succeed)|needs? to be re-?evaluated)", re.I)
STRAT_POS = re.compile(r"(strategy|approach|efforts?)[^.]{0,50}(is|are|was|were)?\s*(effective|working|paying off|successful)|(glad|pleased|happy)", re.I)
STOP = set("the a an and or of to in on for with is are was were be been it its this that as by at from not but we our you your they their them should need needs more less very also than which will can would could may might must have has had do does did so if into about over such these those there here all any each other some only just make made making team agent agents article".split())


def sents(t):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", str(t)) if len(x.strip()) > 25]


def cw(t):
    return set(w for w in re.findall(r"[a-z][a-z'\-]+", t.lower()) if w not in STOP and len(w) > 2)


def pol(text):
    out = collections.defaultdict(set)
    for s in sents(text):
        p, n = bool(POS.search(s)), bool(NEG.search(s))
        if p == n:
            continue
        for k, rx in ITEMS.items():
            if re.search(rx, s, re.I):
                out[k].add("+" if p else "-")
    return out


def events(root, s):
    for f in sorted(glob.glob(f"{root}/{s}/rounds/round_*/events.json")):
        for e in json.load(open(f)):
            yield e


cases = {"A": [], "B": [], "C": [], "D": []}
nli_pairs = []
n_posts = collections.Counter()
for root, lab, scs in SPECS:
    for s in scs.split(","):
        for e in events(root, s):
            d = e.get("coordinator") or e.get("decision") or {}
            post = (d.get("public_post") or d.get("collective_note") or "").strip()
            if not post:
                continue
            n_posts[lab] += 1
            priv_parts = [d.get("analysis", ""), d.get("lessons", ""), d.get("note", "")] + [str(v) for v in (d.get("guidance") or {}).values()]
            priv = " ".join(x for x in priv_parts if x)
            key = (lab, s, e["unit"], e["round"]); label = (e.get("feedback") or {}).get("label")
            # B
            pv, pb = pol(priv), pol(post)
            for k in pb:
                if k in pv and ((pv[k] == {"+"} and pb[k] == {"-"}) or (pv[k] == {"-"} and pb[k] == {"+"})):
                    cases["B"].append((key, k, "".join(pv[k]), "".join(pb[k]), priv, post, label))
            # C
            for sent in sents(post):
                if not re.search(r"\b(we|our|this round|first (round|attempt)|the team's)\b", sent, re.I):
                    continue
                sr, sf = bool(V_REAL.search(sent)), bool(V_FAKE.search(sent))
                if sr != sf and label in ("REAL", "FAKE") and not re.search(r"previous|earlier|last round|so far", sent, re.I):
                    if (sr and label == "FAKE") or (sf and label == "REAL"):
                        privsays = "REAL" if re.search(r"classified as REAL|REAL verdict|labeled REAL", priv) else ("FAKE" if re.search(r"classified as FAKE|FAKE verdict|labeled FAKE", priv) else "non dice")
                        cases["C"].append((key, "REAL" if sr else "FAKE", label, privsays, sent, priv, post))
            # D
            sn, sp = bool(STRAT_NEG.search(priv)), bool(STRAT_POS.search(post))
            pn = bool(re.search(r"not (been )?(effective|successful|working)|failed|ineffective|did not (work|succeed)|too (obvious|explicit)", post, re.I))
            pp = bool(STRAT_POS.search(priv))
            if sn and sp and not pn and not re.search(r"\bbut\b|however|concern|careful|need", post, re.I) and not pp:
                cases["D"].append((key, "privato NEGATIVO / pubblico POSITIVO", priv, post, label))
            if pp and not sn and pn and not sp:
                cases["D"].append((key, "privato POSITIVO / pubblico NEGATIVO", priv, post, label))
            # A pairs
            ps = list(dict.fromkeys(sents(priv))); bs = sents(post)
            for b in bs:
                cb = cw(b)
                for a in ps:
                    ca = cw(a)
                    shared_item = any(re.search(rx, a, re.I) and re.search(rx, b, re.I) for rx in ITEMS.values())
                    if shared_item or (cb and ca and len(cb & ca) / len(cb | ca) >= 0.22):
                        nli_pairs.append((key, a, b, label))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("textattack/roberta-base-MNLI"); mdl = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-MNLI").eval().to(dev)


def contra(prem, hyp):
    out = []
    for i in range(0, len(prem), 24):
        enc = tok(prem[i:i + 24], hyp[i:i + 24], return_tensors="pt", padding=True, truncation=True, max_length=200)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out += torch.softmax(mdl(**enc).logits, -1)[:, 0].tolist()
    return out


sanity = contra(["The team's article was classified as REAL.", "The cat is on the mat."], ["The team's article was classified as FAKE.", "The cat is not on the mat."])
print("sanity NLI (devono essere alte):", [round(x, 2) for x in sanity], flush=True)
print("coppie NLI da valutare:", len(nli_pairs), flush=True)
fw = contra([p[1] for p in nli_pairs], [p[2] for p in nli_pairs]); bw = contra([p[2] for p in nli_pairs], [p[1] for p in nli_pairs])
for p, f, b in zip(nli_pairs, fw, bw):
    pr = min(f, b)  # both directions must agree it is a contradiction
    if pr >= 0.85:
        cases["A"].append((p[0], round(pr, 3), p[1], p[2], p[3]))
cases["A"].sort(key=lambda c: -c[1])

with open(OUT_MD, "x", encoding="utf-8") as f:  # "x": refuses to overwrite an existing report
    f.write("# Privato vs pubblico: casi in cui dicono il contrario\n\n")
    f.write(f"Post analizzati: {dict(n_posts)}. Coppie NLI valutate: {len(nli_pairs)}.\n\n")
    for k, title in (("C", "C. Verdetto dichiarato nel post diverso da quello reale"), ("B", "B. Stessa voce giudicata efficace in un testo e dannosa nell'altro"),
                     ("D", "D. Giudizio complessivo sulla strategia opposto"), ("A", "A. Contraddizione NLI (entrambe le direzioni, prob >= 0.85)")):
        f.write(f"\n## {title} — {len(cases[k])} casi\n")
        for c in cases[k]:
            lab, s, u, r = c[0]
            f.write(f"\n### {lab} {s} {u} round {r}\n")
            if k == "A":
                f.write(f"- prob. contraddizione: {c[1]}  | verdetto reale del round: {c[4]}\n- PRIVATO: {c[2]}\n- PUBBLICO: {c[3]}\n")
            elif k == "B":
                f.write(f"- voce: {c[1]} | privato {c[2]} / pubblico {c[3]} | verdetto reale: {c[6]}\n- PRIVATO: {c[4][:900]}\n- PUBBLICO: {c[5][:600]}\n")
            elif k == "C":
                f.write(f"- il post dice {c[1]}, verdetto reale {c[2]}; il privato dice: {c[3]}\n- FRASE: {c[4]}\n- POST: {c[6][:500]}\n")
            else:
                f.write(f"- {c[1]} | verdetto reale: {c[4]}\n- PRIVATO: {c[2][:700]}\n- PUBBLICO: {c[3][:500]}\n")
json.dump({k: len(v) for k, v in cases.items()}, open(OUT_MD + ".counts.json", "x"))
print({k: len(v) for k, v in cases.items()}, flush=True)
print("scritto", OUT_MD, flush=True)
