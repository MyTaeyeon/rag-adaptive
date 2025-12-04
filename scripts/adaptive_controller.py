"""
adaptive_controller.py
Simple adaptive controller to pick k based on query features.
Two modes:
 - heuristic (noun count + length)
 - llm_entropy (call an LLM to estimate uncertainty) -- template (requires adapter to your LLM provider)

Outputs:
    dict with keys: { 'mode', 'k', 'reason', 'features' }
"""
import math
import re
from typing import Dict
import spacy

# load spaCy small model (user must install 'python -m spacy download en_core_web_sm')
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

def count_nouns(query: str) -> int:
    if not nlp:
        # fallback heuristic: count words starting with uppercase or common noun endings
        return sum(1 for t in re.findall(r"\w+", query) if len(t) > 3)
    doc = nlp(query)
    return sum(1 for tok in doc if tok.pos_ in ("NOUN","PROPN"))

def heuristic_k(query: str, base_k=5, max_k=100) -> Dict:
    length = len(query.split())
    noun_count = count_nouns(query)
    # simple rules
    if length <= 8 and noun_count <= 2:
        k = 3
        reason = "short & few nouns"
    elif length <= 20 and noun_count <= 5:
        k = base_k
        reason = "moderate length"
    else:
        # longer or many nouns -> more contexts
        k = min(max_k, base_k + (length // 2) + noun_count)
        reason = "long/many nouns heuristics"
    return {"mode":"heuristic", "k":int(k), "reason":reason, "features":{"length":length,"noun_count":noun_count}}

def llm_entropy_k(query: str, llm_client, base_k=5, max_k=200):
    """
    Template: use your LLM client to estimate uncertainty/entropy.
    Example approaches:
      - Ask the LLM to score its confidence [0-1]
      - Use sampling (nucleus/temperature) and compute distributional entropy
    llm_client must expose a method `estimate_uncertainty(query)` returning a float 0..1.
    """
    try:
        score = llm_client.estimate_uncertainty(query)  # 0..1 where 1 = high uncertainty
    except Exception as e:
        return {"mode":"llm_entropy", "k": base_k, "reason": f"llm failure: {e}", "features":{}}
    # map uncertainty to k
    k = int(base_k + score * (max_k - base_k))
    return {"mode":"llm_entropy", "k":k, "reason":"mapped from llm uncertainty", "features":{"uncertainty":score}}

# quick test
if __name__ == "__main__":
    samples = [
        "Who is the president of France?",
        "Explain the proof of the Prime Number Theorem using complex analysis and the Riemann zeta function — highlight key steps and references."
    ]
    for q in samples:
        print(q)
        print(heuristic_k(q))
