# scripts/normalize_and_split.py
import argparse
import re
import unicodedata
import json
from pathlib import Path
from html import unescape
import yaml
from collections import Counter, defaultdict
import math

# === helpers ===
def nfkc(t): return unicodedata.normalize("NFKC", t)

def remove_html(t):
    t = re.sub(r"<[^>]+>", " ", t)
    return unescape(t)

def strip_ws(t): return re.sub(r"\s+", " ", t).strip()

SENT_END_RE = re.compile(r'(?<=[.!?…])\s+')

def sentence_split(text):
    """Split into sentences by punctuation heuristics defined above."""
    if not text:
        return []
    text = strip_ws(text)
    if len(text) < 40:
        return [text]
    parts = SENT_END_RE.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def normalize(text, cfg):
    if text is None:
        return ""
    t = str(text)
    if cfg.get("nfkc", False):
        t = nfkc(t)
    if cfg.get("remove_html", False):
        t = remove_html(t)
    if cfg.get("strip_extra_whitespace", False):
        t = strip_ws(t)
    if cfg.get("lowercase", False):
        t = t.lower()
    return t

def percentiles(sorted_list, ps):
    """Return list of percentiles for given sorted_list and percent values (0-100)."""
    n = len(sorted_list)
    if n == 0:
        return [None] * len(ps)
    out = []
    for p in ps:
        if p <= 0:
            out.append(sorted_list[0])
            continue
        if p >= 100:
            out.append(sorted_list[-1])
            continue
        # linear interpolation (nearest rank)
        rank = (p / 100.0) * (n - 1)
        lo = math.floor(rank)
        hi = math.ceil(rank)
        if lo == hi:
            out.append(sorted_list[int(rank)])
        else:
            w = rank - lo
            out.append(sorted_list[lo] * (1 - w) + sorted_list[hi] * w)
    return out

# === main ===
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf8"))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Stats containers
    stats = {
        "n_records": 0,
        "n_invalid_json": 0,
        "n_missing_context": 0,
        "n_missing_question": 0,
        "n_with_answers": 0,
        "total_sentences": 0,
        "avg_sentences_per_context": 0,
        "sentence_char_lengths": [],   # character lengths per sentence
        "sentence_word_lengths": [],   # word counts per sentence
        "context_char_lengths": [],    # context total char length
        "context_word_lengths": [],    # context total word count
        "question_char_lengths": [],
        "question_word_lengths": [],
        "answers_count_hist": Counter(),  # histogram of number of answers per record
        "dataset_counts": Counter(),
    }

    def words_count(s):
        if not s: return 0
        return len(s.split())

    out_file = out / "all.jsonl"
    # use with to ensure closed
    with open(args.input, "r", encoding="utf8") as fi, \
         open(out_file, "w", encoding="utf8") as fo:

        for lineno, line in enumerate(fi, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                stats["n_invalid_json"] += 1
                # optionally print/log error
                print(f"Warning: invalid json at line {lineno}: {e}")
                continue

            stats["n_records"] += 1

            raw_q = rec.get("question")
            raw_ctx = rec.get("context")
            answers = rec.get("answers", [])

            if raw_ctx is None or str(raw_ctx).strip() == "":
                stats["n_missing_context"] += 1
            if raw_q is None or str(raw_q).strip() == "":
                stats["n_missing_question"] += 1

            q = normalize(raw_q, cfg)
            ctx = normalize(raw_ctx, cfg)

            # choose sentence splitting only when enabled; otherwise empty list
            if cfg.get("sentence_split", False):
                ctx_sents = sentence_split(ctx)
            else:
                # treat whole context as single "sentence" if context exists,
                # so stats still capture context lengths.
                ctx_sents = [ctx] if ctx else []

            # update sentence stats
            n_sents = len(ctx_sents)
            stats["total_sentences"] += n_sents

            for s in ctx_sents:
                # normalized sentence string s
                s_str = s or ""
                char_len = len(s_str)
                word_len = words_count(s_str)
                stats["sentence_char_lengths"].append(char_len)
                stats["sentence_word_lengths"].append(word_len)

            # context totals
            ctx_char_len = len(ctx)
            ctx_word_len = words_count(ctx)
            stats["context_char_lengths"].append(ctx_char_len)
            stats["context_word_lengths"].append(ctx_word_len)

            # question totals
            q_char_len = len(q)
            q_word_len = words_count(q)
            stats["question_char_lengths"].append(q_char_len)
            stats["question_word_lengths"].append(q_word_len)

            # answers
            n_answers = len(answers) if answers is not None else 0
            stats["answers_count_hist"][n_answers] += 1
            if n_answers > 0:
                stats["n_with_answers"] += 1

            # dataset
            ds = rec.get("dataset") or "unknown"
            stats["dataset_counts"][ds] += 1

            out_rec = {
                "id": rec.get("id"),
                "question": raw_q,
                "question_norm": q,
                "context": raw_ctx,
                "context_norm": ctx,
                "context_sentences": ctx_sents,
                "answers": answers,
                "dataset": ds
            }

            fo.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    # finalize aggregated stats
    stats["avg_sentences_per_context"] = (
        stats["total_sentences"] / stats["n_records"]
        if stats["n_records"] > 0 else 0
    )

    def make_length_summary(lst):
        if not lst:
            return {
                "n": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "p10": None,
                "p25": None,
                "p75": None,
                "p90": None
            }
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        mean = sum(sorted_lst) / n
        med = percentiles(sorted_lst, [50])[0]
        p10, p25, p75, p90 = percentiles(sorted_lst, [10,25,75,90])
        return {
            "n": n,
            "min": sorted_lst[0],
            "max": sorted_lst[-1],
            "mean": mean,
            "median": med,
            "p10": p10,
            "p25": p25,
            "p75": p75,
            "p90": p90
        }

    summary = {
        "n_records": stats["n_records"],
        "n_invalid_json": stats["n_invalid_json"],
        "n_missing_context": stats["n_missing_context"],
        "n_missing_question": stats["n_missing_question"],
        "n_with_answers": stats["n_with_answers"],
        "total_sentences": stats["total_sentences"],
        "avg_sentences_per_context": stats["avg_sentences_per_context"],
        "sentence_char_lengths": make_length_summary(stats["sentence_char_lengths"]),
        "sentence_word_lengths": make_length_summary(stats["sentence_word_lengths"]),
        "context_char_lengths": make_length_summary(stats["context_char_lengths"]),
        "context_word_lengths": make_length_summary(stats["context_word_lengths"]),
        "question_char_lengths": make_length_summary(stats["question_char_lengths"]),
        "question_word_lengths": make_length_summary(stats["question_word_lengths"]),
        "answers_count_hist": dict(stats["answers_count_hist"]),
        "dataset_counts": dict(stats["dataset_counts"]),
    }

    # write stats file
    with open(out / "stats.json", "w", encoding="utf8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print(f"Done. Records: {summary['n_records']}")
    print(f"Stats written to: {out/'stats.json'}")

if __name__ == "__main__":
    main()
