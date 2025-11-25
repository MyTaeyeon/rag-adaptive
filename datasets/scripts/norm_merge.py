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
from typing import Iterable, Union

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

# === file reading utilities ===
def iter_jsonl(path: Union[str, Path]) -> Iterable[dict]:
    """Yield JSON objects from a .jsonl file (one JSON object per line)."""
    path = Path(path)
    with open(path, "r", encoding="utf8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                # caller handles counting invalid lines
                raise ValueError(f"Invalid JSON in {path} at line {lineno}: {e}")

def iter_json_array(path: Union[str, Path]) -> Iterable[dict]:
    """Yield JSON objects if file contains a JSON array."""
    path = Path(path)
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                yield obj
        else:
            raise ValueError(f"File {path} is not a JSON array.")

def gather_input_files(paths: Iterable[str]) -> list:
    """Given input paths, expand directories and return list of files to process."""
    files = []
    for p in paths:
        pth = Path(p)
        if not pth.exists():
            print(f"Warning: input path {p} does not exist, skipping.")
            continue
        if pth.is_dir():
            # collect jsonl / json files
            for ext in ("*.jsonl", "*.json"):
                for f in sorted(pth.glob(ext)):
                    files.append(f)
        else:
            files.append(pth)
    return files

# === main ===
def main():
    p = argparse.ArgumentParser(description="Normalize, sentence-split and merge multiple json/jsonl QA files into a single jsonl + stats.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--inputs", nargs="+", help="One or more input files or directories (use directories to include all .jsonl/.json inside).")
    group.add_argument("--input", help="(deprecated) single input file for backward compatibility.")
    p.add_argument("--out_dir", required=True, help="Output directory for all.jsonl and stats.json")
    p.add_argument("--config", required=True, help="YAML config for normalization (nfkc, lowercase, remove_html, sentence_split, strip_extra_whitespace)")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf8"))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # decide inputs
    if args.inputs:
        input_paths = args.inputs
    else:
        input_paths = [args.input]

    files = gather_input_files(input_paths)
    if not files:
        raise SystemExit("No input files found. Provide --inputs paths or a directory containing .jsonl/.json files.")

    # Stats containers
    stats = {
        "n_records": 0,
        "n_invalid_json": 0,
        "n_invalid_files": 0,
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
        "source_files": Counter(),  # count per input file
    }

    def words_count(s):
        if not s: return 0
        return len(s.split())

    out_file = out / "all.jsonl"
    # open output once and stream
    with open(out_file, "w", encoding="utf8") as fo:
        for file_path in files:
            file_path = Path(file_path)
            processed_records_in_file = 0
            try:
                # choose iterator based on extension / content
                if file_path.suffix.lower() == ".jsonl":
                    iterator = iter_jsonl(file_path)
                elif file_path.suffix.lower() == ".json":
                    # attempt to detect if JSON array or newline JSON objects
                    # try to load first non-empty char
                    with open(file_path, "r", encoding="utf8") as tf:
                        start = tf.read(1)
                        tf.seek(0)
                        if start == "[":
                            iterator = iter_json_array(file_path)
                        else:
                            # fallback to line-based
                            iterator = iter_jsonl(file_path)
                else:
                    # try as jsonl
                    iterator = iter_jsonl(file_path)

                for rec in iterator:
                    stats["n_records"] += 1
                    processed_records_in_file += 1

                    raw_q = rec.get("question")
                    raw_ctx = rec.get("context")
                    answers = rec.get("answers", [])

                    if raw_ctx is None or str(raw_ctx).strip() == "":
                        stats["n_missing_context"] += 1
                    if raw_q is None or str(raw_q).strip() == "":
                        stats["n_missing_question"] += 1

                    q = normalize(raw_q, cfg)
                    ctx = normalize(raw_ctx, cfg)

                    # choose sentence splitting only when enabled; otherwise treat whole context as single
                    if cfg.get("sentence_split", False):
                        ctx_sents = sentence_split(ctx)
                    else:
                        ctx_sents = [ctx] if ctx else []

                    # update sentence stats
                    n_sents = len(ctx_sents)
                    stats["total_sentences"] += n_sents

                    for s in ctx_sents:
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

                    # dataset (try to detect from record first; else from filename)
                    ds = rec.get("dataset") or file_path.stem or "unknown"
                    stats["dataset_counts"][ds] += 1

                    stats["source_files"][str(file_path)] += 1

                    out_rec = {
                        "id": rec.get("id"),
                        "question": raw_q,
                        "question_norm": q,
                        "context": raw_ctx,
                        "context_norm": ctx,
                        "context_sentences": ctx_sents,
                        "answers": answers,
                        "dataset": ds,
                        "source_file": str(file_path)
                    }

                    fo.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            except ValueError as e:
                stats["n_invalid_json"] += 1
                stats["n_invalid_files"] += 1
                print(f"Warning: skipping file {file_path} due to JSON error: {e}")
                continue
            except Exception as e:
                stats["n_invalid_files"] += 1
                print(f"Warning: error processing file {file_path}: {e}")
                continue
            finally:
                # small log per file
                print(f"Processed {processed_records_in_file} records from {file_path}")

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
        "n_invalid_files": stats["n_invalid_files"],
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
        "source_files": dict(stats["source_files"]),
    }

    # write stats file
    with open(out / "stats.json", "w", encoding="utf8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print(f"Done. Records: {summary['n_records']}")
    print(f"Stats written to: {out/'stats.json'}")

if __name__ == "__main__":
    main()
