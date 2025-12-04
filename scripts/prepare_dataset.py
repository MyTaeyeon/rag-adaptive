#!/usr/bin/env python3
# scripts/prepare_dataset.py
"""Prepare & merge datasets from multiple sources into canonical JSON schema:
  { "id": "...", "question": "...", "ground_truth": "...", "category": "easy|hard|distracting", "source": "hotpotqa" }

This version supports the file formats you provided:
 - naturalquestions.jsonl : each line JSON with keys `question` and `answer`
 - hotpot_train_v1.json : objects with keys like `question`, `answer`, `supporting_facts`, `level`, `context`
 - mmlu.csv : CSV where `choices` is a JSON list or delimiter-separated, `answer` is a letter like 'D'
 - triviaqa.json : flexible structure (tries several candidate fields such as `normalized_value`, `value`, or `wiki_context`)
 - msmarco.tsv : TSV with headers including `query`, `wellFormedAnswers` or `answers` (list-like), `passages`

Usage:
  python scripts/prepare_dataset.py --raw_dir raw_data --out samples/dataset_normalized.json --sample_size 100
"""
import os
import json
import argparse
import random
import csv
import re
from pathlib import Path
from collections import Counter

def safe_load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_text(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def try_parse_json_field(val):
    """Try to parse a string that contains JSON (list/dict). Return parsed value or original."""
    if val is None:
        return val
    if isinstance(val, (list, dict)):
        return val
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s:
        return s
    try:
        if (s[0] == "{" and s[-1] == "}") or (s[0] == "[" and s[-1] == "]"):
            return json.loads(s)
    except Exception:
        try:
            # last resort: eval with no builtins (may parse Python-style lists)
            return eval(s, {"__builtins__": {}})
        except Exception:
            return s
    return s

def parse_nq_jsonl(path):
    out = []
    i = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                parts = line.split("\t")
                if len(parts) >= 2:
                    q = normalize_text(parts[0])
                    a = normalize_text(parts[1])
                    out.append({"id": f"nq_{i}", "question": q, "ground_truth": a, "category": "easy", "source":"naturalquestions"})
                    i += 1
                    continue
                else:
                    continue
            q = normalize_text(obj.get("question") or obj.get("query") or "")
            a = obj.get("answer", "") or obj.get("answers", "") or obj.get("annotated_answer","")
            a = try_parse_json_field(a)
            if isinstance(a, list):
                a = " ; ".join([normalize_text(x) for x in a if x])
            a = normalize_text(a)
            out.append({"id": f"nq_{i}", "question": q, "ground_truth": a, "category": "easy", "source":"naturalquestions"})
            i += 1
    return out

def parse_hotpot(path):
    out = []
    j = None
    with open(path, "r", encoding="utf-8") as f:
        try:
            j = json.load(f)
        except Exception:
            f.seek(0)
            j = []
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    j.append(json.loads(ln))
                except Exception:
                    continue
    if not isinstance(j, list):
        j = [j]
    for i, it in enumerate(j):
        q = normalize_text(it.get("question", ""))
        gt = normalize_text(it.get("answer", ""))
        level = it.get("level") or it.get("difficulty") or ""
        sf = it.get("supporting_facts") or []
        if isinstance(sf, dict):
            sent_ids = sf.get("sent_id") or []
            num_support = len(sent_ids)
        elif isinstance(sf, list):
            num_support = len(sf)
        else:
            num_support = 0
        category = "hard" if (num_support >= 2 or (isinstance(level, str) and level.lower() in ("medium","hard","difficult","comparison","multi-hop"))) else "easy"
        docid = it.get("id") or it.get("_id") or f"{i:06d}"
        out.append({"id": f"hotpot_{docid}", "question": q, "ground_truth": gt, "category": category, "source":"hotpotqa"})
    return out

def parse_mmlu_csv(path):
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(1024)
        f.seek(0)
        delim = "," if sample.count(",") >= sample.count("\t") else "\t"
        reader = csv.DictReader(f, delimiter=delim)
        i = 0
        for row in reader:
            q = normalize_text(row.get("question") or row.get("prompt") or "")
            choices_raw = row.get("choices") or row.get("choice") or row.get("alternatives") or ""
            choices = try_parse_json_field(choices_raw)
            if isinstance(choices, str):
                if "||" in choices:
                    choices = [c.strip() for c in choices.split("||")]
                elif "|" in choices and not choices.strip().startswith("["):
                    choices = [c.strip() for c in choices.split("|")]
                elif ";" in choices:
                    choices = [c.strip() for c in choices.split(";")]
                else:
                    choices = [choices]
            ans_token = (row.get("answer") or row.get("label") or "").strip()
            gt = ""
            if ans_token and choices and isinstance(choices, (list,tuple)):
                at = ans_token.strip().upper()
                if at.isalpha():
                    idx = ord(at[0]) - ord('A')
                else:
                    try:
                        idx = int(at) - 1
                    except Exception:
                        idx = None
                if idx is not None and 0 <= idx < len(choices):
                    gt = normalize_text(choices[idx])
            if not gt:
                gt = normalize_text(row.get("answer_text") or row.get("answer") or row.get("label_text",""))
            out.append({"id": f"mmlu_{i}", "question": q, "ground_truth": gt, "category": "hard", "source":"mmlu"})
            i += 1
    return out

def parse_trivia(path):
    out = []
    try:
        j = safe_load_json(path)
    except Exception:
        j = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    j.append(json.loads(ln))
                except Exception:
                    continue
    items = j if isinstance(j, list) else (j.get("Data") or j.get("questions") or [])
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        q = normalize_text(it.get("question") or it.get("question_text") or it.get("question_id") or "")
        gt = it.get("normalized_value") or it.get("value") or it.get("wiki_context") or it.get("search_context") or ""
        if isinstance(gt, list):
            gt = "; ".join([normalize_text(x) for x in gt])
        gt = normalize_text(gt)
        out.append({"id": f"trivia_{i}", "question": q, "ground_truth": gt, "category": "easy", "source":"triviaqa"})
    return out

def parse_msmarco_tsv(path):
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(2048)
        f.seek(0)
        delim = "\t" if "\t" in sample else ","
        reader = csv.DictReader(f, delimiter=delim)
        i = 0
        if reader.fieldnames is None:
            f.seek(0)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    q = normalize_text(parts[0])
                    gt = normalize_text(parts[1])
                    out.append({"id": f"msmarco_{i}", "question": q, "ground_truth": gt, "category":"hard", "source":"ms_marco"})
                    i += 1
            return out
        for row in reader:
            q = normalize_text(row.get("query") or row.get("Query") or row.get("question") or "")
            ans_field = row.get("wellFormedAnswers") or row.get("wellFormedAnswer") or row.get("answers") or row.get("answer") or ""
            ans = try_parse_json_field(ans_field)
            if isinstance(ans, list):
                gt = "; ".join([normalize_text(x) for x in ans if x])
            else:
                gt = normalize_text(ans)
            if not gt:
                passages_field = row.get("passages") or row.get("passage") or ""
                passages = try_parse_json_field(passages_field)
                if isinstance(passages, list):
                    sel = None
                    for p in passages:
                        if isinstance(p, dict) and (p.get("is_selected") == 1 or p.get("is_selected") == "1"):
                            sel = p.get("passage_text") or p.get("text") or sel
                            break
                    if not sel and passages:
                        first = passages[0]
                        if isinstance(first, dict):
                            sel = first.get("passage_text") or first.get("text") or ""
                        else:
                            sel = str(first)
                    gt = normalize_text(sel)
            out.append({"id": f"msmarco_{i}", "question": q, "ground_truth": gt, "category":"hard", "source":"ms_marco"})
            i += 1
    return out

def is_distracting(q, gt):
    ql = len(q.split())
    if ql < 3 and not gt:
        return True
    patterns = [r"taste of the color", r"weather in number theory", r"become a quaternion"]
    for p in patterns:
        if re.search(p, q.lower()):
            return True
    return False

def dedupe(items):
    seen_q = {}
    out = []
    for it in items:
        q = it["question"].strip().lower()
        if not q:
            continue
        if q in seen_q:
            prev = seen_q[q]
            if prev.get("ground_truth") and not it.get("ground_truth"):
                continue
            if it.get("ground_truth") and not prev.get("ground_truth"):
                seen_q[q] = it
        else:
            seen_q[q] = it
    return list(seen_q.values())

def main(raw_dir, out_path, sample_size=100, seed=42):
    random.seed(seed)
    raw_dir = Path(raw_dir)
    aggregated = []
    mapping = {
        "hotpot": parse_hotpot,
        "natural": parse_nq_jsonl,
        "nq": parse_nq_jsonl,
        # "msmarco": parse_msmarco_tsv,
        # "ms_marco": parse_msmarco_tsv,
        "trivia": parse_trivia,
        "triviaqa": parse_trivia,
        "mmlu": parse_mmlu_csv
    }
    for f in sorted(raw_dir.iterdir()):
        if not f.is_file():
            continue
        lname = f.name.lower()
        for key, parser in mapping.items():
            if key in lname:
                try:
                    print(f"[INFO] Parsing {f.name} with {parser.__name__}")
                    parsed = parser(str(f))
                    print(f"[INFO] -> parsed {len(parsed)} items")
                    aggregated.extend(parsed)
                except Exception as e:
                    print(f"[WARN] Failed parsing {f.name}: {e}")
                break
    for it in aggregated:
        q = it.get("question","")
        gt = it.get("ground_truth","")
        if is_distracting(q, gt):
            it["category"] = "distracting"
    aggregated = dedupe(aggregated)
    counts = Counter(it.get("category","unknown") for it in aggregated)
    print("[INFO] Counts by category (after parse & dedupe):", counts)
    out_folder = Path(out_path).parent
    out_folder.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    per_cat = sample_size // 3
    by_cat = {"easy":[], "hard":[], "distracting":[]}
    for it in aggregated:
        cat = it.get("category","easy")
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(it)
    sample = []
    for cat in ["easy","hard","distracting"]:
        pool = by_cat.get(cat, [])
        if len(pool) <= per_cat:
            sample.extend(pool)
        else:
            sample.extend(random.sample(pool, per_cat))
    remaining = [it for it in aggregated if it not in sample]
    need = sample_size - len(sample)
    if need > 0 and remaining:
        sample.extend(random.sample(remaining, min(need, len(remaining))))
    sample_path = str(Path(out_path).with_name(f"dataset_sample_{sample_size}.json"))
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    report = {
        "total_normalized": len(aggregated),
        "counts": dict(counts),
        "sample_size": len(sample),
        "sample_path": sample_path
    }
    report_path = str(Path(out_path).with_name("prepare_report.json"))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[INFO] Wrote normalized dataset to", out_path)
    print("[INFO] Wrote sample to", sample_path)
    print("[INFO] Report ->", report_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="raw_data", help="Folder containing raw dataset files")
    p.add_argument("--out", default="samples/dataset_normalized.json", help="Output normalized JSON")
    p.add_argument("--sample_size", type=int, default=100)
    args = p.parse_args()
    main(args.raw_dir, args.out, sample_size=args.sample_size)
