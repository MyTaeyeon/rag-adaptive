#!/usr/bin/env python3
"""
Streaming, memory-efficient version of prepare_dataset.py
Use: python process_datasets_stream.py --raw_dir ./raw_data --out_dir ./processed_data --demo_size 120
"""
import argparse
import json
import os
import random
import re
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

def flatten_context(ctx: Any) -> str:
    """
    Chuyển các dạng context khác nhau -> string:
    - HotpotQA: list of [title, [sentences...]]  --> "title. sent1 sent2 ... title2. sent1 ..."
    - list of strings -> join
    - dict -> json.dumps fallback
    - other -> str()
    """
    if ctx is None:
        return ""
    # Hotpot-like: list of [title, [sentences...]]
    if isinstance(ctx, list):
        parts = []
        try:
            for item in ctx:
                # item often = [title, [sents...]] or {"title":..., "sents": ...}
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    title = item[0]
                    sents = item[1]
                    if isinstance(sents, list):
                        parts.append(f"{title}. " + " ".join([str(s) for s in sents]))
                    else:
                        parts.append(str(title))
                elif isinstance(item, dict):
                    # try common keys
                    title = item.get("title") or item.get("doc_title") or ""
                    sents = item.get("sents") or item.get("sentences") or item.get("text") or ""
                    if isinstance(sents, list):
                        parts.append(f"{title}. " + " ".join([str(s) for s in sents]))
                    else:
                        parts.append(title + " " + str(sents))
                else:
                    parts.append(str(item))
            return " ".join(parts)
        except Exception:
            # fallback to JSON
            return json.dumps(ctx, ensure_ascii=False)
    # dict -> try join values or dump
    if isinstance(ctx, dict):
        # try common structure: {'title': '...', 'paragraphs': [...]} etc.
        try:
            pieces = []
            for k, v in ctx.items():
                if isinstance(v, list):
                    pieces.append(str(k) + ". " + " ".join([str(x) for x in v]))
                else:
                    pieces.append(str(v))
            return " ".join(pieces)
        except Exception:
            return json.dumps(ctx, ensure_ascii=False)
    # otherwise just string-cast
    return str(ctx)

def simple_normalize(t: Any, do_lower: bool = True) -> str:
    """
    Robust normalize: accepts str | list | dict | None.
    - Flatten lists/dicts sensibly (Hotpot-friendly)
    - Lowercase (optional), remove punctuation, collapse whitespace
    """
    if t is None:
        return ""
    # if not string, try to flatten (handles Hotpot list/dict)
    if not isinstance(t, str):
        t = flatten_context(t)

    # now t is str
    if do_lower:
        try:
            t = t.lower()
        except Exception:
            t = str(t)

    # remove punctuation (keep unicode word chars and spaces)
    try:
        t = PUNCT_RE.sub(" ", t)
        t = re.sub(r"\s+", " ", t).strip()
    except Exception:
        t = " ".join(str(t).split())

    return t

def stream_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # try to be robust: skip bad lines
                continue

def load_json_maybe_stream(path: Path):
    """If .jsonl -> stream; if .json and small -> load; if .json and large -> try to yield lines if possible"""
    if path.suffix.lower() == ".jsonl":
        yield from stream_jsonl(path)
        return
    # try to detect if file is line-delimited (each line is a JSON object)
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return
        first_strip = first.lstrip()
        # if first non-space char is '[' then it's a JSON array — fallback to full load (user should convert to jsonl or install ijson)
        if first_strip.startswith('['):
            # load fully but warn
            f.seek(0)
            obj = json.load(f)
            if isinstance(obj, list):
                for it in obj:
                    yield it
            elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
                for it in obj["data"]:
                    yield it
            else:
                # fallback
                yield obj
            return
        else:
            # assume line-delimited JSON objects
            # yield first line then the rest
            try:
                yield json.loads(first)
            except Exception:
                pass
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

# ----------------- simple generic mapper (fallback) ------------------
def map_generic(entry):
    q = entry.get('question') or entry.get('query') or entry.get('prompt') or ""
    a = entry.get('answer') or entry.get('answers') or ""
    if isinstance(a, list) and a:
        a = a[0]
    return {
        'id': entry.get('id') or entry.get('qid') or str(hash(q))[:12],
        'question': q or "",
        'ground_truth': a or "",
        'category': entry.get('category') or 'easy',
        'context': entry.get('context') or ""
    }

# ----------------- streaming write helpers ---------------------------
def write_jsonl_line(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

def assign_split(item_id: str, train_frac: float, dev_frac: float) -> str:
    # deterministic assignment via sha1 hash of id
    h = int(hashlib.sha1(item_id.encode('utf-8')).hexdigest(), 16) % 100000
    t_cut = int(train_frac * 100000)
    d_cut = int((train_frac + dev_frac) * 100000)
    if h < t_cut:
        return "train"
    elif h < d_cut:
        return "dev"
    else:
        return "test"

# ---------------- reservoir sampler for demo --------------------------
class ReservoirPerCategory:
    def __init__(self, max_size: int, seed: int = 42):
        self.max_size = max_size
        self.seed = seed
        self.total = 0
        self.by_cat = {}  # cat -> list
        random.seed(seed)

    def add(self, item):
        cat = item.get('category', 'easy')
        self.total += 1
        if cat not in self.by_cat:
            self.by_cat[cat] = []
        lst = self.by_cat[cat]
        # naive per-category reservoir with cap proportional to max_size / (num categories so far)
        # to keep things simple cap per category = max(1, self.max_size // max(1, len(self.by_cat)))
        cap = max(1, self.max_size // max(1, len(self.by_cat)))
        if len(lst) < cap:
            lst.append(item)
        else:
            # replace with decreasing probability
            r = random.randint(0, self.total)
            if r < cap:
                idx = random.randrange(cap)
                lst[idx] = item

    def collect(self, target_size):
        # flatten and trim
        all_items = []
        for lst in self.by_cat.values():
            all_items.extend(lst)
        if len(all_items) <= target_size:
            return all_items
        return random.sample(all_items, target_size)

# ---------------- main streaming pipeline ----------------------------
def process_stream(raw_dir: Path, out_dir: Path, demo_size: int, seed: int,
                   lowercase: bool, train_frac: float, dev_frac: float,
                   max_context_chars: int, no_normalize_context: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_fp = (out_dir / "combined.jsonl").open("w", encoding="utf-8")
    train_fp = (out_dir / "train.jsonl").open("w", encoding="utf-8")
    dev_fp = (out_dir / "dev.jsonl").open("w", encoding="utf-8")
    test_fp = (out_dir / "test.jsonl").open("w", encoding="utf-8")

    demo_sampler = ReservoirPerCategory(max_size=demo_size, seed=seed)

    total = 0
    for p in raw_dir.iterdir():
        if p.suffix.lower() not in ('.json', '.jsonl'):
            continue
        print(f"Processing {p} ... (size={p.stat().st_size/1024/1024:.1f} MB)")
        for raw in load_json_maybe_stream(p):
            # try dataset-specific mapping if possible (kept simple here)
            try:
                rec = map_generic(raw)
            except Exception:
                continue

            # optionally trim context
            ctx = rec.get('context', '') or ''

            # --- FIX HOTPOTQA CONTEXT (convert list -> string) ---
            if isinstance(ctx, list):
                try:
                    new_ctx_parts = []
                    for title, sents in ctx:
                        if isinstance(sents, list):
                            new_ctx_parts.append(title + ". " + " ".join(sents))
                        else:
                            new_ctx_parts.append(title)
                    ctx = " ".join(new_ctx_parts)
                except Exception:
                    ctx = json.dumps(ctx, ensure_ascii=False)

            if max_context_chars and len(ctx) > max_context_chars:
                ctx = ctx[:max_context_chars]

            if not no_normalize_context:
                ctx_norm = simple_normalize(ctx, do_lower=lowercase)
            else:
                ctx_norm = ctx


            q_norm = simple_normalize(rec.get('question', ''), do_lower=lowercase)
            a_norm = simple_normalize(rec.get('ground_truth', ''), do_lower=lowercase)

            item = {
                'id': rec.get('id') or str(hash(q_norm))[:12],
                'question': q_norm,
                'ground_truth': a_norm,
                'context': ctx_norm,
                'category': rec.get('category', 'easy')
            }

            # write combined
            write_jsonl_line(combined_fp, item)

            # deterministic split assignment
            sp = assign_split(item['id'], train_frac, dev_frac)
            if sp == "train":
                write_jsonl_line(train_fp, item)
            elif sp == "dev":
                write_jsonl_line(dev_fp, item)
            else:
                write_jsonl_line(test_fp, item)

            # add to demo sampler
            demo_sampler.add(item)

            total += 1
            if total % 10000 == 0:
                print(f"  processed {total} items...")

    combined_fp.close()
    train_fp.close()
    dev_fp.close()
    test_fp.close()

    demo = demo_sampler.collect(demo_size)
    write_jsonl_line((out_dir / "demo.jsonl").open("w", encoding="utf-8"), demo)
    print(f"Done. Processed ~{total} items. Demo size = {len(demo)}")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--demo_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lowercase', action='store_true', default=True)
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false')
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--dev_frac', type=float, default=0.1)
    parser.add_argument('--max_context_chars', type=int, default=2000,
                        help='Trim context to this many characters (0 = no trim)')
    parser.add_argument('--no_normalize_context', action='store_true', help='Do not run normalize on context (faster)')
    args = parser.parse_args(argv)

    process_stream(Path(args.raw_dir), Path(args.out_dir), args.demo_size, args.seed,
                   args.lowercase, args.train_frac, args.dev_frac,
                   args.max_context_chars, args.no_normalize_context)

if __name__ == '__main__':
    main()