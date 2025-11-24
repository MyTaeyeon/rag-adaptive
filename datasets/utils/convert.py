import json

def convert_hotpot_record(r):
    uid = r['id']
    uquestion = r['question']
    answer = [r['answer']]

    context_parts = []
    context_docs = []

    context = r['context']
    titles = context['title']
    sentences = context['sentences']

    for title, sentence in zip(titles, sentences):
        text = " ".join(sentence)
        context_parts.append(f"{title}. {text}")
        context_docs.append({
            "doc_id": f"{uid}_{title.replace(' ', '_')}",
            "title": title,
            "text": text
        })
    
    flattened_context = "\n".join(context_parts)
    
    sf_full = []
    sf = r['supporting_facts']
    sf_til = sf['title']
    sentid = sf['sent_id']
    for til, idx in zip(sf_til, sentid):
        sent_text = None
        for t, sents in zip(titles, sentences):
            if t == til and idx < len(sents):
                sent_text = sents[idx]
                break
        sf_full.append({
            "title": til,
            "sent_id": idx,
            "sentence": sent_text
        })
    
    return {
        "id": uid,
        "question": uquestion,
        "context": flattened_context,
        "answer": answer,
        "dataset": "hotpotqa",
        "context_docs": context_docs,
        "supporting_fact": sf_full,
        "type": r['type'],
        "level": r['level']
    }

def convert_vimed_record(r):
    # r: dict-like sample from HF dataset
    uid = r.get("question_idx")
    if uid is None:
        # fallback to hash
        uid = f"vimed_{abs(hash(r.get('question', '')+str(r.get('answer', ''))))%10_000_000}"
    question = r.get("question", "").strip()
    answer = r.get("answer", "").strip()
    answers = [answer] if answer else [""]
    # context: sometimes empty, sometimes paragraph
    context_text = r.get("context") or ""
    title = r.get("title") or ""
    # build context_docs: use title + context
    context_docs = []
    if title or context_text:
        doc_id = f"{uid}_doc_{title.replace(' ', '_')[:64] or 'doc'}"
        context_docs.append({"doc_id": doc_id, "title": title, "text": context_text})
    flattened_context = (title + ". " + context_text).strip() if (title or context_text) else ""

    # metadata
    metadata = {
        "keywords": r.get("keyword"),
        "topic": r.get("topic"),
        "article_url": r.get("article_url"),
        "author": r.get("author"),
        "author_url": r.get("author_url")
    }

    unified = {
        "id": uid,
        "question": question,
        "context": flattened_context,
        "answers": answers,
        "dataset": "vimed",
        "context_docs": context_docs,
        "metadata": metadata
    }
    return unified

def convert_viquad_record(r):
    uid = str(r.get("id") or r.get("uit_id") or f"viquad_{abs(hash(r.get('question','')))%10_000_000}")
    title = r.get("title") or ""
    context_text = r.get("context") or ""
    question = r.get("question","").strip()

    # handle impossibility
    is_impossible = bool(r.get("is_impossible"))

    raw_answers = r.get("answers", {})
    # ViQuAD often stores answers as dict: {"text": [...], "answer_start": [...]}
    texts = []
    starts = []

    if isinstance(raw_answers, dict):
        # Try to extract both lists
        t = raw_answers.get("text")
        s = raw_answers.get("answer_start")
        # normalize types
        if isinstance(t, list):
            texts = [x for x in t]
        elif isinstance(t, str):
            texts = [t]
        if isinstance(s, list):
            starts = [int(x) for x in s]
        elif isinstance(s, (int, float, str)) and str(s).isdigit():
            starts = [int(s)]
    elif isinstance(raw_answers, list):
        # maybe list of strings or list of dicts
        for a in raw_answers:
            if isinstance(a, dict):
                if "text" in a: texts.append(a["text"])
                if "answer_start" in a: starts.append(int(a["answer_start"]))
            elif isinstance(a, str):
                texts.append(a)
    elif isinstance(raw_answers, str):
        texts = [raw_answers]

    # If texts empty but plausible_answers exists, you may fallback to plausible (optional)
    if not texts and r.get("plausible_answers"):
        pa = r.get("plausible_answers")
        if isinstance(pa, list):
            # plausible_answers might be list of strings or dicts
            for p in pa:
                if isinstance(p, str):
                    texts.append(p)
                elif isinstance(p, dict) and p.get("text"):
                    texts.append(p["text"])

    # Build answers list (strings) — if impossible, keep empty
    answers = [] if is_impossible else (texts if texts else [""])

    # Build answers_spans: prefer provided starts; if missing try find in context
    answers_spans = []
    used_positions = set()  # track (start,end) used to avoid duplicate match for repeated substring

    for idx, text in enumerate(texts):
        if is_impossible:
            break
        start = None
        end = None
        # if start available from starts list
        if idx < len(starts):
            try:
                start = int(starts[idx])
            except Exception:
                start = None
        # fallback: find the text in context (choose first unused occurrence)
        if start is None:
            search_from = 0
            while True:
                pos = context_text.find(text, search_from)
                if pos == -1:
                    start = None
                    break
                end_cand = pos + len(text)
                if (pos, end_cand) not in used_positions:
                    start = pos
                    break
                else:
                    search_from = pos + 1  # continue search
        if start is not None:
            end = start + len(text)
            used_positions.add((start, end))
            answers_spans.append({"text": text, "answer_start": start, "answer_end": end})
        else:
            # cannot find start -> still include text but leave start/end as None
            answers_spans.append({"text": text, "answer_start": None, "answer_end": None})

    # If there were zero texts but answers placeholder used ([""]), keep spans empty
    if not texts and not is_impossible:
        answers_spans = []

    # Build context_docs (one doc from title + context)
    doc_id = f"{uid}_doc_{title.replace(' ', '_')[:64] or 'doc'}"
    context_docs = [{"doc_id": doc_id, "title": title, "text": context_text}] if title or context_text else []

    unified = {
        "id": uid,
        "question": question,
        "context": (title + ". " + context_text).strip() if title else context_text,
        "answers": answers,
        "answers_spans": answers_spans,      # <-- useful for extractive readers
        "dataset": "viquad",
        "context_docs": context_docs,
        "metadata": {
            "uit_id": r.get("uit_id"),
            "is_impossible": is_impossible,
            "plausible_answers": r.get("plausible_answers")
        }
    }
    return unified

def convert_hotpot_dataset(ds_split, out_path):
    out = []
    for i, r in enumerate(ds_split):
        unified = convert_hotpot_record(r)
        out.append(unified)
    # write jsonl
    with open(out_path, "w", encoding="utf8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} records to {out_path}")

def convert_vimed_dataset(ds_split, out_path):
    out = []
    for i, r in enumerate(ds_split):
        unified = convert_vimed_record(r)
        out.append(unified)
    # write jsonl
    with open(out_path, "w", encoding="utf8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} records to {out_path}")

def convert_viquad_dataset(ds_split, out_path):
    out = []
    for i, r in enumerate(ds_split):
        unified = convert_viquad_record(r)
        out.append(unified)
    # write jsonl
    with open(out_path, "w", encoding="utf8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} records to {out_path}")
