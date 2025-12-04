"""
llm_judge.py
Template to use a strong LLM as a judge.
Given: system_answer, ground_truth, instructions -> return numeric score and rationale.

You must provide an implementation for `call_llm` according to your LLM provider (OpenAI, Anthropic, local LLM...).
"""
import json

def build_prompt(system_answer: str, ground_truth: str, instructions: str = None):
    instr = instructions or ("You are an expert grader. Rate the system answer against the ground truth from 0 to 1, "
                            "where 1 means fully correct and 0 means incorrect. Output JSON: {\"score\":0.0, \"reason\":\"...\"}.")
    prompt = f\"\"\"{instr}

Ground truth:
{ground_truth}

System answer:
{system_answer}

Return only a JSON object with fields: score (0..1), reason (short).
\"\"\"
    return prompt

def call_llm(prompt: str):
    # TODO: implement LLM call
    # Example (pseudocode):
    # resp = openai.ChatCompletion.create(model="gpt-5", messages=[{"role":"user","content":prompt}])
    # return resp["choices"][0]["message"]["content"]
    return json.dumps({"score":0.5, "reason":"LLM judge not configured; returning default 0.5"})

def judge(system_answer: str, ground_truth: str):
    prompt = build_prompt(system_answer, ground_truth)
    out = call_llm(prompt)
    try:
        return json.loads(out)
    except Exception:
        return {"score":0.5, "reason":"invalid llm output"}

if __name__ == "__main__":
    print(judge("Paris is the capital of France.","Paris"))
