#!/usr/bin/env python3
import argparse
import json
import time

import requests


FILLER = (
    "This deployment note discusses AMD ROCm, vLLM, Qwen, MTP, speculative "
    "decoding, FP8, KV cache, long context, quantization, and reproducible "
    "benchmarking. "
)


def build_prompt(repeats: int, needle: str) -> str:
    context = FILLER * repeats
    return (
        "You are doing a strict retrieval test. Return the secret key exactly. "
        "Do not explain.\n"
        f"Context before key:\n{context}\n"
        f"The secret retrieval key is {needle}.\n"
        f"Context after key:\n{context}\n"
        "Question: What is the secret retrieval key?\n"
        "Final answer:"
    )


def run_case(model: str, name: str, repeats: int, needle: str, max_tokens: int):
    prompt = build_prompt(repeats, needle)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    started = time.time()
    response = requests.post(
        "http://127.0.0.1:8000/v1/completions",
        json=payload,
        timeout=900,
    )
    latency = time.time() - started
    row = {
        "case": name,
        "repeats": repeats,
        "needle": needle,
        "status": response.status_code,
        "latency_sec": round(latency, 6),
    }
    try:
        obj = response.json()
    except Exception:
        row["raw_prefix"] = response.text[:500]
        row["pass"] = False
        return row
    text = (obj.get("choices") or [{}])[0].get("text", "")
    usage = obj.get("usage") or {}
    row.update(
        {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "text_tail": text[-500:],
            "pass": needle in text,
        }
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--cases",
        default="2k:40:amber-2749,6k:95:cobalt-8391,8k-near:125:violet-5082",
        help="comma separated name:repeats:needle cases",
    )
    args = parser.parse_args()

    rows = []
    for spec in args.cases.split(","):
        name, repeats, needle = spec.split(":", 2)
        row = run_case(args.model, name, int(repeats), needle, args.max_tokens)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    with open(args.output, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write(
            json.dumps(
                {
                    "summary": {
                        "model": args.model,
                        "cases": len(rows),
                        "passed": sum(1 for r in rows if r.get("pass")),
                        "all_pass": all(r.get("pass") for r in rows),
                    }
                },
                ensure_ascii=False,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()

