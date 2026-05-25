#!/usr/bin/env python3
import argparse
import json
import statistics
import time

import requests


def run_once(model: str, prompt: str, max_tokens: int, timeout: int):
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
        timeout=timeout,
    )
    latency = time.time() - started
    response.raise_for_status()
    obj = response.json()
    usage = obj.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or 0
    text = (obj.get("choices") or [{}])[0].get("text", "")
    return {
        "status": response.status_code,
        "latency_sec": round(latency, 6),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "decode_tokens_per_sec": round(completion_tokens / latency, 6)
        if latency and completion_tokens
        else None,
        "text_prefix": text[:300],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--prompt",
        default="Count from 1 to 100, separated by spaces:\n",
    )
    args = parser.parse_args()

    rows = []
    for i in range(args.repeat):
        row = run_once(args.model, args.prompt, args.max_tokens, args.timeout)
        row.update({"label": args.label, "repeat_index": i})
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    speeds = [
        r["decode_tokens_per_sec"]
        for r in rows
        if isinstance(r.get("decode_tokens_per_sec"), (int, float))
    ]
    summary = {
        "label": args.label,
        "model": args.model,
        "repeat": args.repeat,
        "median_decode_tokens_per_sec": round(statistics.median(speeds), 6)
        if speeds
        else None,
        "mean_decode_tokens_per_sec": round(statistics.mean(speeds), 6)
        if speeds
        else None,
        "median_latency_sec": round(statistics.median(r["latency_sec"] for r in rows), 6)
        if rows
        else None,
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))

    with open(args.output, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

