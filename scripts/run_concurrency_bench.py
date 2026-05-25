#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import math
import statistics
import time

import requests


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * pct) - 1))
    return ordered[idx]


def run_one(endpoint, model, prompt, max_tokens, timeout, request_index):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    started = time.time()
    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        latency = time.time() - started
        response.raise_for_status()
        obj = response.json()
        usage = obj.get("usage") or {}
        completion_tokens = usage.get("completion_tokens") or 0
        text = (obj.get("choices") or [{}])[0].get("text", "")
        return {
            "request_index": request_index,
            "ok": True,
            "status": response.status_code,
            "latency_sec": round(latency, 6),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens"),
            "decode_tokens_per_sec": round(completion_tokens / latency, 6)
            if latency and completion_tokens
            else None,
            "text_prefix": text[:180],
        }
    except Exception as exc:
        latency = time.time() - started
        return {
            "request_index": request_index,
            "ok": False,
            "status": None,
            "latency_sec": round(latency, 6),
            "prompt_tokens": None,
            "completion_tokens": 0,
            "total_tokens": None,
            "decode_tokens_per_sec": None,
            "error": repr(exc),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/completions",
    )
    args = parser.parse_args()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    started = time.time()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_one,
                args.endpoint,
                args.model,
                prompt,
                args.max_tokens,
                args.timeout,
                i,
            )
            for i in range(args.requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            row.update(
                {
                    "label": args.label,
                    "concurrency": args.concurrency,
                    "requested_total": args.requests,
                    "max_tokens": args.max_tokens,
                }
            )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    wall_time = time.time() - started
    ok_rows = [r for r in rows if r.get("ok")]
    latencies = [r["latency_sec"] for r in ok_rows]
    speeds = [
        r["decode_tokens_per_sec"]
        for r in ok_rows
        if isinstance(r.get("decode_tokens_per_sec"), (int, float))
    ]
    completion_tokens = [r.get("completion_tokens") or 0 for r in ok_rows]
    total_completion_tokens = sum(completion_tokens)
    summary = {
        "label": args.label,
        "model": args.model,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "success": len(ok_rows),
        "failures": len(rows) - len(ok_rows),
        "success_rate": round(len(ok_rows) / len(rows), 6) if rows else 0,
        "wall_time_sec": round(wall_time, 6),
        "aggregate_completion_tokens": total_completion_tokens,
        "aggregate_completion_tokens_per_sec": round(
            total_completion_tokens / wall_time, 6
        )
        if wall_time
        else None,
        "avg_latency_sec": round(statistics.mean(latencies), 6)
        if latencies
        else None,
        "p50_latency_sec": round(statistics.median(latencies), 6)
        if latencies
        else None,
        "p95_latency_sec": round(percentile(latencies, 0.95), 6)
        if latencies
        else None,
        "p50_decode_tokens_per_sec": round(statistics.median(speeds), 6)
        if speeds
        else None,
        "completion_tokens": completion_tokens,
        "all_full_max_tokens": all(x == args.max_tokens for x in completion_tokens)
        if completion_tokens
        else False,
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))

    with open(args.output, "a", encoding="utf-8") as f:
        for row in sorted(rows, key=lambda x: x["request_index"]):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

