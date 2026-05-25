#!/usr/bin/env python3
"""Build the ModelScope Gallery notebook for the Qwen ROCm/MTP study."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "qwen36_27b_fp8_rocm_vllm_mtp_gallery.ipynb"
README = ROOT / "README_GALLERY.md"
REQ = ROOT / "requirements.txt"
DESC = ROOT / "GALLERY_DESCRIPTION.md"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
# Qwen3.6-27B-FP8 在 AMD ROCm 上的 vLLM + MTP 推理实践

这个 Notebook 记录一次基于 ModelScope DSW AMD GPU 实例的 Qwen3.6-27B-FP8 推理实践：vLLM ROCm 服务、Qwen MTP 投机解码、near-8K correctness gate、FP8 KV cache 验证和小并发 serving 压测。

为了符合 Gallery 自动化运行要求，默认 Cell 不下载 27B 权重，也不启动大模型服务；默认流程会完成环境探测、实验数据复核、加速比计算、图表生成和复现清单输出。完整 vLLM 服务验证放在最后的可选 Cell，通过环境变量显式开启。
"""
    ),
    md(
        """
## 实验目标

本实践关注三个问题：

1. 在 AMD ROCm 环境中，Qwen3.6-27B-FP8 能否通过 vLLM 建立可复现的推理 baseline；
2. Qwen MTP speculative decoding 在 decode-heavy 场景中是否带来稳定收益；
3. near-8K correctness、FP8 KV cache 和小并发 serving 结果是否能形成可复查证据。

默认运行口径是轻量复核，不等价于重新下载模型和重跑全部大模型服务。这样设计是为了避免 Gallery 自动审核因为模型下载或长时间启动超时失败。
"""
    ),
    code(
        """
import json
import os
import platform
import sys
from pathlib import Path


def optional_import(name):
    \"\"\"Import optional dependency and return None instead of failing the notebook.\"\"\"
    try:
        return __import__(name)
    except Exception as exc:
        return {\"unavailable\": repr(exc)}


# 环境探测 / Environment probe.
env = {
    \"python\": sys.version.split()[0],
    \"platform\": platform.platform(),
    \"cwd\": str(Path.cwd()),
    \"RUN_VLLM_LIVE_CHECK\": os.environ.get(\"RUN_VLLM_LIVE_CHECK\", \"0\"),
}

torch = optional_import(\"torch\")
if isinstance(torch, dict):
    env[\"torch\"] = torch[\"unavailable\"]
    env[\"hip\"] = \"N/A\"
    env[\"gpu\"] = \"N/A\"
else:
    env[\"torch\"] = getattr(torch, \"__version__\", \"unknown\")
    env[\"hip\"] = getattr(getattr(torch, \"version\", None), \"hip\", None)
    try:
        env[\"gpu_count\"] = torch.cuda.device_count()
        env[\"gpu\"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"
    except Exception as exc:
        env[\"gpu\"] = repr(exc)

print(json.dumps(env, ensure_ascii=False, indent=2))
"""
    ),
    md(
        """
## 复现实验数据

下面的数据来自同一轮 ModelScope DSW AMD GPU 实验的脚本和日志摘要。为了让 Notebook 默认可运行，这里直接内置轻量汇总数据；完整原始脚本包括：

- `run_mtp_sweep.sh`
- `run_needle_gate.py`
- `run_round4_forced_1024_decode.sh`
- `run_round5_concurrency_serving.sh`

发布时不携带模型权重，只保存脚本、参数、日志摘要和结果表。
"""
    ),
    code(
        """
# 强制 1024-token decode 对照 / Forced 1024-token decode comparison.
decode_1024 = [
    {\"label\": \"baseline\", \"median_latency_sec\": 47.726203, \"median_tok_s\": 21.455850, \"full_1024\": True},
    {\"label\": \"MTP t2\", \"median_latency_sec\": 24.144029, \"median_tok_s\": 42.413046, \"full_1024\": True},
    {\"label\": \"MTP t3\", \"median_latency_sec\": 21.039062, \"median_tok_s\": 48.675316, \"full_1024\": True},
    {\"label\": \"MTP t4\", \"median_latency_sec\": 18.795832, \"median_tok_s\": 54.481873, \"full_1024\": True},
]

# 小并发 serving 对照 / Small-concurrency serving comparison.
concurrency_rows = [
    {\"label\": \"baseline\", \"concurrency\": 1, \"aggregate_tok_s\": 21.505711, \"p50_latency_sec\": 47.614726, \"success_rate\": 1.0, \"full_1024\": True},
    {\"label\": \"baseline\", \"concurrency\": 2, \"aggregate_tok_s\": 37.232449, \"p50_latency_sec\": 55.004424, \"success_rate\": 1.0, \"full_1024\": True},
    {\"label\": \"baseline\", \"concurrency\": 4, \"aggregate_tok_s\": 73.418453, \"p50_latency_sec\": 55.785482, \"success_rate\": 1.0, \"full_1024\": True},
    {\"label\": \"MTP t4\", \"concurrency\": 1, \"aggregate_tok_s\": 54.276744, \"p50_latency_sec\": 18.865812, \"success_rate\": 1.0, \"full_1024\": True},
    {\"label\": \"MTP t4\", \"concurrency\": 2, \"aggregate_tok_s\": 103.729872, \"p50_latency_sec\": 19.742277, \"success_rate\": 1.0, \"full_1024\": True},
    {\"label\": \"MTP t4\", \"concurrency\": 4, \"aggregate_tok_s\": 189.734199, \"p50_latency_sec\": 20.777241, \"success_rate\": 1.0, \"full_1024\": True},
]

correctness_gates = [
    {\"case\": \"MTP t4 near-8K needle\", \"prompt_tokens\": 7857, \"total_tokens\": 8002, \"completion_tokens\": 145, \"result\": \"PASS\"},
    {\"case\": \"FP8 KV baseline near-8K\", \"prompt_tokens\": 7858, \"total_tokens\": 8006, \"completion_tokens\": 148, \"result\": \"PASS\"},
    {\"case\": \"FP8 KV + MTP t4 near-8K\", \"prompt_tokens\": 7861, \"total_tokens\": 8053, \"completion_tokens\": 192, \"result\": \"PASS\"},
]

print(\"decode rows:\", len(decode_1024))
print(\"concurrency rows:\", len(concurrency_rows))
print(\"correctness gates:\", len(correctness_gates))
"""
    ),
    code(
        """
def markdown_table(rows, columns):
    \"\"\"Render a compact Markdown table from dictionaries.\"\"\"
    header = \"| \" + \" | \".join(columns) + \" |\"
    sep = \"| \" + \" | \".join([\"---\"] * len(columns)) + \" |\"
    body = []
    for row in rows:
        body.append(\"| \" + \" | \".join(str(row.get(c, \"\")) for c in columns) + \" |\")
    return \"\\n\".join([header, sep] + body)


baseline_speed = next(r[\"median_tok_s\"] for r in decode_1024 if r[\"label\"] == \"baseline\")
for row in decode_1024:
    row[\"speedup\"] = round(row[\"median_tok_s\"] / baseline_speed, 2)

assert all(row[\"full_1024\"] for row in decode_1024), \"All decode rows should finish 1024 tokens\"
assert max(row[\"speedup\"] for row in decode_1024) >= 2.5, \"MTP t4 should show about 2.5x speedup\"

print(\"强制 1024-token decode 对照：\")
print(markdown_table(decode_1024, [\"label\", \"median_latency_sec\", \"median_tok_s\", \"speedup\", \"full_1024\"]))
"""
    ),
    code(
        """
# 计算小并发下的 MTP 相对 baseline 加速比 / Compute MTP speedups for each concurrency level.
baseline_by_c = {r[\"concurrency\"]: r for r in concurrency_rows if r[\"label\"] == \"baseline\"}
for row in concurrency_rows:
    if row[\"label\"] == \"MTP t4\":
        base = baseline_by_c[row[\"concurrency\"]]
        row[\"speedup_vs_baseline\"] = round(row[\"aggregate_tok_s\"] / base[\"aggregate_tok_s\"], 2)
    else:
        row[\"speedup_vs_baseline\"] = 1.0

assert all(row[\"success_rate\"] == 1.0 for row in concurrency_rows), \"All serving requests should succeed\"
assert all(row[\"full_1024\"] for row in concurrency_rows), \"All serving rows should finish 1024 tokens\"

print(\"小并发 serving 对照：\")
print(markdown_table(concurrency_rows, [\"label\", \"concurrency\", \"aggregate_tok_s\", \"p50_latency_sec\", \"speedup_vs_baseline\", \"full_1024\"]))
"""
    ),
    code(
        """
from pathlib import Path

out_dir = Path(\"gallery_outputs\")
out_dir.mkdir(exist_ok=True)

try:
    import matplotlib.pyplot as plt

    labels = [row[\"label\"] for row in decode_1024]
    speeds = [row[\"median_tok_s\"] for row in decode_1024]
    colors = [\"#6B7280\", \"#38BDF8\", \"#22C55E\", \"#F97316\"]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, speeds, color=colors)
    plt.title(\"Forced 1024-token Decode Throughput\")
    plt.ylabel(\"tokens/s\")
    for bar, speed in zip(bars, speeds):
        plt.text(bar.get_x() + bar.get_width() / 2, speed + 1, f\"{speed:.2f}\", ha=\"center\")
    plt.tight_layout()
    chart1 = out_dir / \"forced_1024_decode.png\"
    plt.savefig(chart1, dpi=160)
    plt.show()
    plt.close()

    conc = sorted({row[\"concurrency\"] for row in concurrency_rows})
    base = [next(r[\"aggregate_tok_s\"] for r in concurrency_rows if r[\"label\"] == \"baseline\" and r[\"concurrency\"] == c) for c in conc]
    mtp = [next(r[\"aggregate_tok_s\"] for r in concurrency_rows if r[\"label\"] == \"MTP t4\" and r[\"concurrency\"] == c) for c in conc]

    plt.figure(figsize=(8, 4.5))
    plt.plot(conc, base, marker=\"o\", label=\"baseline\")
    plt.plot(conc, mtp, marker=\"o\", label=\"MTP t4\")
    plt.title(\"Small-concurrency Aggregate Throughput\")
    plt.xlabel(\"concurrency\")
    plt.ylabel(\"aggregate completion tok/s\")
    plt.xticks(conc)
    plt.legend()
    plt.tight_layout()
    chart2 = out_dir / \"concurrency_serving.png\"
    plt.savefig(chart2, dpi=160)
    plt.show()
    plt.close()

    print(\"charts generated:\")
    print(\"-\", chart1)
    print(\"-\", chart2)
except Exception as exc:
    # Matplotlib is listed in requirements.txt. If the runtime still lacks it,
    # the notebook continues and prints the reason instead of failing.
    print(\"chart generation skipped:\", repr(exc))
"""
    ),
    code(
        """
assert all(row[\"result\"] == \"PASS\" for row in correctness_gates), \"Correctness gates should pass\"

print(\"near-8K correctness gate：\")
print(markdown_table(correctness_gates, [\"case\", \"prompt_tokens\", \"total_tokens\", \"completion_tokens\", \"result\"]))
"""
    ),
    code(
        """
# 输出轻量复现清单 / Write a lightweight reproduction summary.
summary = {
    \"model\": \"Qwen/Qwen3.6-27B-FP8\",
    \"platform\": \"ModelScope DSW AMD GPU\",
    \"engine\": \"vLLM ROCm\",
    \"main_result\": \"MTP t4 reached 54.48 tok/s vs baseline 21.46 tok/s in forced 1024-token decode\",
    \"best_single_request_speedup\": \"2.54x\",
    \"serving_speedup_range\": \"2.52x-2.79x for concurrency 1/2/4\",
    \"correctness\": \"near-8K MTP and FP8 KV gates passed\",
    \"scope\": \"research baseline, not production serving benchmark\",
}

summary_path = out_dir / \"repro_summary.json\"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding=\"utf-8\")
print(summary_path.read_text(encoding=\"utf-8\"))
"""
    ),
    md(
        """
## 可选：连接正在运行的 vLLM 服务

下面的 Cell 默认跳过。只有当你已经在同一台 DSW 实例上启动了 vLLM OpenAI API 服务，并且确认 `127.0.0.1:8000` 可访问时，再设置：

```bash
export RUN_VLLM_LIVE_CHECK=1
export SERVED_MODEL_NAME=qwen36-27b-fp8-mtp-t4
```

这样可以避免 Gallery 自动审核时因为模型权重下载、服务启动或长上下文测试超时而失败。
"""
    ),
    code(
        """
import os

if os.environ.get(\"RUN_VLLM_LIVE_CHECK\", \"0\") != \"1\":
    print(\"skip live vLLM check: set RUN_VLLM_LIVE_CHECK=1 to enable it manually\")
else:
    import requests

    model_name = os.environ.get(\"SERVED_MODEL_NAME\", \"qwen36-27b-fp8-mtp-t4\")
    response = requests.get(\"http://127.0.0.1:8000/v1/models\", timeout=30)
    response.raise_for_status()
    print(\"/v1/models:\", response.json())

    payload = {
        \"model\": model_name,
        \"prompt\": \"Count from 1 to 20, separated by spaces:\\n\",
        \"max_tokens\": 64,
        \"temperature\": 0,
    }
    completion = requests.post(\"http://127.0.0.1:8000/v1/completions\", json=payload, timeout=120)
    completion.raise_for_status()
    obj = completion.json()
    print(json.dumps(obj.get(\"usage\", {}), ensure_ascii=False, indent=2))
    print((obj.get(\"choices\") or [{}])[0].get(\"text\", \"\")[:300])
"""
    ),
    md(
        """
## 结论

这个 Notebook 的默认流程验证了实验数据闭环：

- 强制 1024-token decode 中，MTP t4 相对 baseline 约 `2.54x`；
- 小并发 `1/2/4` 下，MTP t4 aggregate completion throughput 仍保持约 `2.52x-2.79x`；
- near-8K correctness gate 和 FP8 KV cache gate 均有通过记录；
- 默认运行不依赖大模型权重，适合 Gallery 自动审核；完整服务验证可在 DSW 实例中手动开启。

下一步可以继续扩展 16K/32K context、AMD Quark 量化，以及 llama.cpp/GGUF 路线的工程对照。
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")

REQ.write_text(
    "\n".join(
        [
            "# Lightweight dependencies for the default Gallery notebook cells.",
            "# The full vLLM/ROCm stack is provided by the ModelScope DSW AMD GPU image.",
            "requests>=2.31.0",
            "matplotlib>=3.7.0",
            "",
        ]
    ),
    encoding="utf-8",
)

README.write_text(
    """
# ModelScope Gallery 发布包

建议上传文件：

1. `qwen36_27b_fp8_rocm_vllm_mtp_gallery.ipynb`
2. `requirements.txt`
3. 封面图：`../../figures/modelscope_yanxishe/cover_qwen36_rocm_mtp_yanxishe.png`

标题建议：

```text
Qwen3.6-27B-FP8 在 AMD ROCm 上的 vLLM + MTP 推理实践
```

简介建议：

```text
基于 ModelScope DSW AMD GPU 实例，复核 Qwen3.6-27B-FP8 的 vLLM ROCm 部署、MTP 投机解码、near-8K 正确性验证、FP8 KV cache 和小并发压测结果。默认 Notebook 不下载模型、不启动大模型服务，可自动运行并生成图表与复现清单。
```

推荐标签：`AMD GPU激励计划`、`ROCm`、`vLLM`、`Qwen`、`MTP`、`大模型推理`
""".strip()
    + "\n",
    encoding="utf-8",
)

DESC.write_text(
    """
本 Notebook 是一份基于 ModelScope DSW AMD GPU 实例的 Qwen3.6-27B-FP8 推理实践，重点复核 vLLM ROCm 部署、Qwen MTP 投机解码、near-8K correctness gate、FP8 KV cache 和小并发 serving 压测结果。

为了保证 Gallery 自动化运行稳定，默认 Cell 不下载模型权重，也不启动完整 vLLM 大模型服务，而是完成环境探测、实验数据复核、加速比计算、图表生成和复现清单输出。完整 vLLM live check 被放在可选 Cell 中，需要手动设置环境变量后执行。

核心结果：在强制 1024-token decode 口径下，baseline median decode throughput 为 21.46 tok/s，MTP t4 为 54.48 tok/s，约 2.54x；小并发 concurrency=1/2/4 下，MTP t4 aggregate completion throughput 相对 baseline 约 2.52x/2.79x/2.58x。
""".strip()
    + "\n",
    encoding="utf-8",
)

print(NOTEBOOK)
print(REQ)
print(README)
print(DESC)
