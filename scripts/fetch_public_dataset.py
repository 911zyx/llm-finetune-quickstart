"""抓取公开指令数据集并转换为本项目的 Alpaca JSONL 格式。

流式读取(不整包下载),随机打乱后采样 N 条,字段名可映射。

用法:
  # 中文数学推理(字段天然匹配)
  python scripts/fetch_public_dataset.py --dataset BelleGroup/school_math_0.25M --n 600 \
      --out data/raw/public_school_math.jsonl
  # Firefly 中文多任务(input/target 需映射)
  python scripts/fetch_public_dataset.py --dataset YeungNLP/firefly-train-1.1M --n 600 \
      --instruction-field input --output-field target --out data/raw/public_firefly.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description="公开指令数据集抓取与格式转换")
    ap.add_argument("--dataset", required=True, help="HuggingFace 数据集名")
    ap.add_argument("--n", type=int, default=600, help="采样条数")
    ap.add_argument("--instruction-field", default="instruction")
    ap.add_argument("--input-field", default="input")
    ap.add_argument("--output-field", default="output")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--out", required=True, help="输出 JSONL 路径(相对仓库根目录)")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    ds = ds.shuffle(buffer_size=20000, seed=args.seed)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            if written >= args.n:
                break
            try:
                instruction = str(row[args.instruction_field] or "").strip()
                inp = str(row.get(args.input_field) or "").strip()
                output = str(row[args.output_field] or "").strip()
            except (KeyError, AttributeError):
                skipped += 1
                continue
            if not (4 <= len(instruction) <= 500 and 4 <= len(output) <= args.max_chars):
                skipped += 1
                continue
            f.write(json.dumps(
                {"instruction": instruction, "input": inp, "output": output},
                ensure_ascii=False,
            ) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"[fetch] {written}/{args.n}", flush=True)

    print(f"完成:{args.dataset} -> {out_path},写入 {written} 条,跳过 {skipped} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
