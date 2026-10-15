"""数据清洗脚本:原始 JSONL -> 训练可用的 train/val 划分。

清洗规则(按顺序执行):
  1. 格式校验    —— 必须是合法 JSON,且含 instruction/output 字段
  2. 空白归一化  —— 去除首尾空白、压缩连续空行
  3. 长度过滤    —— 任一字段过短/过长则丢弃
  4. 隐私脱敏    —— 含手机号/邮箱的样本直接丢弃(教学数据不应含 PII)
  5. 精确去重    —— 按 (instruction, input) 哈希去重,保留第一条

用法:
  python scripts/clean_data.py --config configs/sft_lora.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # 无 yaml 时使用内置默认配置
    yaml = None

ROOT = Path(__file__).resolve().parent.parent

# 手机号(中国大陆 11 位)与邮箱的简单正则
PII_PATTERNS = [
    re.compile(r"1[3-9]\d{9}"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
]

DEFAULTS = {
    "raw_path": "data/raw/sample_instructions.jsonl",
    "processed_dir": "data/processed",
    "val_ratio": 0.1,
    "min_chars": 4,
    "max_chars": 1500,
    "seed": 42,
}


def load_config(path: str | None) -> dict:
    if path and yaml:
        with open(ROOT / path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return {**DEFAULTS, **cfg.get("data", {})}
    return DEFAULTS


def normalize(text: str) -> str:
    """去除首尾空白,压缩 3 个以上连续换行为 2 个。"""
    text = text.strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def contains_pii(text: str) -> bool:
    return any(p.search(text) for p in PII_PATTERNS)


def to_messages(sample: dict, system_prompt: str | None = None) -> dict:
    """Alpaca 格式 -> ChatML messages 格式。"""
    instruction = sample["instruction"]
    if sample.get("input"):
        instruction = f"{instruction}\n{sample['input']}"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})
    messages.append({"role": "assistant", "content": sample["output"]})
    return {"messages": messages}


def clean(raw_path: Path, cfg: dict) -> tuple[list[dict], dict]:
    stats = {"total": 0, "bad_format": 0, "too_short_long": 0, "pii": 0, "duplicate": 0, "kept": 0}
    seen: set[str] = set()
    kept: list[dict] = []

    with open(raw_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1

            try:
                sample = json.loads(line)
                assert isinstance(sample["instruction"], str) and isinstance(sample["output"], str)
            except (json.JSONDecodeError, KeyError, AssertionError, TypeError):
                stats["bad_format"] += 1
                continue

            sample = {
                "instruction": normalize(sample["instruction"]),
                "input": normalize(sample.get("input", "") or ""),
                "output": normalize(sample["output"]),
            }

            lengths = [len(sample["instruction"]), len(sample["output"])]
            if min(lengths) < cfg["min_chars"] or max(lengths) > cfg["max_chars"]:
                stats["too_short_long"] += 1
                continue

            if contains_pii(sample["instruction"]) or contains_pii(sample["output"]):
                stats["pii"] += 1
                continue

            key = hashlib.md5((sample["instruction"] + sample["input"]).encode()).hexdigest()
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)

            kept.append(sample)
            stats["kept"] += 1

    return kept, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="清洗指令微调数据")
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    parser.add_argument("--system-prompt", default=None, help="可选:写入每条样本的 system 提示")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_path = ROOT / cfg["raw_path"]
    out_dir = ROOT / cfg["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"[ERROR] 找不到原始数据: {raw_path}", file=sys.stderr)
        return 1

    kept, stats = clean(raw_path, cfg)

    rng = random.Random(cfg["seed"])
    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * cfg["val_ratio"]))
    val, train = kept[:n_val], kept[n_val:]

    for name, rows in [("train", train), ("val", val)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(to_messages(row, args.system_prompt), ensure_ascii=False) + "\n")

    print("=" * 40)
    print(f"原始样本:      {stats['total']}")
    print(f"格式错误丢弃:  {stats['bad_format']}")
    print(f"长度异常丢弃:  {stats['too_short_long']}")
    print(f"含隐私信息丢弃: {stats['pii']}")
    print(f"重复丢弃:      {stats['duplicate']}")
    print(f"最终保留:      {stats['kept']}  (train={len(train)}, val={len(val)})")
    print(f"输出目录:      {out_dir}")
    print("=" * 40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
