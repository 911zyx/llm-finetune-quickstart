"""将 LoRA adapter 合并回基座模型,产出独立可分发的完整模型。

注意:合并在 CPU 上以 bf16 进行,内存需求 ≈ 模型权重 × 2 峰值。
  14B 模型约需 30GB+ 空闲内存;内存不足时可跳过合并,
  部署脚本(app/)支持「基座 + adapter」直接加载,效果等价。

用法:
  python scripts/merge_lora.py --config configs/sft_lora.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="合并 LoRA 权重")
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg, tcfg, dpcfg = cfg["model"], cfg["train"], cfg["deploy"]

    tokenizer = AutoTokenizer.from_pretrained(
        mcfg["base_model"], trust_remote_code=mcfg.get("trust_remote_code", False)
    )
    base = AutoModelForCausalLM.from_pretrained(
        mcfg["base_model"],
        dtype=torch.bfloat16,
        device_map="cpu",  # 合并放 CPU 做,避免占用显存
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )
    model = PeftModel.from_pretrained(base, str(ROOT / tcfg["output_dir"] / "final"))
    merged = model.merge_and_unload()

    out_dir = ROOT / dpcfg["merged_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"合并完成,完整模型已保存到 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
