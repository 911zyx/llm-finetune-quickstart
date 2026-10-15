"""评估脚本:验证集困惑度 + 微调前后生成对比。

用法:
  # 1) 困惑度对比(基座 vs 微调后)
  python scripts/evaluate.py --config configs/sft_lora.yaml --mode ppl
  # 2) 生成对比(同一批问题,基座/微调各答一次)
  python scripts/evaluate.py --config configs/sft_lora.yaml --mode generate
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent

# 人工评估用的测试问题(覆盖人设、知识、格式、编程四类)
EVAL_PROMPTS = [
    "你是谁?",
    "什么是 LoRA 微调?",
    "训练时 loss 不下降怎么办?",
    "把这句话翻译成英文:学而不思则罔,思而不学则殆。",
    "写一句关于深夜写代码的文案",
    "写一段 Python 代码:读取 CSV 文件并统计行数",
    "帮我看看这段代码为什么报错:nums = [1,2,3]; print(nums[5])",
]


def load_pair(cfg: dict, with_adapter: bool):
    mcfg, tcfg = cfg["model"], cfg["train"]
    use_qlora = cfg.get("lora", {}).get("use_qlora", False)

    # 大模型(如 14B)bf16 装不下单卡时,评估也用 4bit 加载
    quant = None
    if use_qlora:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        mcfg["base_model"], trust_remote_code=mcfg.get("trust_remote_code", False)
    )
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["base_model"],
        # 4bit 时同样用 bf16:非量化模块(embed/lm_head/norm)若为 fp32 会多吃数 GB 显存
        dtype=torch.bfloat16,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )
    if with_adapter:
        model = PeftModel.from_pretrained(model, str(ROOT / tcfg["output_dir"] / "final"))
    model.eval()
    return tokenizer, model


@torch.no_grad()
def perplexity(cfg: dict, with_adapter: bool) -> float:
    """在验证集上计算困惑度(对 assistant 回复部分)。"""
    dcfg = cfg["data"]
    tokenizer, model = load_pair(cfg, with_adapter)
    losses, tokens = 0.0, 0
    with open(ROOT / dcfg["processed_dir"] / "val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            messages = json.loads(line)["messages"]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=dcfg["max_length"]).to(model.device)
            out = model(**ids, labels=ids["input_ids"])
            n = ids["input_ids"].numel() - 1
            losses += out.loss.item() * n
            tokens += n
    ppl = math.exp(losses / max(tokens, 1))
    del model, tokenizer
    import gc
    gc.collect()              # 先触发 GC 释放循环引用,再清空显存缓存
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return ppl


@torch.no_grad()
def _generate_all(cfg: dict, with_adapter: bool, name: str) -> list[str]:
    """用单个模型回答全部测试问题,返回回答列表。用完即释放显存。"""
    gcfg, dpcfg = cfg["generate"], cfg["deploy"]
    tokenizer, model = load_pair(cfg, with_adapter)
    replies = []
    for prompt in EVAL_PROMPTS:
        messages = [
            {"role": "system", "content": dpcfg["system_prompt"]},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=gcfg["max_new_tokens"],
            temperature=gcfg["temperature"],
            top_p=gcfg["top_p"],
            repetition_penalty=gcfg["repetition_penalty"],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        replies.append(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    del model, tokenizer
    import gc
    gc.collect()              # 先触发 GC 释放循环引用,再清空显存缓存
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return replies


def generate_compare(cfg: dict):
    # 基座与微调模型先后加载、用完释放,避免两份权重同时占显存(14B 场景必需)
    base_replies = _generate_all(cfg, with_adapter=False, name="基座模型")
    tuned_replies = _generate_all(cfg, with_adapter=True, name="微调后")

    for prompt, rb, rt in zip(EVAL_PROMPTS, base_replies, tuned_replies):
        print("=" * 60)
        print(f"【问题】{prompt}")
        print(f"\n【基座模型】\n{rb}")
        print(f"\n【微调后】\n{rt}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="评估微调效果")
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    parser.add_argument("--mode", choices=["ppl", "generate"], default="generate")
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.mode == "ppl":
        ppl_base = perplexity(cfg, with_adapter=False)
        ppl_tuned = perplexity(cfg, with_adapter=True)
        print(f"验证集困惑度  基座: {ppl_base:.2f}  ->  微调后: {ppl_tuned:.2f}")
        print("困惑度下降说明模型更拟合目标数据分布。")
    else:
        generate_compare(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
