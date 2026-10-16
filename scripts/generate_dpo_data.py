"""DPO 偏好对生成:用 SFT 模型对同一指令生成多个候选,规则打分后
最优为 chosen、最差为 rejected(要求分差 >= 2,保证偏好信号有意义)。

用法:
  python scripts/generate_dpo_data.py --config configs/sft_lora.yaml --n 120 --candidates 3
  # 产物: data/raw/dpo_pairs.jsonl,字段 prompt/chosen/rejected(messages 格式)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from generate_data import score_sample  # 复用同一打分标尺
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(description="DPO 偏好对生成")
    ap.add_argument("--config", default="configs/sft_lora.yaml")
    ap.add_argument("--n", type=int, default=120, help="目标偏好对数")
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--min-gap", type=int, default=2, help="chosen 与 rejected 的最小分差")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg, tcfg, dpcfg = cfg["model"], cfg["train"], cfg["deploy"]

    # 指令来源:清洗后的训练集(保证是目标分布内的真实指令)
    prompts = []
    with open(ROOT / cfg["data"]["processed_dir"] / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            prompts.append(next(m["content"] for m in msgs if m["role"] == "user"))
    random.Random(args.seed).shuffle(prompts)

    out_path = ROOT / "data/raw/dpo_pairs.jsonl"
    have = 0
    if out_path.exists():
        have = sum(1 for _ in open(out_path, encoding="utf-8"))
    print(f"已有 {have} 对,目标 {args.n} 对", flush=True)
    if have >= args.n:
        return 0

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(mcfg["base_model"], trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        mcfg["base_model"], dtype=torch.bfloat16, quantization_config=quant,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(ROOT / tcfg["output_dir"] / "final"))
    model.eval()

    preset = "coding"
    written, skipped = have, 0
    with open(out_path, "a", encoding="utf-8") as fout, torch.no_grad():
        for prompt in prompts:
            if written >= args.n:
                break
            messages = [
                {"role": "system", "content": dpcfg["system_prompt"]},
                {"role": "user", "content": prompt},
            ]
            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
            ).to(model.device)

            cands = []
            for _ in range(args.candidates):
                out = model.generate(
                    **inputs, max_new_tokens=400, temperature=0.9, top_p=0.95,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                if len(text) >= 30:
                    cands.append(text)
            if len(cands) < 2:
                skipped += 1
                continue

            scored = sorted(
                ((score_sample({"output": c}, preset), c) for c in cands),
                key=lambda x: x[0],
            )
            worst_s, worst = scored[0]
            best_s, best = scored[-1]
            if best_s - worst_s < args.min_gap or best == worst:
                skipped += 1
                continue

            fout.write(json.dumps({
                "prompt": messages,
                "chosen": [{"role": "assistant", "content": best}],
                "rejected": [{"role": "assistant", "content": worst}],
                "score_gap": best_s - worst_s,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            if written % 10 == 0:
                print(f"[dpo] {written}/{args.n} 对 (跳过 {skipped})", flush=True)

    print(f"完成:共 {written} 对,跳过 {skipped} 条指令", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
