"""DPO 偏好对齐训练:在 SFT adapter 权重基础上继续训练。

关键设计(适配 TRL 1.15 + PEFT 0.21):
- 策略模型 = 基座 + SFT adapter(作为 "default" 加载,is_trainable=True);
- TRL 检测到 PeftModel 且 ref_model=None 时,会把 "default" adapter 复制为冻结的
  "ref" adapter 作为参考模型,即参考 = 基座 + SFT 初始权重,语义严格正确;
- 训练直接在 SFT adapter 权重上做 DPO 更新,产物是一个「SFT+DPO 合一」的 adapter,
  保存到 outputs/dpo-lora/final;原 SFT adapter 先备份到 outputs/sft-lora-v6-final。
- 部署只需加载:基座 -> outputs/dpo-lora/final(单 adapter)。

用法:
  python scripts/train_dpo.py --config configs/sft_lora.yaml
  # 前置:先跑 scripts/train_sft.py 和 scripts/generate_dpo_data.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="DPO 偏好对齐")
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]

    dataset = load_dataset("json", data_files=str(ROOT / "data/raw/dpo_pairs.jsonl"))["train"]
    dataset = dataset.remove_columns([c for c in dataset.column_names
                                      if c not in ("prompt", "chosen", "rejected")])
    # 10% holdout:rewards/accuracies 的泛化口径(52 对时代只有训练集口径)
    split = dataset.train_test_split(test_size=0.1, seed=cfg["data"].get("seed", 42))
    train_ds, eval_ds = split["train"], split["test"]
    print(f"DPO 偏好对: 训练 {len(train_ds)} / 验证 {len(eval_ds)}")

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        mcfg["base_model"], trust_remote_code=mcfg.get("trust_remote_code", False)
    )
    base = AutoModelForCausalLM.from_pretrained(
        mcfg["base_model"], dtype=torch.bfloat16, quantization_config=quant,
        device_map="auto", trust_remote_code=mcfg.get("trust_remote_code", False),
    )

    # 备份 v6 SFT adapter(DPO 训练产物是独立目录,此处备份仅为版本管理习惯)
    sft_final = ROOT / cfg["train"]["output_dir"] / "final"
    backup = ROOT / "outputs/sft-lora-v6-final"
    if sft_final.exists() and not backup.exists():
        shutil.copytree(sft_final, backup)
        print(f"SFT adapter 已备份到 {backup}")

    # 策略模型 = 基座 + SFT adapter(可训练);TRL 内部会复制 default -> 冻结的 ref
    model = PeftModel.from_pretrained(base, str(sft_final), is_trainable=True)
    print("可训练参数:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    training_args = DPOConfig(
        output_dir=str(ROOT / "outputs/dpo-lora"),
        beta=0.1,                          # 偏离参考模型的惩罚强度
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5.0e-5,              # DPO 用比 SFT 更小的学习率
        lr_scheduler_type="cosine",
        warmup_steps=2,
        logging_steps=2,
        eval_strategy="epoch",             # 每个 epoch 在 holdout 上算 rewards 指标
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="none",
        max_length=cfg["data"]["max_length"],
        seed=cfg["data"].get("seed", 42),
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,   # PeftModel:TRL 会复制 default adapter 为冻结的 ref 作为参考模型
        ref_model=None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    out = ROOT / "outputs/dpo-lora/final"
    # 只保存训练后的 default adapter(SFT+DPO 合一),剔除 TRL 内部复制的 ref
    model.save_pretrained(str(out), selected_adapters=["default"])
    print(f"\nDPO 完成,adapter 已保存到 {out}")
    print("部署只需加载:基座 -> outputs/dpo-lora/final(单 adapter,评估/演示脚本已支持)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
