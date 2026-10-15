"""LoRA SFT 训练脚本(基于 TRL SFTTrainer + PEFT)。

用法:
  python scripts/train_sft.py --config configs/sft_lora.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA 指令微调")
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg, dcfg, lcfg, tcfg = cfg["model"], cfg["data"], cfg["lora"], cfg["train"]

    # ---- 数据集(清洗产物;messages 格式,由 chat template 渲染)----
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(ROOT / dcfg["processed_dir"] / "train.jsonl"),
            "validation": str(ROOT / dcfg["processed_dir"] / "val.jsonl"),
        },
    )

    # ---- 量化(可选 QLoRA)----
    quant = None
    if lcfg.get("use_qlora"):
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
        dtype=torch.bfloat16 if tcfg.get("bf16") else torch.float16,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )

    peft_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        bias=lcfg["bias"],
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=str(ROOT / tcfg["output_dir"]),
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        logging_steps=tcfg["logging_steps"],
        save_strategy=tcfg["save_strategy"],
        bf16=tcfg.get("bf16", False),
        fp16=tcfg.get("fp16", False) and not tcfg.get("bf16", False),
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        max_grad_norm=tcfg["max_grad_norm"],
        report_to=tcfg.get("report_to", "none"),
        max_length=dcfg["max_length"],
        seed=dcfg.get("seed", 42),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(ROOT / tcfg["output_dir"] / "final"))
    tokenizer.save_pretrained(str(ROOT / tcfg["output_dir"] / "final"))
    print(f"\n训练完成,LoRA adapter 已保存到 {tcfg['output_dir']}/final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
