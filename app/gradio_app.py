"""Gradio 演示应用:浏览器里的「小K」聊天界面。

用法:
  python app/gradio_app.py --config configs/sft_lora.yaml
  # 打开终端提示的 http://127.0.0.1:7860
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent


def load_model(cfg: dict):
    mcfg, tcfg = cfg["model"], cfg["train"]
    quant = None
    if cfg.get("lora", {}).get("use_qlora", False):
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
    # 优先使用 DPO 产物(SFT+DPO 合一的单 adapter),否则回退到纯 SFT adapter
    dpo_dir = ROOT / "outputs/dpo-lora/final"
    sft_dir = ROOT / tcfg["output_dir"] / "final"
    adapter = dpo_dir if dpo_dir.exists() else sft_dir
    if adapter.exists():
        model = PeftModel.from_pretrained(model, str(adapter))
        print(f"[INFO] 已加载 adapter: {adapter}")
    else:
        print(f"[WARN] 未找到 adapter({adapter}),以基座模型模式运行")
    model.eval()
    return tokenizer, model


def main() -> int:
    parser = argparse.ArgumentParser(description="小K Gradio 演示")
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公网临时链接")
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gcfg, dpcfg = cfg["generate"], cfg["deploy"]

    tokenizer, model = load_model(cfg)

    @torch.no_grad()
    def chat(message: str, history: list[dict]):
        messages = [{"role": "system", "content": dpcfg["system_prompt"]}]
        messages += history
        messages.append({"role": "user", "content": message})
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
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    demo = gr.ChatInterface(
        fn=chat,
        title="小K · 中文学习助手",
        description="基于 Qwen2.5-14B + QLoRA 微调 | 数据清洗 → 训练 → 评估 → 部署 全流程开源",
        examples=["你是谁?", "什么是 LoRA 微调?", "训练时 loss 不下降怎么办?"],
    )
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
