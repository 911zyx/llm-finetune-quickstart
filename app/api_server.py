"""FastAPI 部署:把微调模型暴露为 OpenAI 风格的 /v1/chat/completions 接口。

用法:
  python app/api_server.py --config configs/sft_lora.yaml
  # 接口文档: http://127.0.0.1:8000/docs

调用示例:
  curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"什么是 LoRA?"}]}'
"""
from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

import torch
import uvicorn
import yaml
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_new_tokens: int | None = None
    temperature: float | None = None


def build_app(cfg: dict) -> FastAPI:
    mcfg, tcfg, gcfg, dpcfg = cfg["model"], cfg["train"], cfg["generate"], cfg["deploy"]

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
        dtype=None if quant else torch.bfloat16,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )
    adapter = ROOT / tcfg["output_dir"] / "final"
    if adapter.exists():
        model = PeftModel.from_pretrained(model, str(adapter))
        print(f"[INFO] 已加载 LoRA adapter: {adapter}")
    model.eval()

    app = FastAPI(title="小K API", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": mcfg["base_model"], "adapter": adapter.exists()}

    @app.post("/v1/chat/completions")
    @torch.no_grad()
    def completions(req: ChatRequest):
        messages = [{"role": "system", "content": dpcfg["system_prompt"]}]
        messages += [m.model_dump() for m in req.messages]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        out = model.generate(
            inputs,
            max_new_tokens=req.max_new_tokens or gcfg["max_new_tokens"],
            temperature=req.temperature or gcfg["temperature"],
            top_p=gcfg["top_p"],
            repetition_penalty=gcfg["repetition_penalty"],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
        }

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="小K FastAPI 服务")
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    app = build_app(cfg)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
