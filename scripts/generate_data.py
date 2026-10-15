"""Self-Instruct 数据合成:用基座模型自己生成训练数据。

思路:给定一批主题 + 少量人工种子样本做 few-shot,让模型按「小K」的人设风格
产出新的 (instruction, input, output) 三元组,再经过与 clean_data 相同的过滤。

用法:
  python scripts/generate_data.py --config configs/sft_lora.yaml --n 400
  # 产物: data/raw/generated_instructions.jsonl(可重复运行,自动跳过已有数量)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent

# 40 个多样化主题,覆盖问答/翻译/代码/写作/学习/数学等意图
TOPICS = [
    "机器学习概念解释", "深度学习调参技巧", "Python 编程问题", "数据结构与算法",
    "英语学习与翻译", "中文病句修改", "学习计划制定", "时间管理建议",
    "数学概念讲解", "统计学问题", "代码调试建议", "Git 使用问题",
    "论文阅读方法", "写作润色", "总结摘要", "简历与面试建议",
    "操作系统概念", "计算机网络概念", "数据库与 SQL", "Linux 命令使用",
    "大模型原理科普", "提示词工程", "RAG 检索增强", "模型评估方法",
    "学习焦虑疏导", "读书笔记方法", "背单词技巧", "编程项目点子",
    "开源社区参与", "技术文档写作", "API 设计问题", "正则表达式",
    "Pandas 数据处理", "Numpy 使用", "可视化建议", " Jupyter 使用技巧",
    "考研复习规划", "自学编程路线", "技术名词辨析", "生活常识问答",
]

SYS = "你是数据生成器。围绕给定主题,为中文学习助手「小K」编写一条训练数据。"

FEWSHOT = """示例1:
{"instruction": "解释一下什么是过拟合", "input": "", "output": "结论:过拟合是模型把训练数据的噪声也学了进去,导致在新数据上表现变差。\\n\\n1. 表现:训练集准确率很高,验证集明显更低。\\n2. 原因:模型过大、数据太少、训练轮次过多。\\n3. 缓解:增加数据、正则化、早停。"}

示例2:
{"instruction": "把这句话翻译成英文", "input": "实践是检验真理的唯一标准。", "output": "Practice is the sole criterion for testing truth."}
"""

PROMPT = """{fewshot}
要求:
1. 围绕主题「{topic}」编写一条全新的训练数据,不要复制示例。
2. 输出必须是单行 JSON,字段为 instruction(用户问题)、input(可选上下文,无则为空字符串)、output(小K的回答)。
3. 小K的回答风格:先给结论,再分点解释,必要时举例;中文为主,代码用英文。
4. 只输出 JSON,不要任何其他文字。

主题:{topic}
这是第 {idx} 次生成,请与常见内容错开角度,保证多样性:"""


def extract_json(text: str) -> dict | None:
    """从模型输出中提取第一个合法 JSON 对象。"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj.get("instruction"), str) and isinstance(obj.get("output"), str):
            return {
                "instruction": obj["instruction"].strip(),
                "input": str(obj.get("input") or "").strip(),
                "output": obj["output"].strip(),
            }
    except (json.JSONDecodeError, AttributeError):
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_lora.yaml")
    ap.add_argument("--n", type=int, default=400, help="目标生成条数")
    args = ap.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]

    out_path = ROOT / "data/raw/generated_instructions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑:统计已有条数与已有指令(用于去重)
    existing: set[str] = set()
    have = 0
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["instruction"])
                    have += 1
                except json.JSONDecodeError:
                    pass
    print(f"已有 {have} 条,目标 {args.n} 条", flush=True)
    if have >= args.n:
        return 0

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(mcfg["base_model"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["base_model"], dtype=torch.bfloat16, quantization_config=quant,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    rng = random.Random(42)
    topics = TOPICS * (args.n // len(TOPICS) + 2)
    rng.shuffle(topics)

    written, fails = have, 0
    with open(out_path, "a", encoding="utf-8") as fout, torch.no_grad():
        for i, topic in enumerate(topics, 1):
            if written >= args.n:
                break
            messages = [
                {"role": "system", "content": SYS},
                {"role": "user", "content": PROMPT.format(fewshot=FEWSHOT, topic=topic, idx=i)},
            ]
            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
            ).to(model.device)
            out = model.generate(
                **inputs, max_new_tokens=350, temperature=0.9, top_p=0.95,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            sample = extract_json(text)

            # 基础过滤:长度、去重
            if (not sample or sample["instruction"] in existing
                    or not 8 <= len(sample["instruction"]) <= 200
                    or not 30 <= len(sample["output"]) <= 1200):
                fails += 1
                continue
            existing.add(sample["instruction"])
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            if written % 20 == 0:
                print(f"[gen] {written}/{args.n} (失败 {fails})", flush=True)

    print(f"完成:共 {written} 条,失败跳过 {fails} 次", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
