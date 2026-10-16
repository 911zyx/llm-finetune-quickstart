"""LLM-as-judge 自动评估:用基座模型当裁判,给基座/微调回答打分。

设计要点:
- 裁判 = 基座模型(Qwen2.5-14B-Instruct),与被评的微调模型同源,
  优点是零额外成本、风格维度它最懂;局限是同源偏好,README 中已注明。
- 绝对打分(1-10,四维度)而非两两比较,天然规避位置偏置。
- 裁判生成用贪心解码(do_sample=False)保证可复现。
- 三个加载阶段:基座作答 -> 微调作答 -> 裁判打分,每阶段用完即释放显存。

用法:
  python scripts/judge_eval.py --config configs/sft_lora.yaml
产物:
  outputs/judge/judge_scores.jsonl   每题每模型一行原始分数
  outputs/judge/judge_report.md      汇总报告(均值表 + 胜率)
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from datetime import datetime
from pathlib import Path

import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EVAL_PROMPTS, _generate_all, load_pair  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

DIMENSIONS = ["conclusion_first", "structure", "quality", "persona"]

JUDGE_TEMPLATE = """你是一位严格、公正的评审。请按以下四个维度给「回答」打分,每个维度 1-10 的整数:
1. conclusion_first:是否先给结论,再展开解释
2. structure:结构是否清晰(分点、分段、代码块完整)
3. quality:内容是否正确、完整、对提问者有用
4. persona:是否符合「耐心、严谨的中文学习助手小K」的人设(结论先行、分点解释、必要时举例)

只输出一行 JSON,不要输出任何其他内容:
{{"conclusion_first": x, "structure": x, "quality": x, "persona": x}}

【问题】
{question}

【回答】
{answer}"""


@torch.no_grad()
def judge_score(tokenizer, model, question: str, answer: str) -> dict | None:
    """让裁判模型给单个回答打分,返回四维分数;解析失败返回 None。"""
    messages = [{"role": "user", "content": JUDGE_TEMPLATE.format(question=question, answer=answer)}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    ).to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=120, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"\{[^{}]*\}", text)
    if not m:
        return None
    try:
        scores = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(k in scores for k in DIMENSIONS):
        return None
    return {k: int(scores[k]) for k in DIMENSIONS}


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-as-judge 自动评估")
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = ROOT / "outputs/judge"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 阶段 1/2:基座与微调分别作答(复用 evaluate.py 的生成流程,用完即释放显存)
    print("== 阶段 1:基座模型作答 ==")
    base_replies = _generate_all(cfg, with_adapter=False, name="基座模型")
    print("== 阶段 2:微调模型作答 ==")
    tuned_replies = _generate_all(cfg, with_adapter=True, name="微调后")

    # 阶段 3:裁判打分
    print("== 阶段 3:裁判打分(基座模型,贪心解码) ==")
    tokenizer, judge = load_pair(cfg, with_adapter=False)
    records = []
    fails = 0
    for prompt, rb, rt in zip(EVAL_PROMPTS, base_replies, tuned_replies):
        for tag, answer in (("base", rb), ("tuned", rt)):
            scores = judge_score(tokenizer, judge, prompt, answer)
            if scores is None:
                fails += 1
                print(f"[WARN] 分数解析失败:{prompt[:20]}... ({tag})")
                continue
            scores["total"] = sum(scores.values())
            records.append({"question": prompt, "model": tag, "answer": answer, **scores})
            print(f"  {tag:5s} | {prompt[:18]:<20s} | total={scores['total']:2d}")
    del judge, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # 汇总
    with open(out_dir / "judge_scores.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def avg(tag: str, dim: str) -> float:
        vals = [r[dim] for r in records if r["model"] == tag]
        return sum(vals) / max(len(vals), 1)

    wins = sum(
        1 for q in EVAL_PROMPTS
        if sum(r["total"] for r in records if r["model"] == "tuned" and r["question"] == q)
           > sum(r["total"] for r in records if r["model"] == "base" and r["question"] == q)
    )
    n_q = len(EVAL_PROMPTS)

    lines = [
        "# LLM-as-judge 自动评估报告",
        "",
        f"- 时间:{datetime.now():%Y-%m-%d %H:%M}",
        f"- 题目数:{n_q}(人设/知识/排查/翻译/文案/编程/调试 七类)",
        f"- 裁判:Qwen2.5-14B-Instruct 基座(4bit,贪心解码,绝对打分 1-10)",
        f"- 解析失败:{fails} 次",
        "",
        "| 维度 | 基座 | 微调后(SFT+DPO) | 差值 |",
        "|---|---|---|---|",
    ]
    dim_names = {"conclusion_first": "结论先行", "structure": "结构清晰",
                 "quality": "内容质量", "persona": "人设一致", "total": "总分"}
    for dim in DIMENSIONS + ["total"]:
        b, t = avg("base", dim), avg("tuned", dim)
        lines.append(f"| {dim_names[dim]} | {b:.1f} | {t:.1f} | {t - b:+.1f} |")
    lines += [
        "",
        f"**微调后胜率:{wins}/{n_q}**(按每题总分比较)",
        "",
        "> 局限:裁判与被评模型同源(同一基座),存在同源偏好;结论需与人工抽检互相印证。",
    ]
    report = "\n".join(lines)
    (out_dir / "judge_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n报告已保存到 {out_dir / 'judge_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
