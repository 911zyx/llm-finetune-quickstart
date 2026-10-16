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

# ============ 编程能力预设 ============
CODING_TOPICS = [
    "Python 列表与字典操作", "Python 字符串处理", "写排序算法代码", "写二分查找代码",
    "调试 IndexError 报错", "调试 KeyError 报错", "调试类型错误 TypeError", "解释装饰器",
    "解释生成器与迭代器", "类与面向对象编程", "异常处理 try-except 写法", "文件读写代码",
    "正则表达式写法", "列表推导式技巧", "函数式编程 map/filter", "类型注解写法",
    "SQL 查询语句", "SQL 多表 JOIN", "pandas 数据清洗代码", "pandas 分组聚合",
    "numpy 数组运算", "Git 合并冲突解决", "Git 回滚与撤销操作", "Linux 常用命令",
    "写 Flask/FastAPI 接口", "requests 发 HTTP 请求", "爬虫基础代码", "单元测试写法",
    "代码性能优化建议", "LeetCode 解题思路", "动态规划入门题", "递归改写为循环",
    "代码审查与重构建议", "解释一段报错堆栈", "并发与多线程基础", "虚拟环境与 pip 依赖",
]

CODING_FEWSHOT = """示例1:
{"instruction": "写一段计算斐波那契数列的 Python 代码", "input": "", "output": "结论:推荐迭代写法,时间 O(n)、空间 O(1)。\\n\\n```python\\ndef fib(n: int) -> int:\\n    a, b = 0, 1\\n    for _ in range(n):\\n        a, b = b, a + b\\n    return a\\n```\\n\\n注意:递归写法有指数级重复计算,n 大时应避免。"}

示例2:
{"instruction": "帮我看看这段代码为什么报错", "input": "nums = [1, 2, 3]\\nprint(nums[3])", "output": "结论:索引越界,列表只有 3 个元素,最大下标是 2。\\n\\n1. 原因:Python 下标从 0 开始,nums[3] 超出了范围。\\n2. 修复:改为 nums[2],或用 nums[-1] 取最后一个元素。\\n3. 建议:访问前用 len(nums) 判断长度更稳妥。"}
"""

CODING_PROMPT = """{fewshot}
要求:
1. 围绕主题「{topic}」编写一条全新的编程类训练数据,不要复制示例。
2. 输出必须是单行 JSON,字段为 instruction(用户问题)、input(可选代码/报错上下文,无则为空字符串)、output(小K的回答)。
3. 小K的回答风格:先给结论,再分点解释;代码必须放在 ```python 代码块中,代码用英文写、注释用中文。
4. JSON 字符串内的换行必须写成 \\n。
5. 只输出 JSON,不要任何其他文字。

主题:{topic}
这是第 {idx} 次生成,请与常见内容错开角度,保证多样性:"""

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


def score_sample(sample: dict, preset: str) -> int:
    """规则打分(拒绝采样用):人设结构 + 代码块完整 + 长度适中 + 无截断。

    打分项全部可解释、可复现,满分 8 分。
    """
    out = sample["output"]
    s = 0
    if out.startswith("结论"):
        s += 2                                # 人设风格:结论先行
    if re.search(r"\n\s*1[.、]", out):
        s += 1                                # 分点结构
    if preset == "coding":
        fences = out.count("```")
        if fences >= 2 and fences % 2 == 0:
            s += 2                            # 完整闭合的代码块
        elif fences == 1:
            s -= 2                            # 代码块未闭合(截断),扣分
    n = len(out)
    if 200 <= n <= 900:
        s += 2                                # 长度甜区:内容充实但不冗长
    elif 100 <= n <= 1500:
        s += 1
    if out.rstrip().endswith(("。", "!", "?", "”", "」", "```", ".")):
        s += 1                                # 完整收尾(未被 max_tokens 截断)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_lora.yaml")
    ap.add_argument("--n", type=int, default=400, help="目标生成条数")
    ap.add_argument("--preset", choices=["general", "coding"], default="general",
                    help="general=通用主题,coding=编程能力主题(带代码块)")
    ap.add_argument("--candidates", type=int, default=1,
                    help="拒绝采样:每题生成 K 个候选,规则打分取最优(1=不启用)")
    args = ap.parse_args()

    if args.preset == "coding":
        topics_pool, fewshot, prompt_tpl = CODING_TOPICS, CODING_FEWSHOT, CODING_PROMPT
        out_name, max_new = "generated_coding.jsonl", 600
    else:
        topics_pool, fewshot, prompt_tpl = TOPICS, FEWSHOT, PROMPT
        out_name, max_new = "generated_instructions.jsonl", 350
    if args.candidates > 1:
        out_name = out_name.replace(".jsonl", "_rs.jsonl")  # 拒绝采样产物单独成文件

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]

    out_path = ROOT / "data/raw" / out_name
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
    topics = topics_pool * (args.n // len(topics_pool) + 2)
    rng.shuffle(topics)

    written, fails = have, 0
    scores: list[int] = []
    max_out = 2000 if args.preset == "coding" else 1200
    with open(out_path, "a", encoding="utf-8") as fout, torch.no_grad():
        for i, topic in enumerate(topics, 1):
            if written >= args.n:
                break
            messages = [
                {"role": "system", "content": SYS},
                {"role": "user", "content": prompt_tpl.format(fewshot=fewshot, topic=topic, idx=i)},
            ]
            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
            ).to(model.device)

            # 拒绝采样:生成 K 个候选,规则打分取最优;K=1 时退化为单次生成
            best, best_score = None, -999
            for _ in range(args.candidates):
                out = model.generate(
                    **inputs, max_new_tokens=max_new, temperature=0.9, top_p=0.95,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                cand = extract_json(text)
                # 基础过滤:长度、去重(编程回答含代码块,上限放宽到 2000)
                if (not cand or cand["instruction"] in existing
                        or not 8 <= len(cand["instruction"]) <= 200
                        or not 30 <= len(cand["output"]) <= max_out):
                    continue
                sc = score_sample(cand, args.preset)
                if sc > best_score:
                    best, best_score = cand, sc

            if best is None:
                fails += 1
                continue
            existing.add(best["instruction"])
            scores.append(best_score)
            fout.write(json.dumps(best, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            if written % 20 == 0:
                avg = sum(scores) / len(scores)
                print(f"[gen] {written}/{args.n} (失败 {fails}, 平均得分 {avg:.1f}/8)", flush=True)

    avg = sum(scores) / len(scores) if scores else 0
    print(f"完成:共 {written} 条,失败跳过 {fails} 次,平均得分 {avg:.1f}/8", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
