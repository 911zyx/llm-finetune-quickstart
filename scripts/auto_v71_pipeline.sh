#!/bin/bash
# v7.1 自动流水线:等偏好数据到 400 对 -> 备份旧 adapter -> 重训 DPO -> LLM-as-judge 评估
# 用法: nohup bash scripts/auto_v71_pipeline.sh > auto_v71.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1

echo "[wait] 等待 data/raw/dpo_pairs.jsonl 达到 400 对..."
while [ "$(wc -l < data/raw/dpo_pairs.jsonl)" -lt 400 ]; do
  sleep 60
done
echo "[wait] 已到 400 对,等 90 秒让生成进程退出并释放显存..."
sleep 90

# 备份 52 对时代的 DPO adapter
if [ -d outputs/dpo-lora/final ] && [ ! -d outputs/dpo-lora-v1-52pairs ]; then
  cp -r outputs/dpo-lora/final outputs/dpo-lora-v1-52pairs
  echo "[backup] 旧 DPO adapter 已备份到 outputs/dpo-lora-v1-52pairs"
fi

echo "[train] 开始 DPO 重训(2 epochs, 10% holdout)..."
./.venv/Scripts/python.exe scripts/train_dpo.py --config configs/sft_lora.yaml > train_dpo_v71.log 2>&1 \
  && echo "[train] 完成" || { echo "[train] 失败,见 train_dpo_v71.log"; echo "TRAIN_FAILED" > outputs/v71_status.txt; exit 1; }

echo "[judge] 开始 LLM-as-judge 评估..."
./.venv/Scripts/python.exe scripts/judge_eval.py --config configs/sft_lora.yaml > judge_eval_v71.log 2>&1 \
  && echo "[judge] 完成" || { echo "[judge] 失败,见 judge_eval_v71.log"; echo "JUDGE_FAILED" > outputs/v71_status.txt; exit 1; }

echo "ALL_DONE" > outputs/v71_status.txt
echo "[done] v7.1 流水线全部完成"
