#!/usr/bin/env bash
# 并行断点续传下载 Qwen2.5-14B-Instruct 到 models/ 目录
# 用法: bash scripts/download_model.sh   (可反复运行,自动续传)
set -u
REPO="Qwen/Qwen2.5-14B-Instruct"
BASE="https://huggingface.co/${REPO}/resolve/main"
DIR="$(cd "$(dirname "$0")/.." && pwd)/models/Qwen2.5-14B-Instruct"
mkdir -p "$DIR"

SMALL=(config.json generation_config.json merges.txt model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json)
SHARDS=(model-00001-of-00008.safetensors model-00002-of-00008.safetensors model-00003-of-00008.safetensors model-00004-of-00008.safetensors model-00005-of-00008.safetensors model-00006-of-00008.safetensors model-00007-of-00008.safetensors model-00008-of-00008.safetensors)

for f in "${SMALL[@]}"; do
  [ -s "$DIR/$f" ] || curl -sL -C - -o "$DIR/$f" "$BASE/$f" || true
done

# 8 个分片全部并行,单连接限速时聚合带宽更高
for f in "${SHARDS[@]}"; do
  (
    # 直到大小与远端一致才停止
    for attempt in $(seq 1 60); do
      remote=$(curl -sIL "$BASE/$f" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
      local_size=$(stat -c "%s" "$DIR/$f" 2>/dev/null || echo 0)
      if [ -n "$remote" ] && [ "$local_size" = "$remote" ]; then
        echo "[OK] $f ($local_size bytes)"
        break
      fi
      curl -sL -C - -o "$DIR/$f" --max-time 280 "$BASE/$f" || true
    done
  ) &
done
wait
echo "---- summary ----"
du -sh "$DIR"
ls -la "$DIR"
