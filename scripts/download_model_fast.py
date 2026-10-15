"""多线程分块下载 HuggingFace 模型(可断点续传)。

把每个 safetensors 分片切成 64MB 的块,16 线程并行下载到 parts/,
全部完成后拼接为最终文件并校验总大小。中断后重跑会自动跳过已完成的块。

用法:
  python scripts/download_model_fast.py            # 默认 Qwen2.5-14B-Instruct
  python scripts/download_model_fast.py --repo Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CHUNK = 16 * 1024 * 1024  # 16MB:连接易被重置,小块更稳
WORKERS = 16

SMALL_FILES = [
    "config.json", "generation_config.json", "merges.txt",
    "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json",
]


def remote_size(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def list_shards(repo: str) -> list[str]:
    api = f"https://huggingface.co/api/models/{repo}"
    r = requests.get(api, timeout=30)
    r.raise_for_status()
    return [s["rfilename"] for s in r.json()["siblings"] if s["rfilename"].endswith(".safetensors")]


def fetch_chunk(urls: list[str], start: int, end: int, dest: Path, retries: int = 12) -> str:
    if dest.exists() and dest.stat().st_size == end - start + 1:
        return "skip"
    for attempt in range(retries):
        url = urls[attempt % len(urls)]  # 重试时在主源/镜像源间轮换
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                              stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    for block in r.iter_content(1024 * 1024):
                        f.write(block)
            if tmp.stat().st_size != end - start + 1:
                raise IOError(f"chunk size mismatch: {tmp.stat().st_size}")
            tmp.replace(dest)
            return "ok"
        except Exception as e:  # noqa: BLE001
            print(f"  [retry {attempt + 1}/{retries}] {dest.name}: {e}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"chunk failed after {retries} retries: {dest}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    base = f"https://huggingface.co/{args.repo}/resolve/main"
    out_dir = ROOT / "models" / args.repo.split("/")[-1]
    parts_dir = out_dir / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(exist_ok=True)

    # 小文件直接下
    for name in SMALL_FILES:
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        r = requests.get(f"{base}/{name}", timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"[small] {name} ok", flush=True)

    shards = list_shards(args.repo)
    print(f"repo={args.repo}, {len(shards)} shards", flush=True)

    # 汇总所有待下载块(每块带主源 + 镜像两个地址)
    mirror = base.replace("huggingface.co", "hf-mirror.com")
    tasks = []
    for shard in shards:
        final = out_dir / shard
        url = f"{base}/{shard}"
        size = remote_size(url)
        if final.exists() and final.stat().st_size == size:
            print(f"[done] {shard} already complete", flush=True)
            continue
        urls = [url, f"{mirror}/{shard}"]
        n = (size + CHUNK - 1) // CHUNK
        for i in range(n):
            start = i * CHUNK
            end = min(start + CHUNK - 1, size - 1)
            tasks.append((urls, start, end, parts_dir / f"{shard}.part{i:04d}"))
    print(f"total chunks: {len(tasks)}", flush=True)

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_chunk, u, s, e, d): d for u, s, e, d in tasks}
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 8 == 0 or done == len(futs):
                gb = done * CHUNK / 1024 ** 3
                mins = (time.time() - t0) / 60
                print(f"[progress] {done}/{len(futs)} chunks (~{gb:.1f}GB) in {mins:.1f} min", flush=True)

    # 拼接并校验
    for shard in shards:
        final = out_dir / shard
        size = remote_size(f"{base}/{shard}")
        if final.exists() and final.stat().st_size == size:
            continue
        n = (size + CHUNK - 1) // CHUNK
        parts = [parts_dir / f"{shard}.part{i:04d}" for i in range(n)]
        for p in parts:
            if not p.exists():
                raise RuntimeError(f"missing part: {p}")
        with open(final, "wb") as f:
            for p in parts:
                f.write(p.read_bytes())
        if final.stat().st_size != size:
            raise RuntimeError(f"final size mismatch: {shard}")
        for p in parts:
            p.unlink()
        print(f"[merged] {shard} ({size / 1024**3:.2f}GB) OK", flush=True)

    parts_dir.rmdir()
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
