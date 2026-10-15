"""用 Playwright 给 Gradio 演示界面截图(首页 + 对话效果)。

用法:
  python scripts/take_screenshots.py                      # 默认通用问答截图
  python scripts/take_screenshots.py "你的问题" 03_code    # 自定义问题与文件名前缀
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "什么是 LoRA 微调?"
CHAT_NAME = sys.argv[2] if len(sys.argv) > 2 else "02_chat"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 860})
    page.goto("http://127.0.0.1:7860/", wait_until="domcontentloaded")
    page.wait_for_selector("textarea", timeout=60000)
    page.wait_for_timeout(5000)
    page.screenshot(path=str(OUT / "01_home.png"))
    print("home shot ok")

    # 输入问题并提交
    box = page.locator("textarea").first
    box.fill(QUESTION)
    box.press("Enter")

    # 固定等待生成完成(14B 4bit 生成 256 token 约 30~60 秒)
    page.wait_for_timeout(75000)
    page.screenshot(path=str(OUT / f"{CHAT_NAME}.png"), full_page=False)
    print("chat shot ok")
    browser.close()
