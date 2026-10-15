"""用 Playwright 给 Gradio 演示界面截图(首页 + 对话效果)。"""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

QUESTION = "什么是 LoRA 微调?"

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

    # 等待回答流式输出完成(气泡内容稳定 6 秒视为完成)
    last, stable = "", 0
    for _ in range(120):
        page.wait_for_timeout(2000)
        texts = page.locator(".message, .bubble, [data-testid='bot'], .prose").all_inner_texts()
        cur = texts[-1] if texts else ""
        if cur == last:
            stable += 1
            if stable >= 3 and len(cur) > 50:
                break
        else:
            stable = 0
            last = cur
    page.wait_for_timeout(1000)
    page.screenshot(path=str(OUT / "02_chat.png"), full_page=False)
    print("chat shot ok, answer preview:", last[:80])
    browser.close()
