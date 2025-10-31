from __future__ import annotations
import os, re, json, textwrap, asyncio
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import httpx
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from readability import Document
from markdownify import markdownify as md

load_dotenv()

app = FastAPI(title="Bassam Ultra-Answer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === إعدادات المزودات (LLM Providers) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# متغيرات اختيارية لبحث الويب أو مزودات بديلة لاحقًا
# مثال: SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

SYSTEM_PROMPT = (
    """
    أنت "Bassam Ultra-Answer" — مساعد عربي/إنجليزي خارق:
    - تفهم السؤال مهما كان مختصرًا.
    - تجيب بإيجاز أولاً، ثم تفاصيل عملية (خطوات/أمثلة/قوائم) عند الحاجة.
    - تلخص النصوص بشكل دقيق، وتستخرج النقاط، والتعليمات.
    - عند وجود روابط: تقرؤها أولًا وتستخلص أهم ما فيها قبل الإجابة.
    - إن طلب المستخدم كودًا: أعطِ كودًا نظيفًا جاهزًا للتنفيذ مع شرح قصير.
    - كن صريحًا عند عدم اليقين، واقترح طرق التحقق.
    - الأسلوب: بسيط، عملي، بدون زخرفة.
    """.strip()
)

URL_RE = re.compile(r"https?://\S+")

# ========= أدوات مساعدة =========
async def fetch_url(session: httpx.AsyncClient, url: str, timeout: int = 20) -> str:
    try:
        r = await session.get(url, timeout=timeout, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Bassam-UltraAnswer/1.0"
        })
        r.raise_for_status()
        html = r.text
        # قراءة المقالة النظيفة
        doc = Document(html)
        title = doc.short_title() or ""
        content_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(content_html, "lxml")
        # إزالة سكربت/ستايل
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
        content_md = md(text)
        return f"# {title}\n\n" + content_md.strip()
    except Exception as e:
        return f"(تعذر جلب الرابط: {e})"

async def call_openai(messages: List[Dict[str, str]], model: str = OPENAI_MODEL) -> str:
    if not OPENAI_API_KEY:
        # وضع بلا مفاتيح — جواب ذكي مبسّط بدوني LLM خارجي
        # (قواعد سريعة: يعيد صياغة السؤال ويعطي خطة عامة)
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        plan = textwrap.dedent(f"""
        (تشغيل مبسط بدون API)\n
        فهمي للسؤال:\n- {user[:500]}\n\nخطة إجابة عامة:\n1) تحديد المطلوب بدقة.\n2) عرض النقاط الأساسية بإيجاز.\n3) إعطاء خطوات عملية قابلة للتنفيذ.\n4) اقتراح مصادر للتحقق.\n\nالإجابة المختصرة:\n- {user[:200]} — يتطلب تفصيلاً حسب السياق.\n        """)
        return plan.strip()

    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "top_p": 0.9,
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

# ========= الراوتر الرئيسي =========
@app.post("/api/ask")
async def ask(req: Request):
    body = await req.json()
    query: str = body.get("query", "").strip()
    mode: str = body.get("mode", "auto")  # auto | summarize | code | chat

    if not query:
        return JSONResponse({"ok": True, "answer": "اكتب سؤالك…"})

    # 1) إن وُجدت روابط — نجلبها ونلخص
    urls = URL_RE.findall(query)
    gathered_contexts: List[str] = []
    if urls:
        async with httpx.AsyncClient() as session:
            tasks = [fetch_url(session, u) for u in urls[:3]]
            gathered_contexts = await asyncio.gather(*tasks)

    # 2) تجهيز الرسائل للموديل
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if gathered_contexts:
        ctx = "\n\n".join(gathered_contexts)
        messages.append({
            "role": "system",
            "content": (
                "السياق التالي جُمِع من الروابط التي أرسلها المستخدم. لخصه ثم أجب بدقة على سؤال المستخدم.\n\n" + ctx
            )
        })

    # أوضاع مخصصة
    if mode == "summarize":
        messages.append({"role": "user", "content": f"لخص بدقة وبنقاط واضحة:\n\n{query}"})
    elif mode == "code":
        messages.append({"role": "user", "content": f"اكتب كودًا نظيفًا مع شرح مختصر لحل المطلوب:\n\n{query}"})
    else:
        messages.append({"role": "user", "content": query})

    answer = await call_openai(messages)

    return JSONResponse({
        "ok": True,
        "answer": answer,
        "meta": {
            "urls": urls,
            "mode": mode,
            "provider": "openai-compatible" if OPENAI_API_KEY else "fallback-local",
        }
    })

# صفحة واجهة بسيطة (تصلح محليًا وعلى Render)
@app.get("/")
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
