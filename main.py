"""
OmniCloud AI Agent v2 — Advanced Telegram AI Assistant
Controls 7 cloud platforms. Multi-step reasoning, 20+ tools, persistent memory.
Inspired by Viktor AI & Manus AI.
"""
import asyncio
import json
import logging
import os
import re
import sys
import io
import time
import base64
import traceback
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters,
)
from telegram.constants import ParseMode, ChatAction

# ── Config ──
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
HF_TOKEN = os.environ["HF_TOKEN"]
OWNER_CHAT_ID = int(os.environ.get("OWNER_CHAT_ID", "5245619457"))
NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY", "")
FLY_TOKEN = os.environ.get("FLY_TOKEN", "")
RENDER_KEY = os.environ.get("RENDER_KEY", "")
RAILWAY_TOKEN = os.environ.get("RAILWAY_TOKEN", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
BACK4APP_TOKEN = os.environ.get("BACK4APP_TOKEN", "")
NORTHFLANK_TOKEN = os.environ.get("NORTHFLANK_TOKEN", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
PORT = int(os.environ.get("PORT", "7860"))

HF_API = "https://router.huggingface.co/v1/chat/completions"
NVIDIA_API = "https://integrate.api.nvidia.com/v1/chat/completions"
MAX_HISTORY = 30
MAX_TOOL_ROUNDS = 6
DATA_DIR = Path("/tmp/omnicloud_data")
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("omnicloud")

# ══════════════════════════════════════
# STATE
# ══════════════════════════════════════
conversations: dict[int, list] = {}
user_memory: dict[int, dict] = {}
agent_stats = {"started_at": datetime.now(timezone.utc).isoformat(), "messages": 0, "tools": 0, "errors": 0}

MODELS = {
    "qwen72b": ("Qwen/Qwen2.5-72B-Instruct", "Qwen 72B 🧠"),
    "llama70b": ("meta-llama/Llama-3.3-70B-Instruct", "Llama 70B 🦙"),
    "qwen7b": ("Qwen/Qwen2.5-7B-Instruct", "Qwen 7B ⚡"),
    "llama8b": ("meta-llama/Llama-3.1-8B-Instruct", "Llama 8B ⚡⚡"),
}

# ══════════════════════════════════════
# HEALTH SERVER
# ══════════════════════════════════════
WEBHOOK_URL = "https://ayoub5550-omni-cloud-agent.hf.space/webhook"
WEBHOOK_PATH = "/webhook"

# ══════════════════════════════════════
# LLM ENGINE — Parallel model calls for speed
# ══════════════════════════════════════

async def llm_call(messages: list, model: str = None, temperature: float = 0.6, max_tokens: int = 2500) -> str:
    model = model or LLM_MODEL
    if model.startswith("meta/") and NVIDIA_KEY:
        url = NVIDIA_API
        headers = {"Authorization": f"Bearer {NVIDIA_KEY}", "Content-Type": "application/json"}
    else:
        url = HF_API
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    
    async with httpx.AsyncClient(timeout=90) as client:
        for attempt in range(2):
            try:
                r = await client.post(url, headers=headers, json=payload)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"] or ""
                if r.status_code in (503, 529, 429):
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                log.error(f"LLM {r.status_code}: {r.text[:200]}")
                if model != "meta-llama/Llama-3.3-70B-Instruct":
                    return await llm_call(messages, "meta-llama/Llama-3.3-70B-Instruct", temperature, max_tokens)
                return f"⚠️ خطأ ({r.status_code})"
            except Exception as e:
                log.error(f"LLM err: {e}")
                if attempt == 1 and model != "meta-llama/Llama-3.3-70B-Instruct":
                    return await llm_call(messages, "meta-llama/Llama-3.3-70B-Instruct", temperature, max_tokens)
                await asyncio.sleep(2)
    return "⚠️ AI غير متاح"


# Fast model for simple routing/classification
async def llm_fast(prompt: str) -> str:
    return await llm_call(
        [{"role": "user", "content": prompt}],
        model="Qwen/Qwen2.5-7B-Instruct", temperature=0.1, max_tokens=500
    )


SYSTEM_PROMPT = """أنت OmniCloud AI v2 — وكيل ذكاء اصطناعي متقدم يتحكم في 7 منصات سحابية.
أنت مثل JARVIS — ذكي، سريع، مبدع، عملي.

## شخصيتك:
- تتكلم بالعربية إذا تكلم المستخدم بالعربية، وبالإنجليزية إذا بالإنجليزية
- مختصر ومفيد — لا تكرر ولا تشرح كثيراً
- تستخدم الأدوات تلقائياً عند الحاجة (بدون استئذان)
- تستخدم إيموجي بذكاء

## الأدوات — لاستخدام أداة أجب بـ:
```tool
{{"action": "TOOL_NAME", "params": {{...}}}}
```

### أدوات عامة:
1. `python` — تنفيذ كود: {{"code": "..."}}
2. `shell` — أمر نظام: {{"cmd": "..."}}
3. `search` — بحث إنترنت: {{"query": "..."}}
4. `browse` — تصفح صفحة: {{"url": "..."}}
5. `generate_image` — صورة AI: {{"prompt": "english desc"}}
6. `create_file` — إنشاء ملف: {{"filename": "...", "content": "...", "caption": "..."}}
7. `qr_code` — QR: {{"data": "..."}}
8. `translate` — ترجمة: {{"text": "...", "to": "ar/en/fr"}}
9. `weather` — طقس: {{"city": "..."}}
10. `wiki` — ويكيبيديا: {{"query": "...", "lang": "ar"}}
11. `calc` — حساب: {{"expression": "..."}}
12. `remember` — حفظ: {{"key": "...", "value": "..."}}
13. `recall` — استرجاع: {{"key": ""}} (فارغ = الكل)
14. `tts` — نص→صوت: {{"text": "..."}}

### أدوات المنصات السحابية:
15. `fly_list` — عرض تطبيقات Fly.io: {{}}
16. `fly_status` — حالة تطبيق: {{"app": "name"}}
17. `fly_scale` — تحجيم: {{"app": "name", "action": "start/stop"}}
18. `fly_logs` — سجلات: {{"app": "name"}}
19. `github_repos` — مستودعات GitHub: {{}}
20. `github_create_repo` — إنشاء مستودع: {{"name": "...", "private": true}}
21. `github_push_file` — رفع ملف: {{"repo": "...", "path": "...", "content": "...", "message": "..."}}
22. `hf_spaces` — عرض Spaces: {{}}
23. `hf_space_status` — حالة Space: {{"space": "owner/name"}}
24. `hf_space_restart` — إعادة تشغيل: {{"space": "owner/name"}}
25. `render_services` — خدمات Render: {{}}
26. `railway_status` — حالة Railway: {{}}
27. `platform_overview` — ملخص كل المنصات: {{}}

## ذاكرة المستخدم:
{memory}

الآن: {date}
أجب مباشرة أو استخدم أداة. لا تشرح الأداة قبل استخدامها."""


# ══════════════════════════════════════
# TOOLS — General
# ══════════════════════════════════════

async def tool_python(p, cid, ctx):
    code = p.get("code", "")
    if not code.strip(): return {"text": "❌ لا يوجد كود"}
    fpath = DATA_DIR / f"run_{cid}.py"
    fpath.write_text(code)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(fpath), stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=str(DATA_DIR))
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        out = stdout.decode()[:4000]
        if stderr.decode().strip(): out += f"\n⚠️ {stderr.decode()[:800]}"
        # Check for output files
        files = [str(f) for f in DATA_DIR.glob(f"output_{cid}_*") if f.stat().st_mtime > time.time() - 5]
        return {"text": out or "✅ تم", "files": files} if files else {"text": out or "✅ تم (لا مخرجات)"}
    except asyncio.TimeoutError: return {"text": "⏱ timeout 30s"}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_shell(p, cid, ctx):
    cmd = p.get("cmd", "")
    blocked = ["rm -rf /", "mkfs", "dd if=/dev", ":(){ ", "> /dev/sd"]
    if any(b in cmd for b in blocked): return {"text": "🚫 محظور"}
    try:
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=str(DATA_DIR))
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        out = stdout.decode()[:4000]
        if stderr.decode().strip(): out += f"\n[stderr]: {stderr.decode()[:800]}"
        return {"text": out or "(empty)"}
    except asyncio.TimeoutError: return {"text": "⏱ timeout"}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_search(p, cid, ctx):
    q = p.get("query", "")
    try:
        async with httpx.AsyncClient(timeout=12) as c:
            r = await c.post("https://html.duckduckgo.com/html/", data={"q": q},
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120"})
            results = []
            for m in re.finditer(r'class="result__a"[^>]*>(.*?)</a>.*?class="result__snippet">(.*?)</(?:a|td)', r.text, re.DOTALL):
                t = re.sub(r'<[^>]+>', '', m.group(1)).strip()
                s = re.sub(r'<[^>]+>', '', m.group(2)).strip()
                if t: results.append(f"• *{t}*: {s}")
            return {"text": "\n".join(results[:6]) or "لا نتائج"}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_browse(p, cid, ctx):
    url = p.get("url", "")
    if not url.startswith("http"): url = "https://" + url
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
            r = await c.get(url, headers={"User-Agent": "Mozilla/5.0 Chrome/120"})
            title = ""
            tm = re.search(r'<title[^>]*>(.*?)</title>', r.text, re.DOTALL|re.I)
            if tm: title = re.sub(r'\s+', ' ', tm.group(1)).strip()
            text = r.text
            for tag in ['script','style','nav','footer','header','aside','noscript']:
                text = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', text, flags=re.DOTALL|re.I)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return {"text": f"📄 *{title}*\n\n{text[:4000]}" if title else text[:4000]}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_generate_image(p, cid, ctx):
    prompt = p.get("prompt", "")
    for model in ["black-forest-labs/FLUX.1-schnell", "stabilityai/stable-diffusion-xl-base-1.0"]:
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(f"https://router.huggingface.co/hf-inference/models/{model}",
                    headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
                    json={"inputs": prompt, "parameters": {"num_inference_steps": 4}})
                if r.status_code == 200 and r.headers.get("content-type","").startswith("image"):
                    fp = DATA_DIR / f"img_{cid}_{int(time.time())}.png"
                    fp.write_bytes(r.content)
                    return {"text": f"🎨 {prompt[:100]}", "files": [str(fp)]}
        except: pass
    return {"text": "⚠️ توليد الصورة فشل — جرب لاحقاً"}

async def tool_create_file(p, cid, ctx):
    fn = p.get("filename", "file.txt")
    fp = DATA_DIR / f"send_{cid}_{fn}"
    fp.write_text(p.get("content", ""))
    return {"text": p.get("caption", f"📁 {fn}"), "files": [str(fp)]}

async def tool_qr_code(p, cid, ctx):
    data = p.get("data", "")
    code = f"""
import sys, subprocess
subprocess.check_call([sys.executable,'-m','pip','install','qrcode[pil]','-q'])
import qrcode; qrcode.make("{data}").save("{DATA_DIR}/qr_{cid}.png")
"""
    fp = DATA_DIR / f"qr_gen.py"
    fp.write_text(code)
    proc = await asyncio.create_subprocess_exec(sys.executable, str(fp),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    await asyncio.wait_for(proc.communicate(), timeout=30)
    qp = DATA_DIR / f"qr_{cid}.png"
    if qp.exists(): return {"text": "📱 QR Code", "files": [str(qp)]}
    return {"text": "❌ فشل QR"}

async def tool_translate(p, cid, ctx):
    return {"text": await llm_call([
        {"role":"system","content":f"Translate to {p.get('to','en')}. Output ONLY the translation."},
        {"role":"user","content":p.get("text","")}
    ], temperature=0.2, max_tokens=2000)}

async def tool_weather(p, cid, ctx):
    city = p.get("city","")
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(f"https://wttr.in/{city}?format=j1")
            if r.status_code == 200:
                d = r.json(); cur = d["current_condition"][0]
                area = d.get("nearest_area",[{}])[0]
                name = area.get("areaName",[{}])[0].get("value",city)
                return {"text": f"🌤 *{name}*\n🌡 {cur['temp_C']}°C (إحساس {cur['FeelsLikeC']}°C)\n💧 رطوبة {cur['humidity']}%\n💨 رياح {cur['windspeedKmph']} كم/س\n☁️ {cur['weatherDesc'][0]['value']}"}
    except Exception as e: return {"text": f"❌ {e}"}
    return {"text": "⚠️ لم أجد"}

async def tool_wiki(p, cid, ctx):
    q = p.get("query",""); lang = p.get("lang","ar")
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{q}")
            if r.status_code == 200:
                d = r.json()
                return {"text": f"📚 *{d.get('title',q)}*\n\n{d.get('extract','')[:2000]}"}
            r2 = await c.get(f"https://{lang}.wikipedia.org/w/api.php",
                params={"action":"query","list":"search","srsearch":q,"format":"json","srlimit":3})
            if r2.status_code == 200:
                res = r2.json().get("query",{}).get("search",[])
                if res: return {"text": "\n".join(f"• *{r['title']}*: {re.sub(r'<[^>]+>','',r.get('snippet',''))}" for r in res)}
    except Exception as e: return {"text": f"❌ {e}"}
    return {"text": "لا نتائج"}

async def tool_calc(p, cid, ctx):
    expr = p.get("expression","")
    code = f"import math; print(eval('''{expr}''', {{'__builtins__':{{}}, 'math':math, 'sqrt':math.sqrt, 'sin':math.sin, 'cos':math.cos, 'tan':math.tan, 'log':math.log, 'pi':math.pi, 'e':math.e, 'abs':abs, 'pow':pow, 'round':round}}))"
    fp = DATA_DIR / f"calc.py"; fp.write_text(code)
    try:
        proc = await asyncio.create_subprocess_exec(sys.executable, str(fp),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(proc.communicate(), timeout=10)
        r = out.decode().strip()
        if err.decode().strip(): return {"text": f"❌ {err.decode()[:200]}"}
        return {"text": f"🔢 `{expr}` = *{r}*"}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_remember(p, cid, ctx):
    k, v = p.get("key","note"), p.get("value","")
    if cid not in user_memory: user_memory[cid] = {}
    user_memory[cid][k] = {"value": v, "t": datetime.now().isoformat()}
    (DATA_DIR / f"mem_{cid}.json").write_text(json.dumps(user_memory[cid], ensure_ascii=False))
    return {"text": f"💾 حفظت *{k}*"}

async def tool_recall(p, cid, ctx):
    k = p.get("key","")
    if cid not in user_memory:
        mf = DATA_DIR / f"mem_{cid}.json"
        user_memory[cid] = json.loads(mf.read_text()) if mf.exists() else {}
    mem = user_memory.get(cid, {})
    if k and k in mem: return {"text": f"💭 *{k}:* {mem[k]['value']}"}
    if not k and mem: return {"text": "💭 *الذاكرة:*\n" + "\n".join(f"• *{k}:* {v['value'][:80]}" for k,v in mem.items())}
    return {"text": "📭 فارغة" if not k else f"❓ '{k}' غير موجود"}

async def tool_tts(p, cid, ctx):
    text = p.get("text","")[:500]
    for model in ["facebook/mms-tts-ara", "facebook/mms-tts-eng"]:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(f"https://router.huggingface.co/hf-inference/models/{model}",
                    headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
                    json={"inputs": text})
                if r.status_code == 200 and len(r.content) > 1000:
                    fp = DATA_DIR / f"tts_{cid}.wav"; fp.write_bytes(r.content)
                    return {"text": "🔊", "voice": str(fp)}
        except: pass
    return {"text": "⚠️ الصوت غير متاح"}


# ══════════════════════════════════════
# TOOLS — CLOUD PLATFORMS (Real API calls)
# ══════════════════════════════════════

async def _fly_gql(query: str, variables: dict = None) -> dict:
    if not FLY_TOKEN: return {"error": "Fly.io غير متصل"}
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post("https://api.fly.io/graphql",
            headers={"Authorization": f"Bearer {FLY_TOKEN}", "Content-Type": "application/json"},
            json={"query": query, "variables": variables or {}})
        return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}

async def _fly_machines(app: str, method: str = "GET", path: str = "", json_data=None) -> dict:
    if not FLY_TOKEN: return {"error": "Fly.io غير متصل"}
    url = f"https://api.machines.dev/v1/apps/{app}{path}"
    async with httpx.AsyncClient(timeout=15) as c:
        if method == "GET":
            r = await c.get(url, headers={"Authorization": f"Bearer {FLY_TOKEN}"})
        else:
            r = await c.request(method, url, headers={"Authorization": f"Bearer {FLY_TOKEN}", "Content-Type": "application/json"}, json=json_data)
        return r.json() if r.status_code in (200, 201) else {"error": f"HTTP {r.status_code}: {r.text[:200]}"}

async def tool_fly_list(p, cid, ctx):
    data = await _fly_gql('{ apps { nodes { name status hostname currentRelease { status } } } }')
    if "error" in data: return {"text": f"❌ {data['error']}"}
    apps = data.get("data",{}).get("apps",{}).get("nodes",[])
    if not apps: return {"text": "📭 لا تطبيقات"}
    lines = []
    for a in apps:
        st = "🟢" if a["status"] == "deployed" else "🟡" if a["status"] == "pending" else "🔴"
        lines.append(f"{st} *{a['name']}* — {a['status']}")
    return {"text": f"✈️ *Fly.io — {len(apps)} تطبيقات:*\n" + "\n".join(lines)}

async def tool_fly_status(p, cid, ctx):
    app = p.get("app","")
    machines = await _fly_machines(app, path="/machines")
    if isinstance(machines, dict) and "error" in machines: return {"text": f"❌ {machines['error']}"}
    if not machines: return {"text": f"📭 {app}: لا machines"}
    lines = []
    for m in machines:
        st = "🟢" if m["state"] == "started" else "🔴"
        region = m.get("region","?")
        cpu = m.get("config",{}).get("guest",{})
        lines.append(f"{st} `{m['id'][:12]}` | {m['state']} | {region} | {cpu.get('cpus','?')}CPU/{cpu.get('memory_mb','?')}MB")
    return {"text": f"✈️ *{app}:*\n" + "\n".join(lines)}

async def tool_fly_scale(p, cid, ctx):
    app = p.get("app",""); action = p.get("action","status")
    machines = await _fly_machines(app, path="/machines")
    if isinstance(machines, dict) and "error" in machines: return {"text": f"❌ {machines['error']}"}
    results = []
    for m in machines:
        mid = m["id"]
        if action == "stop":
            r = await _fly_machines(app, "POST", f"/machines/{mid}/stop")
            results.append(f"⏹ {mid[:10]}: stopped")
        elif action == "start":
            r = await _fly_machines(app, "POST", f"/machines/{mid}/start")
            results.append(f"▶️ {mid[:10]}: started")
    return {"text": f"✈️ *{app}* — {action}:\n" + "\n".join(results) if results else "لا machines"}

async def tool_fly_logs(p, cid, ctx):
    app = p.get("app","")
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"https://api.machines.dev/v1/apps/{app}/machines",
                headers={"Authorization": f"Bearer {FLY_TOKEN}"})
            if r.status_code == 200:
                machines = r.json()
                if machines:
                    events = machines[0].get("events", [])[:5]
                    lines = [f"• {e['type']}: {e['status']} ({e.get('timestamp','')})" for e in events]
                    return {"text": f"📋 *{app} logs:*\n" + "\n".join(lines)}
        return {"text": f"📭 لا سجلات لـ {app}"}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_github_repos(p, cid, ctx):
    if not GITHUB_TOKEN: return {"text": "❌ GitHub غير متصل"}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.github.com/user/repos?sort=updated&per_page=10",
                headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"})
            if r.status_code == 200:
                repos = r.json()
                lines = [f"{'🔒' if r['private'] else '📂'} *{r['name']}* — ⭐{r.get('stargazers_count',0)} | {r.get('language','?')}" for r in repos]
                return {"text": f"🐙 *GitHub — {len(repos)} مستودعات:*\n" + "\n".join(lines)}
        return {"text": "❌ فشل"}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_github_create_repo(p, cid, ctx):
    if not GITHUB_TOKEN: return {"text": "❌ GitHub غير متصل"}
    name = p.get("name",""); private = p.get("private", True)
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post("https://api.github.com/user/repos",
                headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"},
                json={"name": name, "private": private, "auto_init": True})
            if r.status_code == 201:
                d = r.json()
                return {"text": f"✅ تم إنشاء *{d['full_name']}* {'🔒' if private else '📂'}\n🔗 {d['html_url']}"}
        return {"text": f"❌ فشل: {r.status_code}"}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_github_push_file(p, cid, ctx):
    if not GITHUB_TOKEN: return {"text": "❌ GitHub غير متصل"}
    repo = p.get("repo",""); path = p.get("path",""); content = p.get("content",""); msg = p.get("message","Update")
    try:
        encoded = base64.b64encode(content.encode()).decode()
        async with httpx.AsyncClient(timeout=10) as c:
            # Check if file exists
            r = await c.get(f"https://api.github.com/repos/{repo}/contents/{path}",
                headers={"Authorization": f"token {GITHUB_TOKEN}"})
            sha = r.json().get("sha") if r.status_code == 200 else None
            data = {"message": msg, "content": encoded}
            if sha: data["sha"] = sha
            r2 = await c.put(f"https://api.github.com/repos/{repo}/contents/{path}",
                headers={"Authorization": f"token {GITHUB_TOKEN}"},
                json=data)
            if r2.status_code in (200, 201):
                return {"text": f"✅ رفع `{path}` → *{repo}*"}
        return {"text": f"❌ فشل: {r2.status_code}"}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_hf_spaces(p, cid, ctx):
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://huggingface.co/api/spaces?author=ayoub5550",
                headers={"Authorization": f"Bearer {HF_TOKEN}"})
            if r.status_code == 200:
                spaces = r.json()
                lines = []
                for s in spaces[:10]:
                    name = s.get("id","?")
                    sdk = s.get("sdk","?")
                    lines.append(f"• *{name}* ({sdk})")
                return {"text": f"🤗 *HuggingFace — {len(spaces)} Spaces:*\n" + "\n".join(lines)}
        return {"text": "❌ فشل"}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_hf_space_status(p, cid, ctx):
    space = p.get("space","")
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"https://huggingface.co/api/spaces/{space}",
                headers={"Authorization": f"Bearer {HF_TOKEN}"})
            if r.status_code == 200:
                d = r.json(); rt = d.get("runtime",{})
                stage = rt.get("stage","?")
                hw = rt.get("hardware",{})
                st = "🟢" if stage == "RUNNING" else "🔴"
                return {"text": f"{st} *{space}*\nالحالة: {stage}\nHardware: {hw.get('current','?')}\nSDK: {d.get('sdk','?')}"}
        return {"text": f"❌ لم أجد {space}"}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_hf_space_restart(p, cid, ctx):
    space = p.get("space","")
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"https://huggingface.co/api/spaces/{space}/restart",
                headers={"Authorization": f"Bearer {HF_TOKEN}"})
            return {"text": f"🔄 تم إعادة تشغيل *{space}*" if r.status_code == 200 else f"❌ {r.status_code}"}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_render_services(p, cid, ctx):
    if not RENDER_KEY: return {"text": "❌ Render غير متصل"}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.render.com/v1/services?limit=10",
                headers={"Authorization": f"Bearer {RENDER_KEY}"})
            if r.status_code == 200:
                svcs = r.json()
                if not svcs: return {"text": "📭 لا خدمات على Render"}
                lines = [f"• *{s.get('service',{}).get('name','?')}* — {s.get('service',{}).get('type','?')} | {s.get('service',{}).get('serviceDetails',{}).get('region','?')}" for s in svcs]
                return {"text": f"🎨 *Render — {len(svcs)} خدمات:*\n" + "\n".join(lines)}
        return {"text": "❌ فشل"}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_railway_status(p, cid, ctx):
    if not RAILWAY_TOKEN: return {"text": "❌ Railway غير متصل"}
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post("https://backboard.railway.com/graphql/v2",
                headers={"Authorization": f"Bearer {RAILWAY_TOKEN}", "Content-Type": "application/json"},
                json={"query": "{ me { name email } projects { edges { node { name id } } } }"})
            if r.status_code == 200:
                d = r.json().get("data",{})
                me = d.get("me",{})
                projs = d.get("projects",{}).get("edges",[])
                lines = [f"• *{p['node']['name']}*" for p in projs[:10]]
                return {"text": f"🚂 *Railway — {me.get('name','?')}*\n" + ("\n".join(lines) if lines else "لا مشاريع")}
        return {"text": "❌ فشل"}
    except Exception as e: return {"text": f"❌ {e}"}

async def tool_platform_overview(p, cid, ctx):
    """Run all platform checks in parallel for speed."""
    tasks = []
    platforms = [
        ("fly_list", tool_fly_list),
        ("github_repos", tool_github_repos),
        ("hf_spaces", tool_hf_spaces),
    ]
    if RENDER_KEY: platforms.append(("render_services", tool_render_services))
    if RAILWAY_TOKEN: platforms.append(("railway_status", tool_railway_status))
    
    results = await asyncio.gather(*[fn({}, cid, ctx) for _, fn in platforms], return_exceptions=True)
    
    lines = []
    for i, (name, _) in enumerate(platforms):
        r = results[i]
        if isinstance(r, Exception):
            lines.append(f"❌ *{name}:* خطأ")
        else:
            lines.append(r["text"])
        lines.append("")
    
    return {"text": "🌐 *ملخص كل المنصات:*\n━━━━━━━━━━━━━━━\n\n" + "\n".join(lines)}


# Tool registry
TOOLS = {
    "python": tool_python, "shell": tool_shell, "search": tool_search,
    "browse": tool_browse, "generate_image": tool_generate_image,
    "create_file": tool_create_file, "qr_code": tool_qr_code,
    "translate": tool_translate, "weather": tool_weather, "wiki": tool_wiki,
    "calc": tool_calc, "remember": tool_remember, "recall": tool_recall,
    "tts": tool_tts,
    "fly_list": tool_fly_list, "fly_status": tool_fly_status,
    "fly_scale": tool_fly_scale, "fly_logs": tool_fly_logs,
    "github_repos": tool_github_repos, "github_create_repo": tool_github_create_repo,
    "github_push_file": tool_github_push_file,
    "hf_spaces": tool_hf_spaces, "hf_space_status": tool_hf_space_status,
    "hf_space_restart": tool_hf_space_restart,
    "render_services": tool_render_services, "railway_status": tool_railway_status,
    "platform_overview": tool_platform_overview,
}


# ══════════════════════════════════════
# AGENT LOOP
# ══════════════════════════════════════

def parse_tool_call(text: str):
    m = re.search(r'```(?:tool|json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not m: m = re.search(r'(\{[^{}]*"action"\s*:[^{}]*\})', text, re.DOTALL)
    if not m: return None
    try:
        call = json.loads(m.group(1))
        action = call.get("action") or call.get("tool")
        params = call.get("params") or call.get("args") or {}
        if action and action in TOOLS: return (action, params)
    except: pass
    return None

def get_memory_str(cid):
    mem = user_memory.get(cid, {})
    if not mem:
        mf = DATA_DIR / f"mem_{cid}.json"
        if mf.exists():
            try: mem = json.loads(mf.read_text()); user_memory[cid] = mem
            except: pass
    if not mem: return "لا ملاحظات"
    return "\n".join(f"- {k}: {v['value'][:80]}" for k,v in list(mem.items())[:10])


async def agent_loop(user_msg, chat_id, ctx, update, image_data=None):
    if chat_id not in conversations: conversations[chat_id] = []
    history = conversations[chat_id]
    
    sys_prompt = SYSTEM_PROMPT.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        memory=get_memory_str(chat_id),
    )
    
    user_content = user_msg
    if image_data: user_content = f"[صورة مرفقة]\n{user_msg}" if user_msg else "[صورة — حللها]"
    
    messages = [{"role":"system","content":sys_prompt}] + history[-MAX_HISTORY:] + [{"role":"user","content":user_content}]
    
    for round_num in range(MAX_TOOL_ROUNDS):
        response = await llm_call(messages)
        tool_call = parse_tool_call(response)
        
        if not tool_call:
            # Final answer
            clean = re.sub(r'```(?:tool|json)\s*\{[^}]*"action"[^}]*\}\s*```', '', response).strip() or response
            history.append({"role":"user","content":user_content})
            history.append({"role":"assistant","content":clean})
            if len(history) > MAX_HISTORY*2: history[:] = history[-MAX_HISTORY*2:]
            await send_long_message(update, clean)
            agent_stats["messages"] += 1
            return
        
        action, params = tool_call
        agent_stats["tools"] += 1
        
        # Show typing for long ops
        try: await update.effective_chat.send_action(ChatAction.TYPING)
        except: pass
        
        # Execute
        try:
            result = await TOOLS[action](params, chat_id, ctx)
        except Exception as e:
            result = {"text": f"❌ {action}: {e}"}
        
        # Send files immediately
        if "files" in result:
            for fp in result["files"]:
                try:
                    p = Path(fp)
                    if p.exists():
                        with open(fp, "rb") as f:
                            if fp.endswith(('.png','.jpg','.jpeg','.gif','.webp')):
                                await update.message.reply_photo(photo=InputFile(f), caption=result.get("text","")[:1000])
                            else:
                                await update.message.reply_document(document=InputFile(f, filename=p.name), caption=result.get("text","")[:1000])
                except Exception as e: log.error(f"File send: {e}")
        if "voice" in result:
            try:
                with open(result["voice"], "rb") as f:
                    await update.message.reply_voice(voice=InputFile(f))
            except Exception as e: log.error(f"Voice: {e}")
        
        messages.append({"role":"assistant","content":response})
        messages.append({"role":"user","content":f"[نتيجة {action}]: {result['text'][:3000]}\n\n{'استمر أو لخّص.' if round_num < MAX_TOOL_ROUNDS-1 else 'لخّص.'}"})
    
    # Max rounds reached
    final = await llm_call(messages)
    await send_long_message(update, final)
    agent_stats["messages"] += 1


async def send_long_message(update, text):
    if not text.strip(): return
    chunks = []
    while text:
        if len(text) <= 4000: chunks.append(text); break
        sp = text[:4000].rfind('\n')
        if sp < 2000: sp = 4000
        chunks.append(text[:sp]); text = text[sp:].lstrip()
    for chunk in chunks:
        try: await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        except:
            try: await update.message.reply_text(chunk)
            except Exception as e: log.error(f"Send: {e}")


# ══════════════════════════════════════
# HANDLERS
# ══════════════════════════════════════

async def cmd_start(update, context):
    kb = [
        [InlineKeyboardButton("🌐 ملخص المنصات", callback_data="do:platform_overview"),
         InlineKeyboardButton("🔍 بحث", callback_data="ask:search")],
        [InlineKeyboardButton("🌤 الطقس", callback_data="ask:weather"),
         InlineKeyboardButton("🎨 صورة AI", callback_data="ask:image")],
        [InlineKeyboardButton("🧠 النموذج", callback_data="do:model"),
         InlineKeyboardButton("📊 الحالة", callback_data="do:status")],
    ]
    await update.message.reply_text(
        "🤖 *OmniCloud AI v2*\n━━━━━━━━━━━━━━━━━\n\n"
        "مساعدك الذكي — يتحكم في *7 منصات سحابية* 🌐\n\n"
        "🛠 *{n} أداة:*\n"
        "💬 محادثة ذكية · 🐍 Python · ⚡ Shell\n"
        "🔍 بحث · 🌐 تصفح · 🎨 صور AI\n"
        "📁 ملفات · 🌍 ترجمة · 📱 QR\n"
        "🌤 طقس · 📚 ويكي · 🔢 حساب\n"
        "🔊 صوت · 💾 ذاكرة · 📸 تحليل صور\n\n"
        "☁️ *المنصات:*\n"
        "✈️ Fly.io · 🐙 GitHub · 🤗 HuggingFace\n"
        "🎨 Render · 🚂 Railway · 🏗 Northflank · 📦 Back4App\n\n"
        "أرسل أي شيء للبدء! 🚀".format(n=len(TOOLS)),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(kb),
    )

async def cmd_status(update, context):
    t = (datetime.now(timezone.utc) - datetime.fromisoformat(agent_stats["started_at"])).total_seconds()
    d,h,m = int(t//86400), int((t%86400)//3600), int((t%3600)//60)
    ts = f"{d}d {h}h {m}m" if d else f"{h}h {m}m"
    await update.message.reply_text(
        f"📊 *OmniCloud AI v2*\n━━━━━━━━━━━━━━━━━\n"
        f"🟢 يعمل | ⏱ {ts}\n💬 {agent_stats['messages']} رسالة | 🔧 {agent_stats['tools']} أداة\n"
        f"🧠 `{LLM_MODEL.split('/')[-1]}`\n🛠 {len(TOOLS)} أداة | 📡 HuggingFace",
        parse_mode=ParseMode.MARKDOWN)

async def cmd_clear(update, context):
    conversations.pop(update.effective_chat.id, None)
    await update.message.reply_text("🗑 تم مسح المحادثة!")

async def cmd_model(update, context):
    kb = [[InlineKeyboardButton(f"{'✅ ' if m==LLM_MODEL else ''}{label}", callback_data=f"model:{m}")]
        for _, (m, label) in MODELS.items()]
    await update.message.reply_text("🧠 *اختر النموذج:*", parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(kb))

async def cmd_run(update, context):
    code = update.message.text.replace("/run","",1).strip()
    if not code: return await update.message.reply_text("❓ `/run print('hi')`", parse_mode=ParseMode.MARKDOWN)
    await update.effective_chat.send_action(ChatAction.TYPING)
    r = await tool_python({"code": code}, update.effective_chat.id, context)
    await send_long_message(update, f"```\n{r['text']}\n```")

async def cmd_sh(update, context):
    cmd = update.message.text.replace("/sh","",1).strip()
    if not cmd: return await update.message.reply_text("❓ `/sh uname -a`", parse_mode=ParseMode.MARKDOWN)
    await update.effective_chat.send_action(ChatAction.TYPING)
    r = await tool_shell({"cmd": cmd}, update.effective_chat.id, context)
    await send_long_message(update, f"```\n{r['text']}\n```")

async def cmd_img(update, context):
    prompt = update.message.text.replace("/img","",1).strip()
    if not prompt: return await update.message.reply_text("❓ `/img futuristic city`", parse_mode=ParseMode.MARKDOWN)
    await update.effective_chat.send_action(ChatAction.UPLOAD_PHOTO)
    r = await tool_generate_image({"prompt": prompt}, update.effective_chat.id, context)
    if "files" in r:
        for fp in r["files"]:
            try:
                with open(fp,"rb") as f: await update.message.reply_photo(photo=InputFile(f), caption=f"🎨 {prompt[:200]}")
            except Exception as e: await update.message.reply_text(f"⚠️ {e}")
    else: await update.message.reply_text(r["text"])

async def cmd_platforms(update, context):
    await update.effective_chat.send_action(ChatAction.TYPING)
    r = await tool_platform_overview({}, update.effective_chat.id, context)
    await send_long_message(update, r["text"])

async def callback_handler(update, context):
    global LLM_MODEL
    q = update.callback_query; await q.answer()
    if q.data.startswith("model:"):
        LLM_MODEL = q.data[6:]
        await q.edit_message_text(f"✅ النموذج: *{LLM_MODEL.split('/')[-1]}*", parse_mode=ParseMode.MARKDOWN)
    elif q.data == "do:platform_overview":
        await q.answer("⏳ جاري الفحص...", show_alert=False)
        msg = await q.message.reply_text("⏳ *أفحص كل المنصات...*", parse_mode=ParseMode.MARKDOWN)
        r = await tool_platform_overview({}, q.message.chat_id, context)
        await msg.edit_text(r["text"], parse_mode=ParseMode.MARKDOWN)
    elif q.data == "do:status":
        await q.answer("📊")
    elif q.data == "do:model":
        await q.answer("اكتب /model")
    elif q.data.startswith("ask:"):
        hints = {"search": "اكتب سؤالك أو ما تريد البحث عنه","weather":"اكتب: الطقس في [المدينة]","image":"اكتب: ارسم [وصف]"}
        await q.answer(f"💡 {hints.get(q.data[4:],'جرب!')}", show_alert=True)

async def handle_message(update, context):
    if not update.message or not update.message.text: return
    await update.effective_chat.send_action(ChatAction.TYPING)
    try: await agent_loop(update.message.text, update.effective_chat.id, context, update)
    except Exception as e:
        agent_stats["errors"] += 1
        log.error(f"Error: {e}\n{traceback.format_exc()}")
        await update.message.reply_text(f"⚠️ {str(e)[:200]}")

async def handle_document(update, context):
    doc = update.message.document
    if not doc: return
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        file = await context.bot.get_file(doc.file_id)
        path = DATA_DIR / doc.file_name
        await file.download_to_drive(str(path))
        text_ext = {'.py','.js','.ts','.json','.txt','.md','.csv','.yml','.yaml','.html','.css','.sh',
                    '.toml','.xml','.sql','.jsx','.tsx','.env','.cfg','.ini','.log','.java','.cpp',
                    '.c','.h','.rs','.go','.rb','.php','.r','.swift','.kt'}
        ext = os.path.splitext(doc.file_name)[1].lower()
        if ext in text_ext:
            content = path.read_text(errors="replace")[:8000]
            msg = f"[ملف: {doc.file_name}]\n```\n{content}\n```\n{update.message.caption or 'حلل هذا الملف.'}"
        else:
            msg = f"[ملف: {doc.file_name} | {doc.file_size}B | {doc.mime_type}]\n{update.message.caption or 'ما هذا؟'}"
        await agent_loop(msg, update.effective_chat.id, context, update)
    except Exception as e: await update.message.reply_text(f"⚠️ {e}")

async def handle_photo(update, context):
    await update.effective_chat.send_action(ChatAction.TYPING)
    caption = update.message.caption or "حلل هذه الصورة"
    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        fp = DATA_DIR / f"photo_{update.effective_chat.id}_{int(time.time())}.jpg"
        await file.download_to_drive(str(fp))
        # Vision model
        img_desc = ""
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post("https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-large",
                    headers={"Authorization": f"Bearer {HF_TOKEN}"}, content=fp.read_bytes())
                if r.status_code == 200:
                    d = r.json()
                    if isinstance(d, list) and d: img_desc = d[0].get("generated_text","")
        except: pass
        msg = f"[صورة{': ' + img_desc if img_desc else ''}]\n{caption}"
        await agent_loop(msg, update.effective_chat.id, context, update)
    except Exception as e: await update.message.reply_text(f"⚠️ {e}")


# ══════════════════════════════════════
# MAIN
# ══════════════════════════════════════

import aiohttp.web as web

# Global reference to the telegram Application
tg_app: Application | None = None
tg_ready = False


async def health_handler(request):
    """Health endpoint — responds immediately so HF keeps the container alive."""
    return web.json_response({
        "status": "running" if tg_ready else "starting",
        "v": "2.0",
        "model": LLM_MODEL,
        "tools": len(TOOLS),
        "webhook": tg_ready,
    })


async def webhook_handler(request):
    """Receive Telegram updates via webhook — return 200 immediately, process in background."""
    if not tg_ready or tg_app is None:
        return web.Response(status=503, text="Bot not ready")
    try:
        data = await request.json()
        update = Update.de_json(data, tg_app.bot)
        # Process in background so Telegram gets instant 200
        asyncio.create_task(_safe_process(update))
    except Exception as e:
        log.error(f"Webhook parse error: {e}")
    return web.Response(status=200, text="ok")


async def _safe_process(update):
    """Process a Telegram update with error handling."""
    try:
        await tg_app.process_update(update)
    except Exception as e:
        log.error(f"Update processing error: {e}")


async def init_telegram_bot():
    """Connect to Telegram with retries, then set webhook. Runs in background."""
    global tg_app, tg_ready
    from telegram.request import HTTPXRequest

    platforms = []
    if FLY_TOKEN: platforms.append("✈️ Fly.io")
    if GITHUB_TOKEN: platforms.append("🐙 GitHub")
    platforms.append("🤗 HuggingFace")
    if RENDER_KEY: platforms.append("🎨 Render")
    if RAILWAY_TOKEN: platforms.append("🚂 Railway")
    if NORTHFLANK_TOKEN: platforms.append("🏗 Northflank")
    if BACK4APP_TOKEN: platforms.append("📦 Back4App")

    for attempt in range(10):
        try:
            log.info(f"Telegram connect attempt {attempt + 1}/10...")
            request = HTTPXRequest(
                connect_timeout=120, read_timeout=120,
                write_timeout=120, pool_timeout=120,
            )
            app = Application.builder().token(BOT_TOKEN).request(request).build()

            # Register handlers
            for cmd, fn in [("start", cmd_start), ("help", cmd_start), ("status", cmd_status),
                            ("clear", cmd_clear), ("model", cmd_model), ("run", cmd_run),
                            ("sh", cmd_sh), ("img", cmd_img), ("platforms", cmd_platforms)]:
                app.add_handler(CommandHandler(cmd, fn))
            app.add_handler(CallbackQueryHandler(callback_handler))
            app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
            app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

            # initialize() calls get_me() — this is what times out
            await app.initialize()
            log.info("Bot initialized ✓")

            # Set webhook
            await app.bot.set_webhook(
                url=WEBHOOK_URL,
                drop_pending_updates=True,
                allowed_updates=["message", "callback_query"],
            )
            log.info(f"Webhook set → {WEBHOOK_URL}")

            await app.start()
            tg_app = app
            tg_ready = True
            log.info("🟢 Bot fully ready!")

            # Notify owner
            try:
                await app.bot.send_message(
                    chat_id=OWNER_CHAT_ID,
                    text=(
                        "🚀 *OmniCloud AI v2 — Online!*\n━━━━━━━━━━━━━━━━━\n\n"
                        f"🧠 `{LLM_MODEL.split('/')[-1]}`\n"
                        f"🛠 {len(TOOLS)} أداة\n"
                        f"☁️ {len(platforms)} منصات: {', '.join(platforms)}\n"
                        f"⏱ {datetime.now(timezone.utc).strftime('%H:%M UTC')}\n\n"
                        "/start لرؤية القدرات 🤖"
                    ),
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception as e:
                log.warning(f"Notify: {e}")
            return  # success

        except Exception as e:
            log.error(f"Attempt {attempt + 1} failed: {e}")
            try:
                await app.shutdown()
            except Exception:
                pass
            wait = min(15 * (attempt + 1), 120)
            log.info(f"Retrying in {wait}s...")
            await asyncio.sleep(wait)

    log.error("All 10 attempts failed — bot will not respond to messages.")


async def on_startup(app_web):
    """Launch Telegram init as a background task (web server already running)."""
    asyncio.create_task(init_telegram_bot())


def main():
    log.info("=" * 50)
    log.info(f"OmniCloud AI v2 — WEBHOOK MODE")
    log.info(f"Model: {LLM_MODEL} | Port: {PORT} | Tools: {len(TOOLS)}")
    log.info("=" * 50)

    # aiohttp web server — starts IMMEDIATELY, Telegram connects in background
    app_web = web.Application()
    app_web.router.add_get("/", health_handler)
    app_web.router.add_post(WEBHOOK_PATH, webhook_handler)
    app_web.on_startup.append(on_startup)

    log.info(f"Starting web server on 0.0.0.0:{PORT}")
    web.run_app(app_web, host="0.0.0.0", port=PORT, print=log.info)


if __name__ == "__main__":
    main()
