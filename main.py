"""
OmniCloud AI Agent — Telegram Bot with AI + Tools
Runs on Render, uses HuggingFace free inference for LLM reasoning.
"""
import asyncio
import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from telegram.constants import ParseMode, ChatAction

# ── Config ──
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
HF_TOKEN = os.environ["HF_TOKEN"]
OWNER_CHAT_ID = int(os.environ.get("OWNER_CHAT_ID", "5245619457"))
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
LLM_FALLBACK = "meta-llama/Llama-3.3-70B-Instruct"
MAX_HISTORY = 20
PORT = int(os.environ.get("PORT", "10000"))

# HuggingFace Inference API — OpenAI-compatible endpoint
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("agent")

# ── State ──
conversations: dict[int, list] = {}
agent_stats = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "messages_processed": 0,
    "tools_used": 0,
    "errors": 0,
}


# ═══════════════════════════════════════════
# HEALTH SERVER (starts first for Render)
# ═══════════════════════════════════════════

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        data = {
            "status": "ok",
            "model": LLM_MODEL,
            "messages": agent_stats["messages_processed"],
            "tools": agent_stats["tools_used"],
        }
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        pass  # Suppress access logs


def start_health_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    log.info(f"Health server on port {PORT}")
    server.serve_forever()


# ═══════════════════════════════════════════
# LLM — HuggingFace Inference (free, OpenAI-compatible)
# ═══════════════════════════════════════════

SYSTEM_PROMPT = """أنت OmniCloud AI — وكيل ذكاء اصطناعي متقدم يعمل عبر 7 منصات سحابية.

قواعدك:
1. أجب بالعربية إذا تحدث المستخدم بالعربية، وبالإنجليزية إذا تحدث بالإنجليزية.
2. كن مختصراً ومفيداً. لا تكرر السؤال.
3. إذا طُلب منك تنفيذ كود أو أمر، استخدم الأدوات المتاحة.
4. أنت قادر على: البحث في الويب، تنفيذ كود Python، إدارة الخوادم، تحليل البيانات.
5. كن ودوداً ومهنياً.

المنصات المتاحة لك:
- Fly.io: نشر التطبيقات والحاويات
- GitHub: إدارة المستودعات والكود
- Railway: قواعد البيانات
- Render: خدمات الويب (أنت تعمل هنا)
- HuggingFace: نماذج AI مجانية (تستخدمها الآن)
- Northflank: جدولة المهام
- Back4App: قواعد بيانات فورية

لديك الأدوات التالية. إذا احتجت استخدام أداة، أجب بهذا التنسيق JSON فقط (لا تضع نص قبله أو بعده):
```json
{{"tool": "tool_name", "args": {{...}}}}
```

الأدوات:
1. python — تنفيذ كود Python: {{"tool": "python", "args": {{"code": "print('hello')"}}}}
2. shell — تنفيذ أمر: {{"tool": "shell", "args": {{"command": "ls -la"}}}}
3. web_search — بحث ويب: {{"tool": "web_search", "args": {{"query": "search query"}}}}
4. web_fetch — جلب صفحة: {{"tool": "web_fetch", "args": {{"url": "https://example.com"}}}}

إذا لم تحتج أداة، أجب نصاً عادياً.
التاريخ الحالي: {date}"""


async def llm_chat(messages: list, model: str = None) -> str:
    """Call HuggingFace Inference API."""
    model = model or LLM_MODEL

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        for attempt in range(3):
            try:
                resp = await client.post(HF_API_URL, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return content if content else "(empty response)"
                elif resp.status_code in (503, 529):
                    log.warning(f"Model busy, wait 10s (attempt {attempt+1})")
                    await asyncio.sleep(10)
                    continue
                else:
                    log.error(f"LLM {resp.status_code}: {resp.text[:300]}")
                    if model != LLM_FALLBACK:
                        log.info(f"Falling back to {LLM_FALLBACK}")
                        return await llm_chat(messages, model=LLM_FALLBACK)
                    return f"⚠️ خطأ AI: {resp.status_code}"
            except Exception as e:
                log.error(f"LLM error: {e}")
                if attempt < 2:
                    await asyncio.sleep(5)
                    continue
                if model != LLM_FALLBACK:
                    return await llm_chat(messages, model=LLM_FALLBACK)
                return "⚠️ خطأ في الاتصال"

    return "⚠️ AI غير متاح"


async def chat_with_tools(user_msg: str, chat_id: int) -> str:
    """Chat with optional tool use."""
    if chat_id not in conversations:
        conversations[chat_id] = []

    history = conversations[chat_id]

    system_msg = {
        "role": "system",
        "content": SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d %H:%M UTC")),
    }

    messages = [system_msg] + history[-MAX_HISTORY:] + [
        {"role": "user", "content": user_msg}
    ]

    response = await llm_chat(messages)

    # Check for tool call
    tool_result = await try_execute_tool(response)
    if tool_result:
        agent_stats["tools_used"] += 1
        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "user",
            "content": f"نتيجة الأداة:\n```\n{tool_result[:3000]}\n```\nلخّص النتيجة للمستخدم بشكل واضح."
        })
        response = await llm_chat(messages)

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": response})
    if len(history) > MAX_HISTORY * 2:
        history[:] = history[-MAX_HISTORY * 2:]

    return response


# ═══════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════

async def try_execute_tool(response: str) -> str | None:
    """Parse and execute tool call from LLM response."""
    try:
        # Look for JSON in code block or raw
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if not m:
            m = re.search(r'(\{[^{}]*"tool"[^{}]*\})', response)
        if not m:
            return None

        call = json.loads(m.group(1))
        tool = call.get("tool")
        args = call.get("args", {})

        if tool == "python":
            return await run_python(args.get("code", ""))
        elif tool == "shell":
            return await run_shell(args.get("command", ""))
        elif tool == "web_search":
            return await web_search(args.get("query", ""))
        elif tool == "web_fetch":
            return await web_fetch(args.get("url", ""))
        return None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


async def run_python(code: str) -> str:
    if not code.strip():
        return "No code"
    with open("/tmp/agent_run.py", "w") as f:
        f.write(code)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "/tmp/agent_run.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        out = stdout.decode()[:3000]
        if stderr:
            out += f"\n[stderr]: {stderr.decode()[:1000]}"
        return out or "(no output)"
    except asyncio.TimeoutError:
        return "⏱ Timeout (30s)"
    except Exception as e:
        return f"Error: {e}"


async def run_shell(cmd: str) -> str:
    if not cmd.strip():
        return "No command"
    blocked = ["rm -rf /", "mkfs", "dd if=", "> /dev/", ":(){ "]
    if any(b in cmd.lower() for b in blocked):
        return "🚫 محظور"
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        out = stdout.decode()[:3000]
        if stderr:
            out += f"\n[stderr]: {stderr.decode()[:1000]}"
        return out or "(no output)"
    except asyncio.TimeoutError:
        return "⏱ Timeout"
    except Exception as e:
        return f"Error: {e}"


async def web_search(query: str) -> str:
    if not query.strip():
        return "No query"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            results = []
            for m in re.finditer(r'class="result__snippet">(.*?)</a>', resp.text, re.DOTALL):
                snippet = re.sub(r'<[^>]+>', '', m.group(1)).strip()
                if snippet:
                    results.append(snippet)
                if len(results) >= 5:
                    break
            return "\n\n".join(results) if results else "لم يتم العثور على نتائج"
    except Exception as e:
        return f"Search error: {e}"


async def web_fetch(url: str) -> str:
    if not url.startswith("http"):
        return "Invalid URL"
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            text = re.sub(r'<script.*?</script>', '', resp.text, flags=re.DOTALL)
            text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:3000]
    except Exception as e:
        return f"Fetch error: {e}"


# ═══════════════════════════════════════════
# TELEGRAM HANDLERS
# ═══════════════════════════════════════════

async def cmd_start(update: Update, context):
    await update.message.reply_text(
        "🤖 *مرحباً! أنا OmniCloud AI*\n\n"
        "وكيل ذكاء اصطناعي مستقل يعمل عبر 7 منصات سحابية.\n\n"
        "✨ *ما يمكنني فعله:*\n"
        "• 💬 محادثة ذكية (عربي/إنجليزي)\n"
        "• 🐍 تنفيذ كود Python\n"
        "• 🔍 البحث في الإنترنت\n"
        "• 🌐 جلب صفحات الويب\n"
        "• ⚡ أوامر Shell\n\n"
        "💡 *الأوامر:*\n"
        "`/status` — حالة النظام\n"
        "`/clear` — مسح المحادثة\n"
        "`/model` — تغيير نموذج AI\n"
        "`/run code` — تنفيذ Python\n"
        "`/sh command` — أمر shell\n\n"
        "أرسل أي رسالة للبدء! 🚀",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_status(update: Update, context):
    t = (datetime.now(timezone.utc) - datetime.fromisoformat(agent_stats["started_at"])).total_seconds()
    h, m = int(t // 3600), int((t % 3600) // 60)
    await update.message.reply_text(
        f"📊 *حالة النظام*\n\n"
        f"⏱ التشغيل: {h}h {m}m\n"
        f"💬 الرسائل: {agent_stats['messages_processed']}\n"
        f"🔧 الأدوات: {agent_stats['tools_used']}\n"
        f"🧠 النموذج: `{LLM_MODEL}`\n"
        f"📡 المنصة: Render\n"
        f"🟢 الحالة: يعمل",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_clear(update: Update, context):
    conversations.pop(update.effective_chat.id, None)
    await update.message.reply_text("🗑 تم مسح المحادثة!")


async def cmd_model(update: Update, context):
    models = [
        ("Qwen/Qwen2.5-72B-Instruct", "Qwen 72B 🧠"),
        ("meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B 🦙"),
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen 7B ⚡"),
        ("meta-llama/Llama-3.1-8B-Instruct", "Llama 8B ⚡"),
    ]
    keyboard = [
        [InlineKeyboardButton(
            f"{'✓ ' if m == LLM_MODEL else ''}{label}",
            callback_data=f"model:{m}",
        )]
        for m, label in models
    ]
    await update.message.reply_text(
        "🧠 *اختر نموذج AI:*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def callback_handler(update: Update, context):
    global LLM_MODEL
    q = update.callback_query
    await q.answer()
    if q.data.startswith("model:"):
        LLM_MODEL = q.data[6:]
        name = LLM_MODEL.split("/")[-1]
        await q.edit_message_text(f"✅ النموذج: *{name}*", parse_mode=ParseMode.MARKDOWN)


async def cmd_run(update: Update, context):
    code = update.message.text.replace("/run", "", 1).strip()
    if not code:
        await update.message.reply_text("❓ `/run print('hello')`", parse_mode=ParseMode.MARKDOWN)
        return
    await update.effective_chat.send_action(ChatAction.TYPING)
    result = await run_python(code)
    try:
        await update.message.reply_text(f"```\n{result[:4000]}\n```", parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(result[:4000])


async def cmd_sh(update: Update, context):
    cmd = update.message.text.replace("/sh", "", 1).strip()
    if not cmd:
        await update.message.reply_text("❓ `/sh uname -a`", parse_mode=ParseMode.MARKDOWN)
        return
    await update.effective_chat.send_action(ChatAction.TYPING)
    result = await run_shell(cmd)
    try:
        await update.message.reply_text(f"```\n{result[:4000]}\n```", parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(result[:4000])


async def handle_message(update: Update, context):
    if not update.message or not update.message.text:
        return
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        response = await chat_with_tools(update.message.text, update.effective_chat.id)
        agent_stats["messages_processed"] += 1
        if len(response) > 4000:
            for i in range(0, len(response), 4000):
                await update.message.reply_text(response[i:i+4000])
        else:
            try:
                await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(response)
    except Exception as e:
        agent_stats["errors"] += 1
        log.error(f"Error: {e}\n{traceback.format_exc()}")
        await update.message.reply_text(f"⚠️ خطأ: {str(e)[:200]}")


async def handle_document(update: Update, context):
    doc = update.message.document
    if not doc:
        return
    await update.effective_chat.send_action(ChatAction.TYPING)
    file = await context.bot.get_file(doc.file_id)
    path = f"/tmp/{doc.file_name}"
    await file.download_to_drive(path)

    text_ext = {'.py', '.js', '.ts', '.json', '.txt', '.md', '.csv', '.yml',
                '.yaml', '.html', '.css', '.sh', '.toml', '.xml', '.sql', '.jsx', '.tsx'}
    ext = os.path.splitext(doc.file_name)[1].lower()

    if ext in text_ext:
        try:
            with open(path, "r", errors="replace") as f:
                content = f.read()[:5000]
            msg = f"ملف: {doc.file_name}\n```\n{content}\n```\nحلل هذا الملف."
            response = await chat_with_tools(msg, update.effective_chat.id)
            try:
                await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(response)
        except Exception as e:
            await update.message.reply_text(f"⚠️ {e}")
    else:
        await update.message.reply_text(
            f"📁 ملف: `{doc.file_name}` ({doc.file_size} bytes)",
            parse_mode=ParseMode.MARKDOWN,
        )


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    log.info("=" * 50)
    log.info("OmniCloud AI Agent — Starting")
    log.info(f"Model: {LLM_MODEL} | Port: {PORT}")
    log.info("=" * 50)

    # Start health server FIRST (Render needs port bound quickly)
    Thread(target=start_health_server, daemon=True).start()
    log.info("Health server started ✓")

    # Build Telegram app
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("run", cmd_run))
    app.add_handler(CommandHandler("sh", cmd_sh))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Send startup notification
    async def post_init(application):
        try:
            await application.bot.send_message(
                chat_id=OWNER_CHAT_ID,
                text=(
                    "🟢 *OmniCloud AI — Online!*\n\n"
                    f"🧠 `{LLM_MODEL}`\n"
                    f"📡 Render | ⏱ {datetime.now(timezone.utc).strftime('%H:%M UTC')}\n\n"
                    "أرسل /start للمساعدة 🚀"
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:
            log.warning(f"Startup notification failed: {e}")

    app.post_init = post_init

    log.info("Starting Telegram polling...")
    app.run_polling(drop_pending_updates=True)


async def handle_photo(update: Update, context):
    await update.message.reply_text("📸 تحليل الصور قادم في التحديث القادم!")


if __name__ == "__main__":
    main()
