"""
OmniCloud AI Agent — Telegram Bot with AI + Tools
Runs on Fly.io, uses HuggingFace free inference for LLM reasoning.
"""
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

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
MAX_HISTORY = 20  # messages to keep in conversation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("agent")

# ── In-memory state ──
conversations: dict[int, list] = {}  # chat_id -> message history
agent_stats = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "messages_processed": 0,
    "tools_used": 0,
    "errors": 0,
}


# ═══════════════════════════════════════════
# LLM — HuggingFace Inference (free)
# ═══════════════════════════════════════════

SYSTEM_PROMPT = """أنت OmniCloud AI — وكيل ذكاء اصطناعي متقدم يعمل عبر 7 منصات سحابية.

قواعدك:
1. أجب بالعربية إذا تحدث المستخدم بالعربية، وبالإنجليزية إذا تحدث بالإنجليزية.
2. كن مختصراً ومفيداً. لا تكرر السؤال.
3. إذا طُلب منك تنفيذ كود أو أمر، استخدم الأدوات المتاحة.
4. أنت قادر على: البحث في الويب، تنفيذ كود Python، إدارة الخوادم، تحليل البيانات.
5. كن ودوداً ومهنياً.

المنصات المتاحة:
- Fly.io: نشر التطبيقات والحاويات
- GitHub: إدارة المستودعات والكود
- Railway: قواعد البيانات
- Render: خدمات الويب والعمال
- HuggingFace: نماذج AI مجانية
- Northflank: جدولة المهام
- Back4App: قواعد بيانات فورية

أنت تعمل الآن على Fly.io. التاريخ الحالي: {date}"""


async def llm_chat(messages: list, model: str = None) -> str:
    """Call HuggingFace Inference API for chat completion."""
    model = model or LLM_MODEL
    
    # Build the API URL - HF SDK style
    url = f"https://router.huggingface.co/hf-inference/models/{model}/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        for attempt in range(3):
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                elif resp.status_code == 503:
                    # Model loading
                    log.warning(f"Model loading, wait 10s (attempt {attempt+1})")
                    await asyncio.sleep(10)
                    continue
                else:
                    log.error(f"LLM error {resp.status_code}: {resp.text[:200]}")
                    if model != LLM_FALLBACK:
                        log.info(f"Falling back to {LLM_FALLBACK}")
                        return await llm_chat(messages, model=LLM_FALLBACK)
                    return f"⚠️ خطأ في الـ AI: {resp.status_code}"
            except Exception as e:
                log.error(f"LLM request error: {e}")
                if attempt < 2:
                    await asyncio.sleep(5)
                    continue
                if model != LLM_FALLBACK:
                    return await llm_chat(messages, model=LLM_FALLBACK)
                return "⚠️ خطأ في الاتصال بالـ AI"
    
    return "⚠️ الـ AI غير متاح حالياً"


async def llm_with_tools(user_msg: str, chat_id: int) -> str:
    """Chat with tool-use capability via function calling emulation."""
    # Get or create conversation history
    if chat_id not in conversations:
        conversations[chat_id] = []
    
    history = conversations[chat_id]
    
    # Build messages
    system = {
        "role": "system",
        "content": SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d %H:%M UTC")),
    }
    
    # Add tool instructions
    tool_system = {
        "role": "system",
        "content": """لديك الأدوات التالية. إذا احتجت استخدام أداة، أجب بتنسيق JSON فقط:
{"tool": "tool_name", "args": {...}}

الأدوات المتاحة:
1. {"tool": "python", "args": {"code": "python code here"}} — تنفيذ كود Python
2. {"tool": "shell", "args": {"command": "shell command"}} — تنفيذ أمر shell  
3. {"tool": "web_search", "args": {"query": "search query"}} — بحث في الويب
4. {"tool": "web_fetch", "args": {"url": "https://..."}} — جلب صفحة ويب
5. {"tool": "deploy_flyio", "args": {"app": "app-name", "image": "docker-image"}} — نشر على Fly.io

إذا لم تحتج أداة، أجب نصاً عادياً بدون JSON.""",
    }
    
    messages = [system, tool_system] + history[-MAX_HISTORY:] + [
        {"role": "user", "content": user_msg}
    ]
    
    # Get LLM response
    response = await llm_chat(messages)
    
    # Check if response is a tool call
    tool_result = await try_execute_tool(response)
    if tool_result:
        agent_stats["tools_used"] += 1
        # Send tool result back to LLM for final answer
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"نتيجة الأداة:\n```\n{tool_result[:3000]}\n```\nلخّص النتيجة للمستخدم."})
        response = await llm_chat(messages)
    
    # Update conversation history
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": response})
    
    # Trim history
    if len(history) > MAX_HISTORY * 2:
        history[:] = history[-MAX_HISTORY * 2:]
    
    return response


# ═══════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════

async def try_execute_tool(response: str) -> str | None:
    """Try to parse and execute a tool call from LLM response."""
    # Try to find JSON in the response
    try:
        # Look for JSON block
        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response)
        if not json_match:
            # Try to find in code block
            code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if code_match:
                json_match = code_match
            else:
                return None
        
        text = json_match.group(0) if hasattr(json_match, 'group') else json_match
        if isinstance(text, re.Match):
            text = text.group(0)
        
        tool_call = json.loads(text)
        tool = tool_call.get("tool")
        args = tool_call.get("args", {})
        
        if tool == "python":
            return await execute_python(args.get("code", ""))
        elif tool == "shell":
            return await execute_shell(args.get("command", ""))
        elif tool == "web_search":
            return await web_search(args.get("query", ""))
        elif tool == "web_fetch":
            return await web_fetch(args.get("url", ""))
        elif tool == "deploy_flyio":
            return await deploy_flyio(args.get("app", ""), args.get("image", ""))
        else:
            return None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


async def execute_python(code: str) -> str:
    """Execute Python code safely."""
    if not code.strip():
        return "No code provided"
    
    # Write to temp file and execute with timeout
    tmp_path = "/tmp/agent_code.py"
    with open(tmp_path, "w") as f:
        f.write(code)
    
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode()[:3000]
        if stderr:
            output += f"\n[STDERR]: {stderr.decode()[:1000]}"
        return output or "(no output)"
    except asyncio.TimeoutError:
        proc.kill()
        return "⏱ Code execution timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"


async def execute_shell(command: str) -> str:
    """Execute shell command safely."""
    if not command.strip():
        return "No command provided"
    
    # Block dangerous commands
    dangerous = ["rm -rf /", "mkfs", "dd if=", "> /dev/", ":(){ ", "fork"]
    if any(d in command.lower() for d in dangerous):
        return "🚫 الأمر محظور لأسباب أمنية"
    
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode()[:3000]
        if stderr:
            output += f"\n[STDERR]: {stderr.decode()[:1000]}"
        return output or "(no output)"
    except asyncio.TimeoutError:
        proc.kill()
        return "⏱ Command timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"


async def web_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    if not query.strip():
        return "No query provided"
    
    try:
        url = "https://html.duckduckgo.com/html/"
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, data={"q": query})
            # Extract result snippets
            text = resp.text
            results = []
            for match in re.finditer(r'class="result__snippet">(.*?)</a>', text, re.DOTALL):
                snippet = re.sub(r'<[^>]+>', '', match.group(1)).strip()
                if snippet:
                    results.append(snippet)
                if len(results) >= 5:
                    break
            return "\n\n".join(results) if results else "لم يتم العثور على نتائج"
    except Exception as e:
        return f"Search error: {e}"


async def web_fetch(url: str) -> str:
    """Fetch a web page and extract text."""
    if not url.startswith("http"):
        return "Invalid URL"
    
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            # Strip HTML tags and get text
            text = re.sub(r'<script[^>]*>.*?</script>', '', resp.text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:3000]
    except Exception as e:
        return f"Fetch error: {e}"


async def deploy_flyio(app: str, image: str) -> str:
    """Deploy to Fly.io (placeholder — needs flyctl)."""
    return "🔧 نشر Fly.io يتطلب flyctl. سيتم إضافته في التحديث القادم."


# ═══════════════════════════════════════════
# TELEGRAM HANDLERS
# ═══════════════════════════════════════════

async def cmd_start(update: Update, context):
    """Handle /start command."""
    welcome = (
        "🤖 *مرحباً! أنا OmniCloud AI*\n\n"
        "وكيل ذكاء اصطناعي مستقل يعمل عبر 7 منصات سحابية.\n\n"
        "✨ *ما يمكنني فعله:*\n"
        "• 💬 محادثة ذكية بالعربية والإنجليزية\n"
        "• 🐍 تنفيذ كود Python\n"
        "• 🔍 البحث في الإنترنت\n"
        "• 🌐 جلب وتحليل صفحات الويب\n"
        "• ⚡ تنفيذ أوامر Shell\n"
        "• 🚀 نشر التطبيقات\n\n"
        "💡 *الأوامر:*\n"
        "/start — هذه الرسالة\n"
        "/status — حالة النظام\n"
        "/clear — مسح المحادثة\n"
        "/model — تغيير نموذج AI\n"
        "/run `code` — تنفيذ كود Python مباشرة\n"
        "/sh `command` — تنفيذ أمر shell\n\n"
        "أرسل أي رسالة للبدء! 🚀"
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.MARKDOWN)


async def cmd_status(update: Update, context):
    """Show system status."""
    uptime_s = (datetime.now(timezone.utc) - datetime.fromisoformat(agent_stats["started_at"])).total_seconds()
    hours = int(uptime_s // 3600)
    mins = int((uptime_s % 3600) // 60)
    
    status = (
        "📊 *حالة النظام*\n\n"
        f"⏱ *وقت التشغيل:* {hours}h {mins}m\n"
        f"💬 *الرسائل:* {agent_stats['messages_processed']}\n"
        f"🔧 *الأدوات المستخدمة:* {agent_stats['tools_used']}\n"
        f"❌ *الأخطاء:* {agent_stats['errors']}\n"
        f"🧠 *النموذج:* `{LLM_MODEL}`\n"
        f"📡 *المنصة:* Fly.io\n"
        f"🟢 *الحالة:* يعمل"
    )
    await update.message.reply_text(status, parse_mode=ParseMode.MARKDOWN)


async def cmd_clear(update: Update, context):
    """Clear conversation history."""
    chat_id = update.effective_chat.id
    conversations.pop(chat_id, None)
    await update.message.reply_text("🗑 تم مسح المحادثة. ابدأ من جديد!")


async def cmd_model(update: Update, context):
    """Switch LLM model."""
    global LLM_MODEL
    models = [
        ("Qwen/Qwen2.5-72B-Instruct", "Qwen 72B 🧠"),
        ("meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B 🦙"),
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen 7B ⚡ (سريع)"),
        ("meta-llama/Llama-3.1-8B-Instruct", "Llama 8B ⚡ (سريع)"),
    ]
    
    keyboard = [
        [InlineKeyboardButton(f"{'✓ ' if m == LLM_MODEL else ''}{label}", callback_data=f"model:{m}")]
        for m, label in models
    ]
    await update.message.reply_text(
        "🧠 *اختر نموذج AI:*\n\n72B = أذكى لكن أبطأ\n7-8B = أسرع لكن أبسط",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def callback_handler(update: Update, context):
    """Handle inline keyboard callbacks."""
    global LLM_MODEL
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith("model:"):
        LLM_MODEL = query.data[6:]
        model_name = LLM_MODEL.split("/")[-1]
        await query.edit_message_text(f"✅ تم تغيير النموذج إلى: *{model_name}*", parse_mode=ParseMode.MARKDOWN)


async def cmd_run(update: Update, context):
    """Execute Python code directly."""
    code = update.message.text.replace("/run", "", 1).strip()
    if not code:
        await update.message.reply_text("❓ أرسل الكود بعد /run\nمثال: `/run print('Hello')`", parse_mode=ParseMode.MARKDOWN)
        return
    
    await update.effective_chat.send_action(ChatAction.TYPING)
    result = await execute_python(code)
    await update.message.reply_text(f"```\n{result[:4000]}\n```", parse_mode=ParseMode.MARKDOWN)


async def cmd_sh(update: Update, context):
    """Execute shell command directly."""
    cmd = update.message.text.replace("/sh", "", 1).strip()
    if not cmd:
        await update.message.reply_text("❓ أرسل الأمر بعد /sh\nمثال: `/sh uname -a`", parse_mode=ParseMode.MARKDOWN)
        return
    
    await update.effective_chat.send_action(ChatAction.TYPING)
    result = await execute_shell(cmd)
    await update.message.reply_text(f"```\n{result[:4000]}\n```", parse_mode=ParseMode.MARKDOWN)


async def handle_message(update: Update, context):
    """Handle regular text messages."""
    if not update.message or not update.message.text:
        return
    
    chat_id = update.effective_chat.id
    user_msg = update.message.text
    
    # Show typing indicator
    await update.effective_chat.send_action(ChatAction.TYPING)
    
    try:
        response = await llm_with_tools(user_msg, chat_id)
        agent_stats["messages_processed"] += 1
        
        # Split long messages
        if len(response) > 4000:
            for i in range(0, len(response), 4000):
                chunk = response[i:i+4000]
                await update.message.reply_text(chunk)
        else:
            # Try markdown first, fallback to plain text
            try:
                await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(response)
    except Exception as e:
        agent_stats["errors"] += 1
        log.error(f"Error handling message: {e}\n{traceback.format_exc()}")
        await update.message.reply_text(f"⚠️ حدث خطأ: {str(e)[:200]}")


async def handle_document(update: Update, context):
    """Handle file uploads."""
    doc = update.message.document
    if not doc:
        return
    
    await update.effective_chat.send_action(ChatAction.TYPING)
    
    # Download file
    file = await context.bot.get_file(doc.file_id)
    file_path = f"/tmp/{doc.file_name}"
    await file.download_to_drive(file_path)
    
    # Read content if text file
    text_extensions = {'.py', '.js', '.ts', '.json', '.txt', '.md', '.csv', '.yml', '.yaml', '.html', '.css', '.sh', '.env', '.toml', '.cfg', '.ini', '.xml', '.sql'}
    ext = os.path.splitext(doc.file_name)[1].lower()
    
    if ext in text_extensions:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()[:5000]
            user_msg = f"المستخدم أرسل ملف: {doc.file_name}\n\nمحتوى الملف:\n```\n{content}\n```\n\nحلل هذا الملف وأعطني ملخصاً."
            response = await llm_with_tools(user_msg, update.effective_chat.id)
            try:
                await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(response)
        except Exception as e:
            await update.message.reply_text(f"⚠️ خطأ في قراءة الملف: {e}")
    else:
        await update.message.reply_text(f"📁 تم استلام الملف: `{doc.file_name}` ({doc.file_size} bytes)\n\nأنواع الملفات المدعومة للتحليل: نصوص، كود، JSON, CSV", parse_mode=ParseMode.MARKDOWN)


async def handle_photo(update: Update, context):
    """Handle photo uploads."""
    await update.message.reply_text("📸 استلمت الصورة! تحليل الصور سيكون متاحاً قريباً في التحديث القادم.")


# ═══════════════════════════════════════════
# HEALTH CHECK (for Fly.io)
# ═══════════════════════════════════════════

async def health_server():
    """Simple HTTP health check for Fly.io."""
    from aiohttp import web
    
    async def health(request):
        return web.json_response({
            "status": "ok",
            "uptime": (datetime.now(timezone.utc) - datetime.fromisoformat(agent_stats["started_at"])).total_seconds(),
            "messages": agent_stats["messages_processed"],
            "model": LLM_MODEL,
        })
    
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info(f"Health server running on port {port}")


# ═══════════════════════════════════════════
# STARTUP NOTIFICATION
# ═══════════════════════════════════════════

async def send_startup_notification(app):
    """Send notification to owner when bot starts."""
    try:
        await app.bot.send_message(
            chat_id=OWNER_CHAT_ID,
            text=(
                "🟢 *OmniCloud AI Agent — Online!*\n\n"
                f"🧠 النموذج: `{LLM_MODEL}`\n"
                f"📡 المنصة: Fly.io\n"
                f"⏱ التشغيل: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                "جاهز للعمل! أرسل /start للمساعدة."
            ),
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        log.warning(f"Failed to send startup notification: {e}")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    log.info("=" * 50)
    log.info("OmniCloud AI Agent — Starting...")
    log.info(f"Model: {LLM_MODEL}")
    log.info(f"Fallback: {LLM_FALLBACK}")
    log.info("=" * 50)
    
    # Build Telegram application
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("run", cmd_run))
    app.add_handler(CommandHandler("sh", cmd_sh))
    
    # Callback handler (inline keyboards)
    app.add_handler(CallbackQueryHandler(callback_handler))
    
    # Message handlers
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Post-init: start health server + send notification
    async def post_init(application):
        asyncio.create_task(health_server())
        await send_startup_notification(application)
    
    app.post_init = post_init
    
    # Run
    log.info("Starting polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
