"""
FormSaathi AI backend.

Given a user question, this service searches the web, scrapes the top
results for real content, and feeds that context plus the user's profile
(age, experience level, preferred language, chat history) into an LLM
whose system prompt adapts tone and depth to that specific user.
"""

import os
import time
import logging
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
import trafilatura
from bs4 import BeautifulSoup
try:
    from duckduckgo_search import DDGS
except ImportError:
    # Fallback if an alternate namespace is present in some environments
    from ddgs import DDGS

from groq import Groq
from flask import Flask, request, jsonify
from flask_cors import CORS

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("formsaathi")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
# Comma-separated in prod once you know your frontend's domain, e.g.
# "https://formsaathi.app,https://www.formsaathi.app". "*" is fine for now.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

SEARCH_MAX_RESULTS = 3
SCRAPE_TIMEOUT_SECONDS = 5
SCRAPE_CHAR_LIMIT = 2000
CHAT_HISTORY_TURNS = 6
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024
MAX_QUERY_LENGTH = 1000

# Repeat questions ("how do I apply for Aadhaar") are common for this kind
# of assistant, so cache web search+scrape context for a while. This is
# in-memory only -- fine for one Render instance, but won't be shared
# across multiple worker processes/instances. Swap for Redis if you scale.
CONTEXT_CACHE_TTL_SECONDS = 30 * 60
_context_cache = {}
_context_cache_lock = Lock()

# Every /ask call does a web search, up to 3 scrapes, and an LLM call --
# expensive to let anyone hammer, especially with CORS wide open. Simple
# in-memory sliding-window limiter, no new dependency required.
RATE_LIMIT_MAX_REQUESTS = 15
RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_buckets = defaultdict(deque)
_rate_limit_lock = Lock()

# Groq's free/developer-tier catalog shifts over time -- see
# console.groq.com/docs/models and /docs/deprecations. Groq announced on
# 2026-06-17 that llama-3.1-8b-instant and llama-3.3-70b-versatile were
# being deprecated for free/developer accounts in favor of the GPT-OSS
# models below, and by late Aug 2026 both had moved to Enterprise-only
# "Contact Sales" pricing. They're kept last here in case your account has
# enterprise access, but don't rely on them. Reorder freely as Groq's
# lineup changes -- get_chat_candidates() below only tries whichever of
# these are actually live on your account.
PREFERRED_CHAT_MODELS = [
    "openai/gpt-oss-120b",       # best quality, still free/developer tier
    "openai/gpt-oss-20b",        # fastest, still strong
    "qwen/qwen3.6-27b",          # preview tier -- Groq's suggested llama-3.3 replacement
    "llama-3.3-70b-versatile",   # Enterprise-only as of late Aug 2026 on most accounts
    "llama-3.1-8b-instant",      # same
]
EXCLUDED_MODEL_KEYWORDS = ["whisper", "guard", "audio", "embed", "orpheus", "vision", "tts", "compound"]
MODEL_CACHE_TTL_SECONDS = 60 * 60
_model_cache = {"ids": None, "ts": 0}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def safe_int(value, default):
    """Profile fields arrive from JSON and may be strings, missing, or
    junk -- never let a bad `age` 500 the whole request."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Web search + scrape
# --------------------------------------------------------------------------

def search_web(query, max_results=SEARCH_MAX_RESULTS):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        logger.warning("Search failed for %r: %s", query, e)
    return results


def scrape_url(url, timeout=SCRAPE_TIMEOUT_SECONDS):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            if text and len(text.strip()) > 80:
                return text.strip()[:SCRAPE_CHAR_LIMIT]

        # Fallback to BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:SCRAPE_CHAR_LIMIT] if text else None
    except Exception:
        return None


def scrape_all(results, timeout=SCRAPE_TIMEOUT_SECONDS):
    """Scrape every result in parallel instead of one at a time. Sequential
    scraping of 3 sources at up to `timeout`s each could add ~15s of pure
    waiting before the LLM is even called -- this cuts that to ~timeout."""
    scraped = {}
    if not results:
        return scraped
    with ThreadPoolExecutor(max_workers=len(results)) as executor:
        future_to_url = {executor.submit(scrape_url, r["url"], timeout): r["url"] for r in results}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                text = future.result()
                if text:
                    scraped[url] = text
            except Exception:
                continue
    return scraped


def get_context_for_query(query):
    """Search + scrape, with a short-lived cache for repeat questions.
    Only the raw web context is cached, never the final answer -- the
    system prompt (and therefore the personalization) is still rebuilt
    fresh per request from that user's own profile."""
    cache_key = query.strip().lower()
    with _context_cache_lock:
        cached = _context_cache.get(cache_key)
        if cached and (time.time() - cached["ts"] < CONTEXT_CACHE_TTL_SECONDS):
            return cached["context"], cached["sources"], True

    search_results = search_web(query)
    all_context, sources = [], []
    scraped_map = scrape_all(search_results)

    for r in search_results:
        text = scraped_map.get(r["url"])
        if text:
            all_context.append(f"[{r['title']}] ({r['url']})\n{text}")
            sources.append({"title": r["title"], "url": r["url"]})

    if not all_context:
        all_context = [f"[{r['title']}] {r['snippet']}" for r in search_results]
        sources = [{"title": r["title"], "url": r["url"]} for r in search_results]

    context = "\n\n---\n\n".join(all_context)
    with _context_cache_lock:
        _context_cache[cache_key] = {"context": context, "sources": sources, "ts": time.time()}
    return context, sources, False


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------

def is_rate_limited(client_id):
    now = time.time()
    with _rate_limit_lock:
        bucket = _rate_limit_buckets[client_id]
        while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
            return True
        bucket.append(now)
        return False


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def build_system_prompt(profile, language):
    name = profile.get("name", "User")
    age = int(profile.get("age", 30)) if str(profile.get("age", "")).isdigit() else 30
    ward = profile.get("ward", "Mumbai")
    experience = profile.get("experience", "first_time")

    # 1. Experience Level
    if experience == "first_time":
        exp_detail = "NOVICE: The user has never navigated government forms. Explain the 'why', and hold their hand through the process."
    elif experience == "some":
        exp_detail = "INTERMEDIATE: The user knows the basics. Skip elementary definitions but provide clear procedural steps."
    else:
        exp_detail = "EXPERT: Highly familiar with government tasks. Provide only URLs, exact document lists, and fees."

    # 2. Age-Based Tone & STRICT Formatting
    if age >= 60:
        tone = "Extremely warm, patient, and respectful (Namaste/Pranam). Speak as if guiding a grandparent. Limit to 3-4 simple steps."
        format_rule = "CRITICAL: ABSOLUTELY NO MARKDOWN TABLES. DO NOT use the '|' character. DO NOT use grids. Write in simple, short sentences. You MUST leave a blank empty line between EVERY single bullet point so it is visually easy for seniors to read. Keep it spacious and clean."
    elif age >= 35:
        tone = "Professional, structured, and helpful."
        format_rule = "Use clean bullet points. Avoid dense paragraphs. Tables are allowed ONLY if strictly necessary for fees."
    else:
        tone = "Modern, fast, and direct. Zero fluff."
        format_rule = "Use crisp bullet points and direct URLs."

    # 3. Language Directives
    if language == "hi":
        lang_rule = "ALWAYS respond purely in Hindi (Devanagari script). Use respectful pronouns (आप, जी)."
    elif language == "mr":
        lang_rule = "ALWAYS respond purely in Marathi (Devanagari script). Use a warm, culturally respectful tone."
    elif language == "en":
        lang_rule = "ALWAYS respond in clear Indian English."
    else:
        lang_rule = "AUTO-DETECT the language of the user's question and respond in the EXACT same language."

    # Assemble the final prompt
    return f"""You are FormSaathi AI, an intelligent Indian government document assistant for Mumbai, Maharashtra.

USER PROFILE:
- Name: {name}
- Age: {age} years old
- Location: {ward}, Mumbai, Maharashtra
- Background: {exp_detail}

COMMUNICATION STYLE TO ENFORCE:
- Tone: {tone}
- Formatting: {format_rule}
- Language: {lang_rule}

CRITICAL INSTRUCTIONS:
1. NO JARGON FOR SENIORS: If the user is over 60, replace words like 'URL' or 'Portal' with 'Website' or 'Link'.
2. LOCAL CONTEXT: Include Mumbai-specific office addresses or landmarks if asked for offline routes.
3. BE DIRECT: Answer the question fully FIRST. Do not output internal thinking.
4. SPACING: Obey the spacing and formatting rules strictly based on the user's age."""


# --------------------------------------------------------------------------
# Groq model selection + call
# --------------------------------------------------------------------------

def get_chat_candidates(client):
    """Priority-ordered list of chat-capable model IDs to try, refreshed at
    most once an hour instead of on every single request. Checks our
    vetted PREFERRED_CHAT_MODELS against what's actually live on the
    account, so a model Groq deprecates just quietly falls out of
    rotation instead of needing a redeploy."""
    now = time.time()
    if _model_cache["ids"] and (now - _model_cache["ts"] < MODEL_CACHE_TTL_SECONDS):
        return _model_cache["ids"]

    try:
        live_ids = {m.id for m in client.models.list().data}
        candidates = [m for m in PREFERRED_CHAT_MODELS if m in live_ids]
        if not candidates:
            # None of our picks are live on this account -- fall back to
            # anything chat-shaped rather than failing outright.
            candidates = [m for m in live_ids if not any(bad in m.lower() for bad in EXCLUDED_MODEL_KEYWORDS)]
        _model_cache["ids"] = candidates
        _model_cache["ts"] = now
        return candidates
    except Exception as e:
        logger.warning("Could not refresh Groq model list: %s", e)
        return _model_cache["ids"] or PREFERRED_CHAT_MODELS


def ask_ai(query, context, profile, language, chat_history):
    if not GROQ_API_KEY:
        return "Please configure GROQ_API_KEY in Render.", None

    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_candidates = get_chat_candidates(client)
        system_prompt = build_system_prompt(profile, language)

        if not context:
            system_prompt += (
                "\n\nNOTE: No live web search results are available for this "
                "answer. Answer from general knowledge, but tell the user to "
                "verify fees, deadlines, and portal URLs on the official site "
                "since these can change."
            )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for msg in chat_history[-CHAT_HISTORY_TURNS:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        if context:
            messages.append({"role": "user", "content": f"WEB CONTEXT:\n{context}\n\nUSER QUESTION: {query}"})
        else:
            messages.append({"role": "user", "content": query})

        for model_id in chat_candidates:
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                )
                return response.choices[0].message.content, model_id
            except Exception as e:
                logger.warning("Model %s failed: %s", model_id, e)
                continue

        if context:
            return f"**Live Information Found:**\n\n{context[:800]}...\n\n*(Note: AI summary is currently unavailable)*", None
        return "No specific details found.", None

    except Exception as e:
        logger.error("ask_ai error: %s", e)
        return f"AI Error: {str(e)}", None


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/", methods=["GET", "HEAD"])
def home():
    return jsonify({"status": "FormSaathi AI Online"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "groq_configured": bool(GROQ_API_KEY)})


@app.route("/ask", methods=["POST"])
def ask():
    start = time.time()

    client_id = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    if is_rate_limited(client_id):
        return jsonify({"error": "Too many requests. Please wait a moment and try again."}), 429

    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        query = (data.get("query") or "").strip()
        mode = data.get("mode") or "standard"
        language = data.get("language") or "auto"
        profile = data.get("profile") or {}
        chat_history = data.get("chat_history") or []

        if not query:
            return jsonify({"error": "Missing query"}), 400
        if len(query) > MAX_QUERY_LENGTH:
            return jsonify({"error": f"Query too long (max {MAX_QUERY_LENGTH} characters)"}), 400

        if mode == "quick":
            context, sources, from_cache = "", [], False
        else:
            context, sources, from_cache = get_context_for_query(query)

        answer, model_used = ask_ai(query, context, profile, language, chat_history)

        logger.info(
            "query=%r mode=%s model=%s cache=%s took=%.2fs",
            query, mode, model_used, from_cache, time.time() - start
        )

        return jsonify({"success": True, "answer": answer, "sources": sources})

    except Exception as e:
        logger.error("ask() error: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)
