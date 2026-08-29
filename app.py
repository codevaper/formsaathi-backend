"""
FormSaathi AI Backend — Final Production Build
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
    from ddgs import DDGS

from groq import Groq
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("formsaathi")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

SEARCH_MAX_RESULTS = 3
SCRAPE_TIMEOUT_SECONDS = 5
SCRAPE_CHAR_LIMIT = 2000
CHAT_HISTORY_TURNS = 4

# HIGH temperature prevents repetition loops
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 800
MAX_QUERY_LENGTH = 1000

CONTEXT_CACHE_TTL_SECONDS = 30 * 60
_context_cache = {}
_context_cache_lock = Lock()

RATE_LIMIT_MAX_REQUESTS = 25
RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_buckets = defaultdict(deque)
_rate_limit_lock = Lock()

# ONLY proven, stable models — no experimental ones
PREFERRED_CHAT_MODELS = [
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile",
    "llama-3.3-70b-specdec",
    "llama-3.1-8b-instant"
]
EXCLUDED_MODEL_KEYWORDS = [
    "whisper", "guard", "audio", "embed", "orpheus",
    "vision", "tts", "compound", "gpt-oss", "canopy"
]
MODEL_CACHE_TTL_SECONDS = 60 * 60
_model_cache = {"ids": None, "ts": 0}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
        logger.warning("Search failed: %s", e)
    return results


def scrape_url(url, timeout=SCRAPE_TIMEOUT_SECONDS):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            if text and len(text.strip()) > 80:
                return text.strip()[:SCRAPE_CHAR_LIMIT]
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


# ======================================================================
# AGE-SPECIFIC SYSTEM PROMPTS (The Brain of FormSaathi)
# ======================================================================

def build_system_prompt(profile, language):
    name = profile.get("name") or "User"
    age = safe_int(profile.get("age"), 30)
    ward = profile.get("ward") or "Mumbai"
    experience = profile.get("experience") or "first_time"

    # --- Language Rule (shared across all ages) ---
    if language == "hi":
        lang_rule = "Respond ONLY in Hindi (Devanagari). Use आप and जी. Never mix English words unless they are official portal names."
    elif language == "mr":
        lang_rule = "Respond ONLY in Marathi (Devanagari). Use warm, respectful Maharashtrian tone."
    elif language == "en":
        lang_rule = "Respond ONLY in clear Indian English."
    else:
        lang_rule = "Detect the user's language from their question. If they write in Hindi, reply in Hindi. If Marathi, reply in Marathi. If English, reply in English. Never mix languages in one response."

    # --- Senior Prompt (60+) ---
    if age >= 60:
        return f"""You are FormSaathi, a warm and patient government assistant for {name} ji, a senior citizen in {ward}, Mumbai.

LANGUAGE: {lang_rule}

HOW TO ANSWER:
- Start with "🙏 नमस्ते {name} ji" or equivalent greeting in the response language
- Use very short, simple sentences. No jargon. No abbreviations.
- Give maximum 4-5 numbered steps. Each step = one action only.
- Always mention the NEAREST physical office with a landmark. Example: "Andheri West BMC office, DN Nagar metro ke paas"
- List exact documents to carry: "Saath mein ye le jaayein: 1) Aadhaar card asli, 2) 2 photocopy"
- End with a reassuring line: "Ghabraiye mat, ye kaam bahut aasaan hai."
- If a photo is needed, say: "Photo lagane ke liye upar Photo Resizer tab use karein"

STRICT RULES:
- NEVER repeat the same sentence twice
- NEVER show your thinking or reasoning process
- NEVER say "I am an AI" or "as a language model"
- Answer the question directly. Do not ask clarifying questions.
- If you don't know something specific, say so honestly and suggest visiting the nearest Seva Kendra."""

    # --- Youth Prompt (18-34) ---
    elif age <= 34:
        return f"""You are FormSaathi, a fast and direct government tech assistant for {name}, a young user in {ward}, Mumbai.

LANGUAGE: {lang_rule}

HOW TO ANSWER:
- No greetings, no fluff. Start with the answer immediately.
- Use bullet points and bold text for key info.
- Focus on DIGITAL-FIRST: online portals, OTP verification, DigiLocker, mParivahan, UMANG app.
- Always include: Portal URL, Processing Fee, Turnaround Time (TAT), Required Documents.
- Format example:
  **Portal:** nvsp.in
  **Fee:** ₹0
  **TAT:** 7 working days
  **Docs:** Aadhaar, Address Proof, Photo
- If a photo is needed: "Use the Photo Resizer tab above for compliant photos."

STRICT RULES:
- NEVER repeat the same sentence twice
- NEVER show your thinking or reasoning process
- NEVER say "I am an AI" or "as a language model"
- Keep total response under 150 words unless the process genuinely needs more detail.
- Answer the question directly. Do not ask clarifying questions."""

    # --- Standard Prompt (35-59) ---
    else:
        return f"""You are FormSaathi, a professional government document assistant for {name} in {ward}, Mumbai.

LANGUAGE: {lang_rule}

HOW TO ANSWER:
- Start with a brief, polite greeting.
- Use clear headings: **Eligibility**, **Documents**, **Process**, **Fees**, **Timeline**.
- Provide both online AND offline options.
- Include Maharashtra-specific portals: aaplesarkar.mahaonline.gov.in, rto.maharashtra.gov.in
- Mention the nearest ward office in {ward} when offline steps are involved.
- If a photo is needed: "You can prepare your photo using the Photo Resizer tab."

STRICT RULES:
- NEVER repeat the same sentence twice
- NEVER show your thinking or reasoning process
- NEVER say "I am an AI" or "as a language model"
- Be comprehensive but structured. Use formatting to make it scannable.
- Answer the question directly. Do not ask clarifying questions."""


def get_chat_candidates(client):
    now = time.time()
    if _model_cache["ids"] and (now - _model_cache["ts"] < MODEL_CACHE_TTL_SECONDS):
        return _model_cache["ids"]

    try:
        live_ids = {m.id for m in client.models.list().data}
        candidates = [m for m in PREFERRED_CHAT_MODELS if m in live_ids]
        if not candidates:
            candidates = [
                m.id for m in client.models.list().data
                if not any(bad in m.id.lower() for bad in EXCLUDED_MODEL_KEYWORDS)
            ]
        _model_cache["ids"] = candidates
        _model_cache["ts"] = now
        return candidates
    except Exception as e:
        logger.warning("Model list refresh failed: %s", e)
        return _model_cache["ids"] or PREFERRED_CHAT_MODELS


def ask_ai(query, context, profile, language, chat_history):
    if not GROQ_API_KEY:
        return "Please configure GROQ_API_KEY in Render.", None

    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_candidates = get_chat_candidates(client)
        system_prompt = build_system_prompt(profile, language)

        if not context:
            system_prompt += "\n\nNote: No live web data available. Answer from knowledge but tell the user to verify fees and deadlines on official portals."

        messages = [{"role": "system", "content": system_prompt}]

        if chat_history:
            for msg in chat_history[-CHAT_HISTORY_TURNS:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        if context:
            messages.append({"role": "user", "content": f"LIVE WEB DATA:\n{context}\n\nQUESTION: {query}"})
        else:
            messages.append({"role": "user", "content": query})

        for model_id in chat_candidates:
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    top_p=0.9,
                    frequency_penalty=0.5,
                    presence_penalty=0.3
                )
                return response.choices[0].message.content, model_id
            except Exception as e:
                logger.warning("Model %s failed: %s", model_id, e)
                continue

        if context:
            return f"**Live Data Found:**\n\n{context[:800]}...\n\n*(AI summary unavailable — please read sources below)*", None
        return "No details found. Please try rephrasing your question.", None

    except Exception as e:
        logger.error("ask_ai error: %s", e)
        return f"AI Error: {str(e)}", None


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
        return jsonify({"error": "Too many requests. Please wait."}), 429

    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        query = (data.get("query") or "").strip()
        mode = data.get("mode") or "standard"
        language = data.get("language") or "auto"
        profile = data.get("profile") or {}
        chat_history = data.get("chat_history") or []

        if not query:
            return jsonify({"error": "Missing query"}), 400
        if len(query) > MAX_QUERY_LENGTH:
            return jsonify({"error": f"Query too long (max {MAX_QUERY_LENGTH})"}), 400

        if mode == "quick":
            context, sources, from_cache = "", [], False
        else:
            context, sources, from_cache = get_context_for_query(query)

        answer, model_used = ask_ai(query, context, profile, language, chat_history)

        logger.info("query=%r model=%s cache=%s %.2fs", query, model_used, from_cache, time.time() - start)
        return jsonify({"success": True, "answer": answer, "sources": sources})

    except Exception as e:
        logger.error("ask() error: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)
