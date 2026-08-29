"""
FormSaathi AI backend.
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
from duckduckgo_search import DDGS
from groq import Groq
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("formsaathi")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

SEARCH_MAX_RESULTS = 3
SCRAPE_TIMEOUT_SECONDS = 5
SCRAPE_CHAR_LIMIT = 2000
CHAT_HISTORY_TURNS = 6
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024
MAX_QUERY_LENGTH = 1000

# Cache search context to save API rate limits
CONTEXT_CACHE_TTL_SECONDS = 30 * 60
_context_cache = {}
_context_cache_lock = Lock()

# Simple sliding-window rate limiter (15 requests/min per IP)
RATE_LIMIT_MAX_REQUESTS = 25
RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_buckets = defaultdict(deque)
_rate_limit_lock = Lock()

PREFERRED_CHAT_MODELS = [
    "llama-3.3-70b-specdec",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]
EXCLUDED_MODEL_KEYWORDS = ["whisper", "guard", "audio", "embed", "orpheus", "vision", "tts", "compound"]
MODEL_CACHE_TTL_SECONDS = 60 * 60
_model_cache = {"ids": None, "ts": 0}

app = Flask(__name__)
# Force wide-open CORS so browsers never block requests
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
        logger.warning("Search failed for %r: %s", query, e)
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

def build_system_prompt(profile, language):
    name = profile.get("name") or "User"
    age = safe_int(profile.get("age"), 30)
    ward = profile.get("ward") or "Mumbai"
    experience = profile.get("experience") or "first_time"
    if experience not in ("first_time", "some", "experienced"):
        experience = "first_time"

    if experience == "first_time":
        exp_detail = "NOVICE: The user has never navigated government forms. Define basic terms, explain the 'why', and guide them through the process."
    elif experience == "some":
        exp_detail = "INTERMEDIATE: The user knows the basics. Skip elementary definitions but provide clear procedural steps."
    else:
        exp_detail = "EXPERT: The user is highly familiar with government tasks. Skip all explanations. Provide only necessary endpoints, URLs, exact document lists, and fees."

    if age >= 60:
        if experience == "first_time":
            tone = "Extremely warm, patient, and respectful (Namaste/Pranam). Use very simple language. Prioritize OFFLINE methods (physical office locations, landmarks in Mumbai). Break instructions into small, digestible numbered steps."
        else:
            tone = "Respectful (Namaste) and clear. Provide both online links and offline office details in Mumbai. Keep sentences short, readable, and highly polite."
    elif age >= 35:
        if experience == "first_time":
            tone = "Professional, structured, and helpful. Explain how to navigate portals (like Aaple Sarkar) step-by-step. Use clear headings and avoid bureaucratic jargon."
        else:
            tone = "Highly concise and professional. Focus strictly on turnaround times (TAT), exact fees, and direct portal links. Do not waste time on pleasantries."
    else:
        if experience == "first_time":
            tone = "Modern, friendly, and encouraging. Recommend digital-first solutions (DigiLocker, mParivahan, online portals). Fast-paced but explanatory."
        else:
            tone = "Ultra-crisp, fast, and direct. Zero fluff. Provide checklist formats, direct URLs, and API-like efficiency."

    if language == "hi":
        lang_rule = "ALWAYS respond purely in Hindi (Devanagari script). Use respectful pronouns (आप, जी)."
    elif language == "mr":
        lang_rule = "ALWAYS respond purely in Marathi (Devanagari script). Use a warm, culturally respectful tone appropriate for Maharashtra."
    elif language == "en":
        lang_rule = "ALWAYS respond in clear Indian English."
    else:
        lang_rule = "AUTO-DETECT the language of the user's question and respond in the EXACT same language (Marathi for Marathi, Hindi for Hindi, etc.)."

    return f"""You are FormSaathi AI, an intelligent Indian government document and scheme assistant for Mumbai, Maharashtra.

USER PROFILE:
- Name: {name}
- Age: {age} years old
- Location: {ward}, Mumbai, Maharashtra
- Background: {exp_detail}

COMMUNICATION STYLE TO ENFORCE:
- Tone & Strategy: {tone}
- Language: {lang_rule}

CRITICAL INSTRUCTIONS:
1. ADAPT TO THE USER: Strictly apply the Tone and Background rules defined above. Do not talk to a 20-year-old expert the same way you talk to a 70-year-old novice.
2. BE DIRECT: Answer the question fully and honestly FIRST. Do not refuse to answer.
3. LOCAL CONTEXT: Include Mumbai-specific office addresses, landmarks, and Maharashtra portals (aaplesarkar.mahaonline.gov.in, mumbaicity.gov.in) when relevant.
4. PHOTO REQUIREMENT: If a form requires a passport photo, remind them they can use the FormSaathi Photo Resizer tab.
5. NO INTERNAL MONOLOGUE: Do not output your thinking process. Just output the final, tailored response directly to {name}.
6. ELIGIBILITY: If applicable, add a brief eligibility note ONLY at the very end of your response."""

def get_chat_candidates(client):
    now = time.time()
    if _model_cache["ids"] and (now - _model_cache["ts"] < MODEL_CACHE_TTL_SECONDS):
        return _model_cache["ids"]

    try:
        live_ids = {m.id for m in client.models.list().data}
        candidates = [m for m in PREFERRED_CHAT_MODELS if m in live_ids]
        if not candidates:
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
