"""
FormSaathi AI Unified Backend — High-Speed Build
Endpoints: /ask, /summarize-doc, /health
"""
import os, io, time, logging, re, json
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
import trafilatura
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS

from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("formsaathi")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

SEARCH_MAX_RESULTS = 2
SCRAPE_TIMEOUT_SECONDS = 2.0  # Fast timeout to prevent 1+ minute freezes
CHAT_HISTORY_TURNS = 4
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 600          # Keeps response fast (<1.5 seconds)

# Groq's fast model catalog
CHAT_MODEL = "llama-3.1-8b-instant"

_context_cache = {}
_context_cache_lock = Lock()
CONTEXT_CACHE_TTL = 30 * 60

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def safe_int(value, default):
    try: return int(value)
    except (TypeError, ValueError): return default

def strip_think_tags(text):
    if not text: return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

def search_web(query, max_results=SEARCH_MAX_RESULTS):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")})
    except Exception as e:
        logger.warning(f"Search error: {e}")
    return results

def scrape_url(url, timeout=SCRAPE_TIMEOUT_SECONDS):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if text and len(text.strip()) > 80:
                return text.strip()[:1000]
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]): tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:1000]
    except Exception:
        return None

def get_context_for_query(query):
    cache_key = query.strip().lower()
    with _context_cache_lock:
        cached = _context_cache.get(cache_key)
        if cached and (time.time() - cached["ts"] < CONTEXT_CACHE_TTL):
            return cached["context"], cached["sources"], True

    search_results = search_web(query)
    all_context, sources = [], []
    with ThreadPoolExecutor(max_workers=len(search_results) or 1) as executor:
        future_to_url = {executor.submit(scrape_url, r["url"]): r["url"] for r in search_results}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                text = future.result()
                if text:
                    title = next((r["title"] for r in search_results if r["url"] == url), "Gov Portal")
                    all_context.append(f"[{title}] ({url})\n{text}")
                    sources.append({"title": title, "url": url})
            except Exception:
                continue

    if not all_context:
        all_context = [f"[{r['title']}] {r['snippet']}" for r in search_results]
        sources = [{"title": r["title"], "url": r["url"]} for r in search_results]

    context = "\n\n".join(all_context)
    with _context_cache_lock:
        _context_cache[cache_key] = {"context": context, "sources": sources, "ts": time.time()}
    return context, sources, False

def build_system_prompt(profile, language):
    name = profile.get("name") or "User"
    age = safe_int(profile.get("age"), 30)
    ward = profile.get("ward") or "Mumbai"

    lang_rule = "Respond ONLY in Hindi (Devanagari). Use आप and जी." if language == "hi" else \
                "Respond ONLY in Marathi (Devanagari)." if language == "mr" else \
                "Respond ONLY in clear Indian English." if language == "en" else "Respond in the user's language."

    base = f"""You are FormSaathi, an Indian government assistant for {name} in {ward}, Mumbai.
RULES:
1. {lang_rule}
2. Answer immediately in bullet points. NO <think> tags. NO preamble.
3. Include: Portal Link, Fee, Documents, Timeline."""
    if age >= 60:
        return base + "\n4. Very simple words. Max 3-4 steps. Mention nearest ward office."
    return base

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "FormSaathi Backend Online"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "groq": bool(GROQ_API_KEY)})

@app.route("/ask", methods=["POST"])
def ask():
    start = time.time()
    try:
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Missing query"}), 400

        context, sources, _ = get_context_for_query(query)
        system_prompt = build_system_prompt(data.get("profile") or {}, data.get("language") or "auto")

        messages = [{"role": "system", "content": system_prompt}]
        for msg in (data.get("chat_history") or [])[-CHAT_HISTORY_TURNS:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        user_content = f"WEB CONTEXT:\n{context}\n\nQUESTION: {query}" if context else query
        messages.append({"role": "user", "content": user_content})

        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )

        clean_ans = strip_think_tags(resp.choices[0].message.content)
        logger.info("Chat finished in %.2fs", time.time() - start)
        return jsonify({"success": True, "answer": clean_ans, "sources": sources})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": "Service temporarily busy. Please try again."}), 500

# Ultra-Fast Document Context Endpoint (Uses ~150 tokens)
@app.route("/summarize-doc", methods=["POST"])
def summarize_doc():
    try:
        data = request.get_json(silent=True) or {}
        extracted_text = (data.get("text") or "").strip()
        if not extracted_text:
            return jsonify({"error": "No text provided"}), 400

        prompt = f"""You are an Indian government document analyst. Based on this extracted OCR text:
'''{extracted_text[:1200]}'''

Output ONLY a JSON object with this exact structure:
{{
  "document_type": "Short name (e.g. Aadhaar Card, Income Certificate, Marksheet, Electricity Bill, Other)",
  "description": "1 clear sentence explaining what this document proves.",
  "actionable_steps": "1 brief sentence on where or how to use this document."
}}"""

        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"}
        )

        parsed = json.loads(resp.choices[0].message.content)
        return jsonify({"success": True, "summary": parsed})
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        return jsonify({"error": "Failed to analyze document text."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), threaded=True)
