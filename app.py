"""
FormSaathi AI Backend — Document Analyzer + Chat + Optimizer
"""

import os, io, time, base64, logging, re, json
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
import trafilatura
from bs4 import BeautifulSoup
from PIL import Image, ImageOps, ImageEnhance
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

SEARCH_MAX_RESULTS = 3
SCRAPE_TIMEOUT_SECONDS = 5
SCRAPE_CHAR_LIMIT = 2000
CHAT_HISTORY_TURNS = 4
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

PREFERRED_CHAT_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
VISION_MODELS = ["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]
EXCLUDED_MODEL_KEYWORDS = ["whisper", "guard", "audio", "embed", "orpheus", "tts", "compound", "gpt-oss", "canopy", "vision"]
MODEL_CACHE_TTL_SECONDS = 60 * 60
_model_cache = {"ids": None, "ts": 0}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def safe_int(value, default):
    try: return int(value)
    except (TypeError, ValueError): return default

def search_web(query, max_results=SEARCH_MAX_RESULTS):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")})
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
    if not results: return scraped
    with ThreadPoolExecutor(max_workers=len(results)) as executor:
        future_to_url = {executor.submit(scrape_url, r["url"], timeout): r["url"] for r in results}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                text = future.result()
                if text: scraped[url] = text
            except Exception: continue
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

def get_chat_models(client):
    now = time.time()
    if _model_cache["ids"] and (now - _model_cache["ts"] < MODEL_CACHE_TTL_SECONDS):
        return _model_cache["ids"]
    try:
        live_ids = {m.id for m in client.models.list().data}
        candidates = [m for m in PREFERRED_CHAT_MODELS if m in live_ids]
        if not candidates:
            candidates = [m.id for m in client.models.list().data if not any(bad in m.id.lower() for bad in EXCLUDED_MODEL_KEYWORDS)]
        _model_cache["ids"] = candidates
        _model_cache["ts"] = now
        return candidates
    except Exception as e:
        logger.warning("Model list refresh failed: %s", e)
        return _model_cache["ids"] or PREFERRED_CHAT_MODELS

def build_system_prompt(profile, language):
    name = profile.get("name") or "User"
    age = safe_int(profile.get("age"), 30)
    ward = profile.get("ward") or "Mumbai"
    experience = profile.get("experience") or "first_time"
    if language == "hi":
        lang_rule = "Respond ONLY in Hindi (Devanagari). Use आप and जी."
    elif language == "mr":
        lang_rule = "Respond ONLY in Marathi (Devanagari)."
    elif language == "en":
        lang_rule = "Respond ONLY in clear Indian English."
    else:
        lang_rule = "Detect the user's language and respond in the same language."
    if age >= 60:
        return f"""You are FormSaathi, a warm government assistant for {name} ji, senior citizen in {ward}, Mumbai.
LANGUAGE: {lang_rule}
- Start with greeting. Short simple sentences. No jargon.
- Max 4-5 numbered steps. Mention nearest physical office with landmark.
- List exact documents to carry. End with reassurance.
- NEVER repeat sentences. NEVER show thinking. Answer directly."""
    elif age <= 34:
        return f"""You are FormSaathi, a fast government tech assistant for {name} in {ward}, Mumbai.
LANGUAGE: {lang_rule}
- No fluff. Bullet points. Digital-first: portals, OTP, DigiLocker.
- Always include: Portal URL, Fee, TAT, Documents.
- NEVER repeat sentences. NEVER show thinking. Under 150 words."""
    else:
        return f"""You are FormSaathi, a professional government assistant for {name} in {ward}, Mumbai.
LANGUAGE: {lang_rule}
- Use headings: Eligibility, Documents, Process, Fees, Timeline.
- Both online and offline options. Include Maharashtra portals.
- NEVER repeat sentences. NEVER show thinking. Be structured."""

def analyze_with_vision(image_base64, user_context=""):
    if not GROQ_API_KEY: return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        context_hint = f"\nUser noted: {user_context}" if user_context else ""
        prompt = f"""Analyze this Indian government document image carefully. Return a strict JSON response with:
{{
  "document_type": "Type of document (Aadhaar, PAN, Voter ID, Driving License, Passport, Income Certificate, Domicile Certificate, Ration Card, Caste Certificate, Birth Certificate, Marksheet, Bank Statement, Application Form, Receipt, Other)",
  "language": "Primary language (Hindi, English, Marathi, Other)",
  "extracted_fields": {{
    "name": "Full name if visible",
    "father_name": "Father/Husband name if visible",
    "dob": "Date of birth if visible",
    "gender": "Gender if visible",
    "address": "Full address if visible",
    "id_number": "Any ID number (Aadhaar, PAN, DL etc)",
    "issue_date": "Issue date if visible",
    "expiry_date": "Expiry date if visible",
    "district": "District if visible",
    "state": "State if visible"
  }},
  "full_text": "Extract ALL readable text from the document exactly as it appears.",
  "quality": "Good/Fair/Poor",
  "issues": ["List any issues: blurry, torn, folded, dark, missing fields"],
  "suggestions": ["Suggestions to improve the document for upload"]
}}{context_hint}"""
        try:
            live_ids = {m.id for m in client.models.list().data}
            vision_models = [m for m in VISION_MODELS if m in live_ids] or VISION_MODELS
        except Exception:
            vision_models = VISION_MODELS
        for model_id in vision_models:
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]}],
                    temperature=0.2,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning("Vision model %s failed: %s", model_id, e)
                continue
        return None
    except Exception as e:
        logger.error("Vision error: %s", e)
        return None

@app.route("/", methods=["GET", "HEAD"])
def home():
    return jsonify({"status": "FormSaathi AI Online", "features": ["chat", "documents", "optimize"]})

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
        system_prompt = build_system_prompt(profile, language)
        if not context:
            system_prompt += "\n\nNote: No live web data available. Answer from knowledge but tell the user to verify fees and deadlines on official portals."
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history[-CHAT_HISTORY_TURNS:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        if context:
            messages.append({"role": "user", "content": f"LIVE WEB DATA:\n{context}\n\nQUESTION: {query}"})
        else:
            messages.append({"role": "user", "content": query})
        client = Groq(api_key=GROQ_API_KEY)
        models = get_chat_models(client)
        answer = None
        model_used = None
        for mid in models:
            try:
                response = client.chat.completions.create(
                    model=mid,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    top_p=0.9,
                    frequency_penalty=0.5,
                    presence_penalty=0.3
                )
                answer = response.choices[0].message.content
                model_used = mid
                break
            except Exception as e:
                logger.warning("Model %s failed: %s", mid, e)
                continue
        if not answer:
            answer = context[:800] + "\n\n*(AI summary unavailable — please read sources below)*" if context else "No details found. Please try rephrasing your question."
        logger.info("query=%r model=%s cache=%s %.2fs", query, model_used, from_cache, time.time() - start)
        return jsonify({"success": True, "answer": answer, "sources": sources})
    except Exception as e:
        logger.error("ask() error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/analyze-document", methods=["POST"])
def analyze_document():
    try:
        if "document" not in request.files:
            return jsonify({"error": "No document uploaded"}), 400
        file = request.files["document"]
        context = request.form.get("context", "")
        file_bytes = file.read()
        file_size_kb = len(file_bytes) / 1024
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        pil_img.thumbnail((1200, 1200), Image.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        vision_result = analyze_with_vision(img_b64, context)
        results = {
            "file_size_kb": round(file_size_kb, 2),
            "dimensions": f"{pil_img.width}x{pil_img.height}",
            "ocr_text": "",
            "document_type": "Unknown",
            "extracted_fields": {},
            "quality": "Unknown"
        }
        if vision_result:
            try:
                json_match = re.search(r'\{.*\}', vision_result, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    results["extracted_fields"] = parsed.get("extracted_fields", {})
                    results["document_type"] = parsed.get("document_type", "Unknown")
                    results["quality"] = parsed.get("quality", "Unknown")
                    results["ocr_text"] = parsed.get("full_text", "")
            except Exception as e:
                logger.warning("JSON parse failed: %s", e)
        return jsonify({"success": True, **results})
    except Exception as e:
        logger.error("analyze error: %s", e)
        return jsonify({"error": str(e)}), 500

def optimize_document(image_bytes, target_kb=200, max_width=1500):
    pil_img = Image.open(io.BytesIO(image_bytes))
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    if pil_img.width > max_width:
        ratio = max_width / pil_img.width
        pil_img = pil_img.resize((max_width, int(pil_img.height * ratio)), Image.LANCZOS)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.2)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.3)
    quality = 95
    buffer = io.BytesIO()
    while quality >= 10:
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality, optimize=True)
        if (buffer.tell() / 1024) <= target_kb:
            break
        quality -= 5
    return buffer.getvalue(), round(buffer.tell() / 1024, 2), pil_img.size

@app.route("/optimize-document", methods=["POST"])
def optimize_document_route():
    try:
        if "document" not in request.files:
            return jsonify({"error": "No document uploaded"}), 400
        file = request.files["document"]
        target_kb = int(request.form.get("target_kb", 200))
        file_bytes = file.read()
        optimized_bytes, final_kb, dimensions = optimize_document(file_bytes, target_kb)
        return jsonify({
            "success": True,
            "original_size_kb": round(len(file_bytes) / 1024, 2),
            "optimized_size_kb": final_kb,
            "dimensions": f"{dimensions[0]}x{dimensions[1]}",
            "optimized_image_base64": base64.b64encode(optimized_bytes).decode("utf-8"),
            "within_limit": final_kb <= target_kb
        })
    except Exception as e:
        logger.error("optimize error: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)
