"""
FormSaathi AI Backend — Final Production Build
Smart Document Analyzer + AI Chat + Optimizer
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

PREFERRED_CHAT_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]
VISION_MODELS = [
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview"
]
EXCLUDED_MODEL_KEYWORDS = [
    "whisper", "guard", "audio", "embed", "orpheus",
    "tts", "compound", "gpt-oss", "canopy", "vision"
]
MODEL_CACHE_TTL_SECONDS = 60 * 60
_model_cache = {"ids": None, "ts": 0}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def safe_int(value, default):
    try: return int(value)
    except (TypeError, ValueError): return default


# ══════════════════════════════════════════════
# SEARCH + SCRAPE
# ══════════════════════════════════════════════
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


# ══════════════════════════════════════════════
# MODEL MANAGEMENT
# ══════════════════════════════════════════════
def get_chat_models(client):
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
        logger.warning("Model refresh failed: %s", e)
        return _model_cache["ids"] or PREFERRED_CHAT_MODELS


# ══════════════════════════════════════════════
# SYSTEM PROMPTS
# ══════════════════════════════════════════════
def build_system_prompt(profile, language):
    name = profile.get("name") or "User"
    age = safe_int(profile.get("age"), 30)
    ward = profile.get("ward") or "Mumbai"

    if language == "hi":
        lang_rule = "Respond ONLY in Hindi (Devanagari). Use आप and जी."
    elif language == "mr":
        lang_rule = "Respond ONLY in Marathi (Devanagari)."
    elif language == "en":
        lang_rule = "Respond ONLY in clear Indian English."
    else:
        lang_rule = "Detect the user's language and respond in the same language. Never mix."

    if age >= 60:
        return f"""You are FormSaathi, a warm and patient government assistant for {name} ji, senior citizen in {ward}, Mumbai.
LANGUAGE: {lang_rule}
- Start with greeting. Short simple sentences. Max 4-5 numbered steps.
- Mention nearest physical office with landmark. List exact documents to carry.
- End with reassurance. NEVER say "I am an AI". Answer directly."""
    elif age <= 34:
        return f"""You are FormSaathi, a fast government tech assistant for {name} in {ward}, Mumbai.
LANGUAGE: {lang_rule}
- No fluff. Bullet points. Digital-first: portals, DigiLocker, UMANG.
- Include: Portal URL, Fee, TAT, Documents. Under 150 words.
- NEVER say "I am an AI". Answer directly."""
    else:
        return f"""You are FormSaathi, a professional government assistant for {name} in {ward}, Mumbai.
LANGUAGE: {lang_rule}
- Headings: Eligibility, Documents, Process, Fees, Timeline.
- Both online and offline options. Maharashtra portals when relevant.
- NEVER say "I am an AI". Answer directly."""


# ══════════════════════════════════════════════
# SMART DOCUMENT VISION ANALYZER
# ══════════════════════════════════════════════
def analyze_with_vision(image_base64, user_context=""):
    if not GROQ_API_KEY:
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        context_hint = f"\nUser context: {user_context}" if user_context else ""

        prompt = f"""You are an expert Indian government document analyzer. Analyze this document image deeply.

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{{
  "document_type": "Exact document name (e.g. Aadhaar Card, PAN Card, Voter ID, Ration Card, Income Certificate, Domicile Certificate, Caste Certificate, Birth Certificate, Driving License, Passport, Electricity Bill, Bank Passbook, Marksheet, College Form, Job Application Form, Government Scheme Form, Unknown)",
  "document_category": "ID Proof / Address Proof / Income Proof / Educational / Application Form / Government Scheme / Financial / Unknown",
  "language": "Hindi / English / Marathi / Mixed",
  "quality": "Good / Fair / Poor",
  "is_filled": true or false,
  "extracted_fields": {{
    "name": "full name or Not visible",
    "father_name": "father name or Not visible",
    "date_of_birth": "DOB or Not visible",
    "gender": "gender or Not visible",
    "address": "full address or Not visible",
    "id_number": "ID number masked last 4 digits or Not visible",
    "issue_date": "issue date or Not visible",
    "expiry_date": "expiry date or Not visible",
    "mobile": "mobile number or Not visible",
    "email": "email or Not visible"
  }},
  "full_text": "Every single word visible in this document extracted verbatim",
  "completeness_check": {{
    "missing_fields": ["list any blank required fields visible in the form"],
    "missing_signature": true or false,
    "missing_photo": true or false,
    "missing_stamp": true or false
  }},
  "what_is_this_document": "2-3 sentence plain English explanation of what this document is and what it proves",
  "what_to_do_next": [
    "Step 1: specific actionable instruction",
    "Step 2: specific actionable instruction",
    "Step 3: specific actionable instruction"
  ],
  "where_to_submit": "Exact office or portal where this document should be submitted",
  "important_warnings": ["Any warnings about the document condition, expiry, or missing info"],
  "portal_url": "Official portal URL if applicable or null"
}}{context_hint}

Be precise. If you cannot read something clearly, say Not visible. Never hallucinate field values."""

        try:
            live_ids = {m.id for m in client.models.list().data}
            vision_models = [m for m in VISION_MODELS if m in live_ids] or VISION_MODELS
        except Exception:
            vision_models = VISION_MODELS

        for model_id in vision_models:
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }}
                        ]
                    }],
                    temperature=0.1,
                    max_tokens=1500
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning("Vision model %s failed: %s", model_id, e)
                continue
        return None
    except Exception as e:
        logger.error("Vision error: %s", e)
        return None


# ══════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════
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
            return jsonify({"error": f"Query too long"}), 400

        if mode == "quick":
            context, sources, from_cache = "", [], False
        else:
            context, sources, from_cache = get_context_for_query(query)

        system_prompt = build_system_prompt(profile, language)
        if not context:
            system_prompt += "\n\nNote: No live web data. Answer from knowledge, advise user to verify on official portals."

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
            answer = context[:800] + "\n\n*(AI unavailable — read sources below)*" if context else "Please rephrase your question."

        logger.info("ask model=%s cache=%s %.2fs", model_used, from_cache, time.time() - start)
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
        pil_img.thumbnail((1400, 1400), Image.LANCZOS)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=88)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        vision_result = analyze_with_vision(img_b64, context)

        base_result = {
            "file_size_kb": round(file_size_kb, 2),
            "dimensions": f"{pil_img.width}x{pil_img.height}",
            "document_type": "Unknown",
            "document_category": "Unknown",
            "language": "Unknown",
            "quality": "Unknown",
            "is_filled": False,
            "extracted_fields": {},
            "full_text": "",
            "completeness_check": {
                "missing_fields": [],
                "missing_signature": False,
                "missing_photo": False,
                "missing_stamp": False
            },
            "what_is_this_document": "",
            "what_to_do_next": [],
            "where_to_submit": "",
            "important_warnings": [],
            "portal_url": None
        }

        if vision_result:
            try:
                # Strip markdown code fences if present
                clean = vision_result.strip()
                if clean.startswith("```"):
                    clean = re.sub(r"^```(?:json)?\n?", "", clean)
                    clean = re.sub(r"\n?```$", "", clean)

                json_match = re.search(r'\{.*\}', clean, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    base_result.update({
                        "document_type": parsed.get("document_type", "Unknown"),
                        "document_category": parsed.get("document_category", "Unknown"),
                        "language": parsed.get("language", "Unknown"),
                        "quality": parsed.get("quality", "Unknown"),
                        "is_filled": parsed.get("is_filled", False),
                        "extracted_fields": parsed.get("extracted_fields", {}),
                        "full_text": parsed.get("full_text", ""),
                        "completeness_check": parsed.get("completeness_check", base_result["completeness_check"]),
                        "what_is_this_document": parsed.get("what_is_this_document", ""),
                        "what_to_do_next": parsed.get("what_to_do_next", []),
                        "where_to_submit": parsed.get("where_to_submit", ""),
                        "important_warnings": parsed.get("important_warnings", []),
                        "portal_url": parsed.get("portal_url", None)
                    })
            except Exception as e:
                logger.warning("JSON parse failed: %s | raw: %s", e, vision_result[:200])
                base_result["full_text"] = vision_result[:1000]

        return jsonify({"success": True, **base_result})

    except Exception as e:
        logger.error("analyze error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/optimize-document", methods=["POST"])
def optimize_document_route():
    try:
        if "document" not in request.files:
            return jsonify({"error": "No document uploaded"}), 400

        file = request.files["document"]
        target_kb = safe_int(request.form.get("target_kb"), 200)
        file_bytes = file.read()

        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")

        if pil_img.width > 1500:
            ratio = 1500 / pil_img.width
            pil_img = pil_img.resize((1500, int(pil_img.height * ratio)), Image.LANCZOS)

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

        final_bytes = buffer.getvalue()
        final_kb = round(len(final_bytes) / 1024, 2)

        return jsonify({
            "success": True,
            "original_size_kb": round(len(file_bytes) / 1024, 2),
            "optimized_size_kb": final_kb,
            "dimensions": f"{pil_img.width}x{pil_img.height}",
            "optimized_image_base64": base64.b64encode(final_bytes).decode("utf-8"),
            "within_limit": final_kb <= target_kb
        })

    except Exception as e:
        logger.error("optimize error: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)
