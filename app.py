"""
FormSaathi AI Unified Backend — Production Build
Features: AI Chat, Fast Web Scraper, Document Analyzer (Vision), Fast Document Summarizer (OCR Text), Document Optimizer, Interactive Tutorial Generator.
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

# ======================================================================
# MULTI-KEY ARCHITECTURE (Bypasses Rate Limits)
# ======================================================================
DEFAULT_GROQ_KEY = os.environ.get("GROQ_API_KEY", "").strip()

GROQ_API_KEY_CHAT = os.environ.get("GROQ_API_KEY_CHAT", "").strip() or DEFAULT_GROQ_KEY
GROQ_API_KEY_VISION = os.environ.get("GROQ_API_KEY_VISION", "").strip() or DEFAULT_GROQ_KEY
GROQ_API_KEY_TUTORIAL = os.environ.get("GROQ_API_KEY_TUTORIAL", "").strip() or DEFAULT_GROQ_KEY

# ======================================================================
# CONFIGURATION & CONSTANTS
# ======================================================================
SEARCH_MAX_RESULTS = 3
SCRAPE_TIMEOUT_SECONDS = 2.0  # Ultra-fast timeout to prevent 1+ minute freezes
SCRAPE_CHAR_LIMIT = 2000
CHAT_HISTORY_TURNS = 4

# Chat Limits
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 4096  # High limit prevents half-messages

# Caching & Rate Limiting
_context_cache = {}
_context_cache_lock = Lock()
CONTEXT_CACHE_TTL = 30 * 60

_rate_limit_buckets = defaultdict(deque)
_rate_limit_lock = Lock()
RATE_LIMIT_MAX_REQUESTS = 25
RATE_LIMIT_WINDOW_SECONDS = 60

# STRICT Model Definitions (Prevents DeepSeek <think> tag hallucinations)
PREFERRED_CHAT_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
VISION_MODELS = ["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# ======================================================================
# UTILITIES
# ======================================================================
def safe_int(value, default):
    try: return int(value)
    except (TypeError, ValueError): return default

def strip_think_tags(text):
    """Aggressively removes <think> tags, even if they get cut off halfway."""
    if not text: return text
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

def is_rate_limited(client_id):
    now = time.time()
    with _rate_limit_lock:
        bucket = _rate_limit_buckets[client_id]
        while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_MAX_REQUESTS: return True
        bucket.append(now)
        return False


# ======================================================================
# WEB SEARCH & SCRAPING ENGINE
# ======================================================================
def search_web(query, max_results=SEARCH_MAX_RESULTS):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")})
    except Exception as e:
        logger.warning(f"Search failed: {e}")
    return results

def scrape_url(url, timeout=SCRAPE_TIMEOUT_SECONDS):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if text and len(text.strip()) > 80: return text.strip()[:SCRAPE_CHAR_LIMIT]
        
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]): tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:SCRAPE_CHAR_LIMIT]
    except Exception: return None

def get_context_for_query(query):
    cache_key = query.strip().lower()
    with _context_cache_lock:
        cached = _context_cache.get(cache_key)
        if cached and (time.time() - cached["ts"] < CONTEXT_CACHE_TTL):
            return cached["context"], cached["sources"], True

    search_results = search_web(query)
    all_context, sources = [], []
    with ThreadPoolExecutor(max_workers=min(len(search_results) or 1, 3)) as executor:
        future_to_url = {executor.submit(scrape_url, r["url"]): r["url"] for r in search_results}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                text = future.result()
                if text:
                    title = next((r["title"] for r in search_results if r["url"] == url), "Gov Portal")
                    all_context.append(f"[{title}] ({url})\n{text}")
                    sources.append({"title": title, "url": url})
            except Exception: continue
    
    if not all_context:
        all_context = [f"[{r['title']}] {r['snippet']}" for r in search_results]
        sources = [{"title": r["title"], "url": r["url"]} for r in search_results]

    context = "\n\n---\n\n".join(all_context)
    with _context_cache_lock:
        _context_cache[cache_key] = {"context": context, "sources": sources, "ts": time.time()}
    return context, sources, False


# ======================================================================
# CHAT PROMPT GENERATOR
# ======================================================================
def build_system_prompt(profile, language):
    name = profile.get("name") or "User"
    age = safe_int(profile.get("age"), 30)
    ward = profile.get("ward") or "Mumbai"

    lang_rule = "Respond ONLY in Hindi (Devanagari). Use आप and जी." if language == "hi" else \
                "Respond ONLY in Marathi (Devanagari)." if language == "mr" else \
                "Respond ONLY in Indian English." if language == "en" else "Detect user's language and respond in the same."

    base = f"""You are FormSaathi, a government assistant for {name} in {ward}, Mumbai.
CRITICAL RULES:
1. {lang_rule}
2. ABSOLUTELY NO <think> TAGS. DO NOT output your reasoning process. Answer immediately.
3. Keep answers direct. Use bullet points."""

    if age >= 60: return base + "\n4. Keep it very simple. Max 3-4 steps. Mention physical offices and landmarks."
    else: return base + "\n4. Be concise, digital-first, include portal links and exact fees."


# ======================================================================
# API ENDPOINTS
# ======================================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "FormSaathi Multi-Key Backend Online"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    """Main AI Chat Assistant"""
    if request.method == "OPTIONS": return "", 204
    client_id = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    if is_rate_limited(client_id): return jsonify({"error": "Too many requests."}), 429

    try:
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query: return jsonify({"error": "Missing query"}), 400

        context, sources, _ = get_context_for_query(query)
        system_prompt = build_system_prompt(data.get("profile") or {}, data.get("language") or "auto")

        messages = [{"role": "system", "content": system_prompt}]
        for msg in (data.get("chat_history") or [])[-CHAT_HISTORY_TURNS:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
        user_msg = f"REFERENCE DATA:\n{context[:3000]}\n\nQUESTION: {query}" if context else query
        messages.append({"role": "user", "content": user_msg})

        # 🔑 Uses Chat API Key
        client = Groq(api_key=GROQ_API_KEY_CHAT)
        response = client.chat.completions.create(
            model=PREFERRED_CHAT_MODELS[0], 
            messages=messages, 
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        clean_answer = strip_think_tags(response.choices[0].message.content)
        return jsonify({"success": True, "answer": clean_answer, "sources": sources})
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": "AI service temporarily unavailable. Please try again."}), 500


@app.route("/summarize-doc", methods=["POST", "OPTIONS"])
def summarize_doc():
    """Ultra-Fast Endpoint to summarize text extracted from browser Tesseract OCR"""
    if request.method == "OPTIONS": return "", 204
    try:
        data = request.get_json(silent=True) or {}
        extracted_text = (data.get("text") or "").strip()
        
        if not extracted_text: return jsonify({"error": "No text provided"}), 400

        prompt = f"""You are an Indian government document analyst. Based on this extracted OCR text:
--- START TEXT ---
{extracted_text[:1500]}
--- END TEXT ---

Output ONLY a JSON object with this exact structure. Do not use markdown wrappers.
{{
  "document_type": "Short name (e.g. Aadhaar Card, Income Certificate, Marksheet, Electricity Bill, Other)",
  "description": "1 clear sentence explaining what this document proves.",
  "actionable_steps": "1 brief sentence on where or how to use this document."
}}"""

        # 🔑 Uses Chat API Key (because it's text-only, not Vision)
        client = Groq(api_key=GROQ_API_KEY_CHAT)
        resp = client.chat.completions.create(
            model=PREFERRED_CHAT_MODELS[1], # Use fast 8B model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )

        raw_response = resp.choices[0].message.content.strip()
        
        # Extremely robust JSON parser (prevents 500 errors)
        summary = {}
        try:
            clean = re.sub(r"^```(?:json)?\n?", "", raw_response)
            clean = re.sub(r"\n?```$", "", clean)
            json_match = re.search(r'\{.*\}', clean, re.DOTALL)
            if json_match:
                summary = json.loads(json_match.group())
            else:
                raise ValueError("No JSON object found")
        except Exception as parse_err:
            logger.warning(f"Summary JSON parsing failed. Using fallback. Error: {parse_err}")
            summary = {
                "document_type": "Official Document",
                "description": "Analyzed document based on extracted text.",
                "actionable_steps": raw_response[:200]
            }

        return jsonify({"success": True, "summary": summary})
        
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        return jsonify({"error": "Failed to analyze document text."}), 500


@app.route("/analyze-document", methods=["POST", "OPTIONS"])
def analyze_document():
    """Deep Groq Vision Analysis directly from uploaded image"""
    if request.method == "OPTIONS": return "", 204
    try:
        if "document" not in request.files: return jsonify({"error": "No document"}), 400
        
        file_bytes = request.files["document"].read()
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        
        # Scale to 1024px to prevent Groq API Payload Size rejections
        pil_img.thumbnail((1024, 1024), Image.LANCZOS)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
        
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = """Extract the details from this document. Output ONLY a raw JSON object. Do not use markdown. Do not add explanations.
{
  "document_type": "Aadhaar Card/PAN Card/Voter ID/Certificate/Other",
  "quality": "Good/Fair/Poor",
  "extracted_fields": {"Name": "value", "DOB": "value", "ID Number": "value", "Address": "value"},
  "full_text": "all readable text",
  "what_is_this_document": "short description",
  "where_to_submit": "portal name"
}"""
        
        # 🔑 Uses Vision API Key
        client = Groq(api_key=GROQ_API_KEY_VISION)
        response = client.chat.completions.create(
            model=VISION_MODELS[0],
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
            temperature=0.1, max_tokens=1000
        )
        
        # Robust parsing to prevent 500 error
        raw_text = strip_think_tags(response.choices[0].message.content.strip())
        data = {}
        try:
            clean = re.sub(r"^```(?:json)?\n?", "", raw_text)
            clean = re.sub(r"\n?```$", "", clean)
            json_match = re.search(r'\{.*\}', clean, re.DOTALL)
            if json_match: data = json.loads(json_match.group())
        except Exception:
            data = {"document_type": "Document", "quality": "Fair", "full_text": raw_text[:500]}
            
        return jsonify({
            "success": True, 
            "document_type": data.get("document_type", "Scanned Document"),
            "quality": data.get("quality", "Fair"),
            "extracted_fields": data.get("extracted_fields", {}),
            "ocr_text": data.get("full_text", raw_text[:500]),
            "what_is_this_document": data.get("what_is_this_document", ""),
            "where_to_submit": data.get("where_to_submit", ""),
            "file_size_kb": round(len(file_bytes) / 1024, 2),
            "dimensions": f"{pil_img.width}x{pil_img.height}"
        })
    except Exception as e:
        logger.error(f"Doc error: {e}")
        return jsonify({"error": "Failed to parse document. Please ensure the image is clear."}), 500


@app.route("/optimize-document", methods=["POST", "OPTIONS"])
def optimize_document():
    """Client Upload -> Server Resize -> Client Download"""
    if request.method == "OPTIONS": return "", 204
    try:
        file = request.files["document"]
        target_kb = int(request.form.get("target_kb", 200))
        
        pil_img = Image.open(io.BytesIO(file.read()))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        
        if pil_img.width > 1200:
            pil_img = pil_img.resize((1200, int(pil_img.height * (1200/pil_img.width))), Image.LANCZOS)
        
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.2)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.3)
        
        q = 95
        buf = io.BytesIO()
        while q >= 10:
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=q, optimize=True)
            if (buf.tell() / 1024) <= target_kb: break
            q -= 5
            
        return jsonify({
            "success": True,
            "optimized_size_kb": round(buf.tell() / 1024, 2),
            "dimensions": f"{pil_img.width}x{pil_img.height}",
            "optimized_image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "within_limit": (buf.tell() / 1024) <= target_kb
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate-tutorial", methods=["POST", "OPTIONS"])
def generate_tutorial():
    """Generates JSON array of steps for the interactive frontend tutorial player"""
    if request.method == "OPTIONS": return "", 204
    try:
        data = request.get_json(silent=True) or {}
        form_name = data.get("form_name", "Aadhaar Card Application Form")
        language = data.get("language") or "en"

        if language == "hi": prompt_lang = "Hindi (Devanagari script)"
        elif language == "mr": prompt_lang = "Marathi (Devanagari script)"
        else: prompt_lang = "simple Indian English"

        prompt = f"""Create a step-by-step form-filling tutorial for: {form_name}.
Instructions must be in {prompt_lang}.

Output ONLY a JSON list of exactly 5 objects. No markdown.
[
  {{
    "step_number": 1,
    "field_label": "Name",
    "instruction": "Short instruction",
    "voiceover_text": "What to speak out loud",
    "x_pct": 20,
    "y_pct": 10,
    "sample_input": "RAM KUMAR"
  }}
]"""
        # 🔑 Uses Tutorial API Key
        client = Groq(api_key=GROQ_API_KEY_TUTORIAL)
        response = client.chat.completions.create(
            model=PREFERRED_CHAT_MODELS[1], # Fast 8B model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=1000
        )
        
        raw_text = strip_think_tags(response.choices[0].message.content.strip())
        
        try:
            clean = re.sub(r"^```(?:json)?\n?", "", raw_text)
            clean = re.sub(r"\n?```$", "", clean)
            json_match = re.search(r'\[.*\]', clean, re.DOTALL)
            if json_match:
                steps = json.loads(json_match.group())
                return jsonify({"success": True, "steps": steps})
        except Exception: pass
        
        return jsonify({"error": "Failed to generate structured tutorial."}), 500
    except Exception as e:
        logger.error(f"Tutorial error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), threaded=True)
