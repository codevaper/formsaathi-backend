"""
FormSaathi AI Unified Backend — Model-Cascade Production Build
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

# Keys
DEFAULT_GROQ_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_API_KEY_CHAT = os.environ.get("GROQ_API_KEY_CHAT", "").strip() or DEFAULT_GROQ_KEY
GROQ_API_KEY_VISION = os.environ.get("GROQ_API_KEY_VISION", "").strip() or DEFAULT_GROQ_KEY
GROQ_API_KEY_TUTORIAL = os.environ.get("GROQ_API_KEY_TUTORIAL", "").strip() or DEFAULT_GROQ_KEY

# Limits & Settings
SEARCH_MAX_RESULTS = 2
SCRAPE_TIMEOUT_SECONDS = 2.0  
CHAT_HISTORY_TURNS = 4
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048

_rate_limit_buckets = defaultdict(deque)
_rate_limit_lock = Lock()
RATE_LIMIT_MAX_REQUESTS = 25
RATE_LIMIT_WINDOW_SECONDS = 60

# Model Cascades: If [0] fails, it automatically tries [1], then [2]
PREFERRED_CHAT_MODELS = [
    "llama-3.3-70b-versatile", 
    "llama-3.1-8b-instant", 
    "llama3-8b-8192"
]
VISION_MODELS = [
    "llama-3.2-11b-vision-preview", 
    "llama-3.2-90b-vision-preview"
]

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ======================================================================
# SMART MODEL FALLBACK ENGINE
# ======================================================================
def call_groq_with_fallback(client, models_list, **kwargs):
    """Tries models sequentially. If one is down or rate-limited, tries the next."""
    last_error = None
    for model_name in models_list:
        try:
            kwargs['model'] = model_name
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.warning(f"Model {model_name} failed. Attempting next... Error: {e}")
            last_error = e
            continue
    # If we exhaust the list and all fail, throw the last error
    raise last_error

# ======================================================================
# UTILS & SEARCH
# ======================================================================
def safe_int(value, default):
    try: return int(value)
    except (TypeError, ValueError): return default

def strip_think_tags(text):
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

def search_web_safe(query, max_results=SEARCH_MAX_RESULTS):
    results = []
    try:
        with DDGS(timeout=3) as ddgs: 
            for r in ddgs.text(query, max_results=max_results):
                results.append({"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")})
    except Exception as e:
        logger.warning(f"DuckDuckGo blocked/failed: {e}")
    return results

def scrape_url(url, timeout=SCRAPE_TIMEOUT_SECONDS):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if text and len(text.strip()) > 80: return text.strip()[:1500]
        
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]): tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:1500]
    except Exception: return None

def get_context_for_query(query):
    search_results = search_web_safe(query)
    all_context, sources = [], []
    
    if not search_results: return "", []

    with ThreadPoolExecutor(max_workers=min(len(search_results), 3)) as executor:
        future_to_url = {executor.submit(scrape_url, r["url"]): r["url"] for r in search_results}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                text = future.result()
                if text:
                    title = next((r["title"] for r in search_results if r["url"] == url), "Source")
                    all_context.append(f"[{title}] ({url})\n{text}")
                    sources.append({"title": title, "url": url})
            except Exception: continue
    
    return "\n\n---\n\n".join(all_context), sources

def build_system_prompt(profile, language):
    name = profile.get("name") or "User"
    age = safe_int(profile.get("age"), 30)
    ward = profile.get("ward") or "Mumbai"

    lang_rule = "Respond ONLY in Hindi (Devanagari). Use आप and जी." if language == "hi" else \
                "Respond ONLY in Marathi (Devanagari)." if language == "mr" else \
                "Respond ONLY in Indian English." if language == "en" else "Detect user language."

    base = f"You are FormSaathi, an Indian government assistant for {name} in {ward}, Mumbai.\nCRITICAL RULES:\n1. {lang_rule}\n2. NO <think> TAGS. DO NOT output your reasoning. Answer immediately.\n3. Be helpful."
    if age >= 60: return base + "\n4. Keep it simple. Max 3-4 steps. Mention physical offices."
    else: return base + "\n4. Be concise, digital-first, include links."

# ======================================================================
# API ENDPOINTS
# ======================================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "FormSaathi Backend Online"})

@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS": return "", 204
    client_id = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    if is_rate_limited(client_id): return jsonify({"error": "Too many requests."}), 429

    try:
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query: return jsonify({"error": "Missing query"}), 400

        context, sources = get_context_for_query(query)
        system_prompt = build_system_prompt(data.get("profile") or {}, data.get("language") or "auto")

        messages = [{"role": "system", "content": system_prompt}]
        for msg in (data.get("chat_history") or [])[-CHAT_HISTORY_TURNS:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
        user_msg = f"REFERENCE DATA:\n{context[:2500]}\n\nQUESTION: {query}" if context else query
        messages.append({"role": "user", "content": user_msg})

        client = Groq(api_key=GROQ_API_KEY_CHAT)
        
        # 🔥 Using the new Fallback Engine
        response = call_groq_with_fallback(
            client=client,
            models_list=PREFERRED_CHAT_MODELS,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        clean_answer = strip_think_tags(response.choices[0].message.content)
        return jsonify({"success": True, "answer": clean_answer, "sources": sources})
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": f"Groq AI Error: {str(e)}"}), 500


@app.route("/summarize-doc", methods=["POST", "OPTIONS"])
def summarize_doc():
    if request.method == "OPTIONS": return "", 204
    try:
        data = request.get_json(silent=True) or {}
        extracted_text = (data.get("text") or "").strip()
        if not extracted_text: return jsonify({"error": "No text provided"}), 400

        prompt = f"""Based on this extracted OCR text:
--- START TEXT ---
{extracted_text[:1500]}
--- END TEXT ---

Output ONLY a JSON object with this exact structure. Do not use markdown wrappers.
{{
  "document_type": "Short name (e.g. Aadhaar Card, PAN Card, Income Certificate, Marksheet, Other)",
  "description": "1 clear sentence explaining what this document proves.",
  "actionable_steps": "1 brief sentence on where or how to use this document."
}}"""

        client = Groq(api_key=GROQ_API_KEY_CHAT)
        
        # 🔥 Using the new Fallback Engine
        response = call_groq_with_fallback(
            client=client,
            models_list=PREFERRED_CHAT_MODELS,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )

        raw_response = response.choices[0].message.content.strip()
        summary = {}
        try:
            clean = re.sub(r"^```(?:json)?\n?", "", raw_response)
            clean = re.sub(r"\n?```$", "", clean)
            json_match = re.search(r'\{.*\}', clean, re.DOTALL)
            if json_match: summary = json.loads(json_match.group())
            else: raise ValueError("No JSON object found")
        except Exception:
            summary = {
                "document_type": "Official Document",
                "description": "Analyzed document based on extracted text.",
                "actionable_steps": raw_response[:150]
            }

        return jsonify({"success": True, "summary": summary})
        
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        return jsonify({"error": f"Groq API Error: {str(e)}"}), 500


@app.route("/analyze-document", methods=["POST", "OPTIONS"])
def analyze_document():
    # Only used if you bypass OCR and send raw images to backend
    if request.method == "OPTIONS": return "", 204
    try:
        if "document" not in request.files: return jsonify({"error": "No document"}), 400
        
        file_bytes = request.files["document"].read()
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        pil_img.thumbnail((1024, 1024), Image.LANCZOS)
        
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = """Extract the details from this document. Output ONLY a raw JSON object. Do not use markdown. Do not add explanations.
{
  "document_type": "Aadhaar Card/PAN Card/Voter ID/Certificate/Other",
  "quality": "Good/Fair/Poor",
  "extracted_fields": {"Name": "value", "DOB": "value", "ID Number": "value", "Address": "value"},
  "full_text": "all readable text",
  "what_is_this_document": "short description"
}"""
        
        client = Groq(api_key=GROQ_API_KEY_VISION)
        
        # 🔥 Using the new Fallback Engine for Vision
        response = call_groq_with_fallback(
            client=client,
            models_list=VISION_MODELS,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
            temperature=0.1, max_tokens=1000
        )
        
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
            "file_size_kb": round(len(file_bytes) / 1024, 2),
            "dimensions": f"{pil_img.width}x{pil_img.height}"
        })
    except Exception as e:
        logger.error(f"Doc error: {e}")
        return jsonify({"error": f"Vision API Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), threaded=True)
