"""
FormSaathi AI Unified Backend
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

# Constants
SEARCH_MAX_RESULTS = 3
SCRAPE_TIMEOUT_SECONDS = 5
SCRAPE_CHAR_LIMIT = 2000
CHAT_HISTORY_TURNS = 4
LLM_TEMPERATURE = 0.1 # Very low to prevent rambling
LLM_MAX_TOKENS = 4096 # High limit to prevent cut-off messages

# Caching & Rate Limiting
_context_cache = {}
_context_cache_lock = Lock()
_rate_limit_buckets = defaultdict(deque)
_rate_limit_lock = Lock()

PREFERRED_CHAT_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
VISION_MODELS = ["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def safe_int(value, default):
    try: return int(value)
    except (TypeError, ValueError): return default

def strip_think_tags(text):
    """Aggressively removes <think> tags, even if they get cut off halfway."""
    if not text: return text
    # Remove fully closed tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove unclosed tags (if max_tokens cuts the message off)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

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
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            if text and len(text.strip()) > 80: return text.strip()[:SCRAPE_CHAR_LIMIT]
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]): tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:SCRAPE_CHAR_LIMIT] if text else None
    except Exception: return None

def get_context_for_query(query):
    search_results = search_web(query)
    all_context, sources = [], []
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
    if not all_context:
        all_context = [f"[{r['title']}] {r['snippet']}" for r in search_results]
        sources = [{"title": r["title"], "url": r["url"]} for r in search_results]
    return "\n\n---\n\n".join(all_context), sources

def build_system_prompt(profile, language):
    name = profile.get("name") or "User"
    age = safe_int(profile.get("age"), 30)
    ward = profile.get("ward") or "Mumbai"

    lang_rule = "Respond ONLY in Hindi (Devanagari). Use आप and जी." if language == "hi" else \
                "Respond ONLY in Marathi (Devanagari)." if language == "mr" else \
                "Respond ONLY in Indian English." if language == "en" else "Detect user's language and respond in the same."

    base = f"""You are FormSaathi, a government assistant for {name} in {ward}.
CRITICAL RULES:
1. {lang_rule}
2. ABSOLUTELY NO <think> TAGS. DO NOT output your reasoning process. Answer immediately.
3. Keep answers direct and helpful."""

    if age >= 60: return base + "\n4. Keep it very simple. Max 3-4 steps. Mention physical offices."
    else: return base + "\n4. Be concise, digital-first, include portal links and fees."

# ==================== ROUTES ====================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "FormSaathi Backend Online"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS": return "", 204
    try:
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query: return jsonify({"error": "Missing query"}), 400

        context, sources = get_context_for_query(query)
        system_prompt = build_system_prompt(data.get("profile") or {}, data.get("language") or "auto")

        messages = [{"role": "system", "content": system_prompt}]
        for msg in (data.get("chat_history") or [])[-CHAT_HISTORY_TURNS:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
        user_msg = f"REFERENCE DATA:\n{context[:3000]}\n\nQUESTION: {query}" if context else query
        messages.append({"role": "user", "content": user_msg})

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=PREFERRED_CHAT_MODELS[0], 
            messages=messages, 
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        # Strip <think> tags safely
        clean_answer = strip_think_tags(response.choices[0].message.content)
        return jsonify({"success": True, "answer": clean_answer, "sources": sources})
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": "AI service temporarily unavailable."}), 500

@app.route("/analyze-document", methods=["POST", "OPTIONS"])
def analyze_document():
    if request.method == "OPTIONS": return "", 204
    try:
        if "document" not in request.files: return jsonify({"error": "No document"}), 400
        
        file_bytes = request.files["document"].read()
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        pil_img.thumbnail((1024, 1024), Image.LANCZOS)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
        
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        client = Groq(api_key=GROQ_API_KEY)
        prompt = """Analyze this document. Return ONLY valid JSON:
{
  "document_type": "Aadhaar/PAN/Voter ID/Certificate/Other",
  "quality": "Good/Fair/Poor",
  "extracted_fields": {
    "name": "full name or Not visible",
    "dob": "date of birth or Not visible",
    "id_number": "ID number masked or Not visible",
    "address": "address or Not visible"
  },
  "full_text": "all visible text",
  "what_is_this_document": "brief description"
}"""
        response = client.chat.completions.create(
            model=VISION_MODELS[0],
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
            temperature=0.1, max_tokens=1000
        )
        
        # Robust parsing to prevent 500 error
        raw_text = response.choices[0].message.content
        clean = raw_text.strip()
        clean = re.sub(r"^```(?:json)?\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean)
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        
        data = json.loads(json_match.group()) if json_match else {}
        return jsonify({
            "success": True, 
            "document_type": data.get("document_type", "Unknown"),
            "quality": data.get("quality", "Fair"),
            "extracted_fields": data.get("extracted_fields", {}),
            "ocr_text": data.get("full_text", ""),
            "file_size_kb": round(len(file_bytes) / 1024, 2),
            "dimensions": f"{pil_img.width}x{pil_img.height}"
        })
    except Exception as e:
        logger.error(f"Doc error: {e}")
        return jsonify({"error": "Failed to parse document"}), 500

@app.route("/optimize-document", methods=["POST", "OPTIONS"])
def optimize_document():
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
            "original_size_kb": round(len(file.read()) / 1024, 2),
            "optimized_size_kb": round(buf.tell() / 1024, 2),
            "dimensions": f"{pil_img.width}x{pil_img.height}",
            "optimized_image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "within_limit": (buf.tell() / 1024) <= target_kb
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), threaded=True)
