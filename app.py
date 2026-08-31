"""
FormSaathi AI Unified Backend — Auto-Adapting Production Build
"""

import os, io, time, base64, logging, re, json
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
LLM_MAX_TOKENS = 4000  # Increased heavily for detailed thinking & output

# Updated to Groq's new free tier & open-source models
PREFERRED_CHAT_MODELS = [
    "llama-3.3-70b-versatile",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b", 
    "openai/gpt-oss-20b"
]

_rate_limit_buckets = defaultdict(deque)
_rate_limit_lock = Lock()
RATE_LIMIT_MAX_REQUESTS = 25
RATE_LIMIT_WINDOW_SECONDS = 60

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ======================================================================
# SMART MODEL FALLBACK ENGINES
# ======================================================================
def call_groq_with_fallback(client, preferred_models, **kwargs):
    try:
        live_models_data = client.models.list().data
        live_ids = {m.id for m in live_models_data}
        available_models = [m for m in preferred_models if m in live_ids]
        if not available_models:
            available_models = [m.id for m in live_models_data if "whisper" not in m.id and "vision" not in m.id]
    except Exception as e:
        logger.warning(f"Could not fetch live models: {e}")
        available_models = preferred_models

    last_error = None
    for model_name in available_models:
        try:
            kwargs['model'] = model_name
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.warning(f"Chat model {model_name} failed. Error: {e}")
            last_error = e
            continue
    raise last_error

def call_groq_vision_with_fallback(client, messages, **kwargs):
    try:
        live_models_data = client.models.list().data
        # Auto-detect ANY active vision model currently hosted by Groq
        live_ids = [m.id for m in live_models_data if "vision" in m.id.lower()]
        if not live_ids:
            live_ids = ["llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]
    except Exception as e:
        logger.warning(f"Could not fetch live vision models: {e}")
        live_ids = ["llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]

    last_error = None
    for model_name in live_ids:
        try:
            kwargs['model'] = model_name
            return client.chat.completions.create(messages=messages, **kwargs)
        except Exception as e:
            logger.warning(f"Vision model {model_name} failed. Error: {e}")
            last_error = e
            continue
    raise last_error

# ======================================================================
# UTILS & SEARCH
# ======================================================================
def safe_int(value, default):
    try: return int(value)
    except (TypeError, ValueError): return default

def strip_think_tags(text):
    if not text: return text
    # Strip full closed <think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Strip any dangling literal <think> or </think> tags to prevent UI breakage
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
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
        logger.warning(f"DuckDuckGo blocked: {e}")
    return results

def scrape_url(url, timeout=SCRAPE_TIMEOUT_SECONDS):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if text and len(text.strip()) > 80: return text.strip()[:1500]
        
        headers = {"User-Agent": "Mozilla/5.0"}
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
    experience = profile.get("experience") or "first_time"

    lang_rule = "Respond ONLY in Hindi (Devanagari). Use आप and जी." if language == "hi" else \
                "Respond ONLY in Marathi (Devanagari)." if language == "mr" else \
                "Respond ONLY in Indian English." if language == "en" else "Detect user language."

    base = f"You are FormSaathi, an Indian government assistant for {name} in {ward}, Mumbai.\nCRITICAL RULES:\n1. {lang_rule}\n2. NO <think> TAGS. DO NOT output your reasoning.\n3. Formatting: Use Markdown nicely.\n4. REDACT AADHAAR NUMBERS ALWAYS."
    
    if experience in ("expert", "experienced") and age < 60:
        return base + "\n5. User is highly experienced. Be ultra-crisp, provide exact portal links, and TAT. Skip hand-holding."
    elif age >= 60: 
        return base + "\n5. Keep it simple. Max 3-4 steps. Mention physical offices."
    else: 
        return base + "\n5. Be concise, digital-first, include links."

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
        
        response = call_groq_with_fallback(
            client=client,
            preferred_models=PREFERRED_CHAT_MODELS,
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

Output ONLY a JSON object with this exact structure.
{{
  "document_type": "Short name",
  "description": "1 clear sentence explaining what this document proves.",
  "actionable_steps": "1 brief sentence on where or how to use this document."
}}"""

        client = Groq(api_key=GROQ_API_KEY_CHAT)
        response = call_groq_with_fallback(
            client=client,
            preferred_models=PREFERRED_CHAT_MODELS,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=LLM_MAX_TOKENS
        )

        raw_response = response.choices[0].message.content.strip()
        raw_response = strip_think_tags(raw_response)
        
        try:
            start_idx = raw_response.find('{')
            end_idx = raw_response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_string = raw_response[start_idx:end_idx+1]
                summary = json.loads(json_string)
            else:
                raise ValueError("No braces found")
        except Exception as e:
            logger.error(f"Summarize JSON Parse Error: {e}")
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
    if request.method == "OPTIONS": return "", 204
    try:
        if "document" not in request.files:
            return jsonify({"error": "No document uploaded"}), 400
            
        file = request.files["document"]
        context = request.form.get("context", "Analyze this document.")
        
        image_bytes = file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = file.content_type or "image/jpeg"
        
        client = Groq(api_key=GROQ_API_KEY_VISION)
        
        prompt = f"""You are an expert AI assistant helping Indian citizens with paperwork.
Look at this document image carefully. 
User's prompt: "{context}"

CRITICAL RULE: If you see an Aadhaar number, you MUST redact it as XXXX XXXX XXXX. Do not output the real digits.

Analyze the image and return ONLY a JSON object with these exact keys. No markdown wrappers around the JSON.
{{
  "document_type": "Name of document (e.g. PAN, Aadhaar, Utility Bill)",
  "quality": "Legibility (e.g. Clear, Blurry)",
  "extracted_fields": {{
      "AI Analysis": "Write a highly detailed, friendly, and comprehensive paragraph here. Explain what the document is, what information it contains, and directly answer the User's prompt. Make it incredibly helpful."
  }},
  "ocr_text": "Transcribe the raw text visible in the document exactly as it appears. REDACT AADHAAR NUMBERS."
}}"""
        
        response = call_groq_vision_with_fallback(
            client=client,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        raw_response = response.choices[0].message.content.strip()
        raw_response = strip_think_tags(raw_response)
        
        try:
            start_idx = raw_response.find('{')
            end_idx = raw_response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_string = raw_response[start_idx:end_idx+1]
                result_data = json.loads(json_string)
            else:
                raise ValueError("No braces found in AI response")
        except Exception as e:
            logger.error(f"Vision JSON Parse Error: {e}")
            # Absolute Bulletproof Fallback: Dump the raw, un-truncated response into the AI Analysis block!
            result_data = {
                "document_type": "Analyzed Document",
                "quality": "Processed",
                "extracted_fields": {
                    "AI Analysis": raw_response 
                },
                "ocr_text": "Raw text extracted and passed to chat." 
            }
            
        return jsonify(result_data)
        
    except Exception as e:
        logger.error(f"Vision error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), threaded=True)
