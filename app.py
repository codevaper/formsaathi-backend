"""
FormSaathi AI Unified Backend — Production Build
Endpoints: /ask, /analyze-document, /optimize-document, /generate-tutorial
No OpenCV. No NumPy. No Tesseract. No Rembg. Ultra-light for Render.
"""

import os, io, time, base64, logging, re, json, math
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

# Configuration
SEARCH_MAX_RESULTS = 3
SCRAPE_TIMEOUT_SECONDS = 5
SCRAPE_CHAR_LIMIT = 2000
CHAT_HISTORY_TURNS = 4
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 800
MAX_QUERY_LENGTH = 1000

# Caching & Rate Limiting
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
_model_cache = {"ids": None, "ts": 0}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def safe_int(value, default):
    try: return int(value)
    except (TypeError, ValueError): return default


# ======================================================================
# WEB SEARCH + SCRAPE (Threaded)
# ======================================================================
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


# ======================================================================
# DYNAMIC MODEL DISCOVERY
# ======================================================================
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


# ======================================================================
# AGE-SPECIFIC SYSTEM PROMPTS (Full 3-Tier Restored)
# ======================================================================
def build_system_prompt(profile, language):
    name = profile.get("name") or "User"
    age = safe_int(profile.get("age"), 30)
    ward = profile.get("ward") or "Mumbai"

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


# ======================================================================
# SMART DOCUMENT VISION ANALYZER WITH ROBUST PARSING (Refined 1024px Engine)
# ======================================================================
def analyze_with_vision(image_base64, user_context=""):
    if not GROQ_API_KEY:
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        context_hint = f"\nUser context: {user_context}" if user_context else ""

        prompt = f"""You are an expert Indian government document analyzer. Extract the text and verify this document image.

If this is an Aadhaar Card, PAN Card, Voter ID, Income Certificate, or Class 10 (X) Marksheet, make sure you extract the corresponding identifier keys with high precision.

Return ONLY a valid JSON object. Do not include markdown wraps or conversational introduction. Start directly with the open bracket.

{{
  "document_type": "Aadhaar Card / Income Certificate / Class X Marksheet / PAN Card / Voter ID / Passport / Driving License / Unknown",
  "quality": "Good / Fair / Poor",
  "name": "Full Name as printed on the document or Not visible",
  "father_or_spouse_name": "Father or Spouse name or Not visible",
  "date_of_birth": "DOB (e.g. DD/MM/YYYY) or Not visible",
  "gender": "Male / Female / Other / Not visible",
  "address": "Full address or Not visible",
  "id_number": "Show only the last 4 digits (e.g. XXXX-XXXX-1234 for Aadhaar, Certificate Numbers, or Roll Numbers)",
  "full_text": "Extract ALL readable letters and numbers verbatim from the document for OCR lookup.",
  "what_is_this_document": "A 1-sentence simple description of what this document is and what government authority issued it.",
  "where_to_submit": "Tell the user exactly which portals or local government desks usually require this document."
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
                    max_tokens=1024
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning("Vision model %s failed: %s", model_id, e)
                continue
        return None
    except Exception as e:
        logger.error("Vision error: %s", e)
        return None


def robust_json_parser(raw_text):
    """Fallback Regex parsing parser to prevent JSON structure anomalies"""
    parsed = {}
    if not raw_text:
        return parsed

    try:
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.warning("JSON parsing anomaly. Starting regex parser.")

    try:
        type_match = re.search(r'"document_type"\s*:\s*"([^"]+)"', raw_text)
        parsed["document_type"] = type_match.group(1) if type_match else "Unknown Document"

        # Auto classification if LLM drops out
        if "aadhaar" in raw_text.lower():
            parsed["document_type"] = "Aadhaar Card"
        elif "income" in raw_text.lower() or "tahsildar" in raw_text.lower():
            parsed["document_type"] = "Income Certificate"
        elif "marksheet" in raw_text.lower() or "secondary school" in raw_text.lower() or "marks statement" in raw_text.lower():
            parsed["document_type"] = "Class X Marksheet"

        qual_match = re.search(r'"quality"\s*:\s*"([^"]+)"', raw_text)
        parsed["quality"] = qual_match.group(1) if qual_match else "Fair"

        what_match = re.search(r'"what_is_this_document"\s*:\s*"([^"]+)"', raw_text)
        parsed["what_is_this_document"] = what_match.group(1) if what_match else "Identified government issued document."

        where_match = re.search(r'"where_to_submit"\s*:\s*"([^"]+)"', raw_text)
        parsed["where_to_submit"] = where_match.group(1) if where_match else "Verify submission endpoints on official guidelines."

        text_match = re.search(r'"full_text"\s*:\s*"([^"]+)"', raw_text)
        parsed["full_text"] = text_match.group(1) if text_match else raw_text[:500]

        parsed["extracted_fields"] = {}
        for k in ["name", "father_or_spouse_name", "date_of_birth", "gender", "address", "id_number"]:
            v_match = re.search(fr'"{k}"\s*:\s*"([^"]+)"', raw_text)
            if v_match:
                parsed["extracted_fields"][k] = v_match.group(1)

    except Exception as err:
        logger.error("All parsers failed: %s", err)
        parsed["document_type"] = "Unreadable Image"
        parsed["full_text"] = raw_text[:200]

    return parsed


# ======================================================================
# ROUTES
# ======================================================================
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
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        mode = data.get("mode") or "standard"
        language = data.get("language") or "auto"
        profile = data.get("profile") or {}
        chat_history = data.get("chat_history") or []

        if not query:
            return jsonify({"error": "Missing query"}), 400

        context, sources, from_cache = get_context_for_query(query) if mode != "quick" else ("", [], False)
        system_prompt = build_system_prompt(profile, language)

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
                    model=mid, messages=messages,
                    temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS,
                    top_p=0.9, frequency_penalty=0.5, presence_penalty=0.3
                )
                answer = response.choices[0].message.content
                model_used = mid
                break
            except Exception as e:
                logger.warning("Model %s failed: %s", mid, e)
                continue

        if not answer:
            answer = context[:800] if context else "Please try rephrasing."

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

        # 🧠 KEY TILE RESOLUTION MATCH (1024px width max dimension)
        # Prevents Llama 3.2 Vision from tiling images too large, while retaining razor text
        pil_img.thumbnail((1024, 1024), Image.LANCZOS)

        # Enhance Sharpness and Contrast for tiny numbers & ink stamps
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.3)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75) 
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        vision_result = analyze_with_vision(img_b64, context)
        logger.info("Vision analysis completed.")

        # Unified Frontend Mapping template
        base_result = {
            "file_size_kb": round(file_size_kb, 2),
            "dimensions": f"{pil_img.width}x{pil_img.height}",
            "document_type": "Unknown Document",
            "quality": "Unknown",
            "extracted_fields": {},
            "full_text": "",
            "what_is_this_document": "Processing failed.",
            "where_to_submit": ""
        }

        if vision_result:
            parsed = robust_json_parser(vision_result)
            
            base_result["document_type"] = parsed.get("document_type", "Unknown Document")
            base_result["quality"] = parsed.get("quality", "Fair")
            base_result["full_text"] = parsed.get("full_text", "")
            base_result["what_is_this_document"] = parsed.get("what_is_this_document", "")
            base_result["where_to_submit"] = parsed.get("where_to_submit", "")
            
            # Map fields properly to frontend DocumentTools
            fields = parsed.get("extracted_fields", {})
            base_result["extracted_fields"] = {
                "Name": fields.get("name", parsed.get("name", "Not visible")),
                "Relative/Spouse Name": fields.get("father_or_spouse_name", parsed.get("father_or_spouse_name", "Not visible")),
                "DOB": fields.get("date_of_birth", parsed.get("date_of_birth", "Not visible")),
                "Gender": fields.get("gender", parsed.get("gender", "Not visible")),
                "ID Number": fields.get("id_number", parsed.get("id_number", "Not visible")),
                "Address": fields.get("address", parsed.get("address", "Not visible"))
            }

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


# ======================================================================
# TUTORIAL VIDEO GENERATOR
# ======================================================================
@app.route("/generate-tutorial", methods=["POST"])
def generate_tutorial():
    try:
        data = request.get_json(silent=True) or {}
        form_name = data.get("form_name", "Aadhaar Card Application Form")
        language = data.get("language") or "en"

        client = Groq(api_key=GROQ_API_KEY)

        if language == "hi":
            prompt_lang = "Hindi (Devanagari script)"
            system_instruction = "Always give steps and instructions in Hindi."
        elif language == "mr":
            prompt_lang = "Marathi (Devanagari script)"
            system_instruction = "Always give steps and instructions in Marathi."
        else:
            prompt_lang = "simple Indian English"
            system_instruction = "Always give steps and instructions in English."

        prompt = f"""Create a highly detailed, step-by-step form-filling tutorial for the: {form_name}.
The tutorial must be returned strictly in JSON format. Generate exactly 5-6 steps to fill out this form.

Provide instructions in {prompt_lang}. Keep text crisp and short.

Return ONLY a valid JSON list of objects with the exact structure (no markdown conversational wrappers, start directly with open square bracket):
[
  {{
    "step_number": 1,
    "field_label": "Field/Section name (e.g. 'Full Name' or 'Candidate Name')",
    "instruction": "Short clear direction on how to write it (e.g., 'Write your name in CAPITAL LETTERS as shown in your leaving certificate.')",
    "voiceover_text": "What the narrator will speak out loud to guide the user.",
    "x_pct": X coordinate of this field on a virtual page (0 to 100),
    "y_pct": Y coordinate of this field on a virtual page (0 to 100),
    "sample_input": "An example value of what to write (e.g., 'ARUN SHARMA')"
  }}
]

Make sure coordinates are logically distributed (e.g., step 1 near top of page, step 6 near bottom)."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if json_match:
            steps = json.loads(json_match.group())
            return jsonify({"success": True, "steps": steps})
        
        return jsonify({"error": "Failed to generate structured tutorial."}), 500

    except Exception as e:
        logger.error("tutorial error: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)
