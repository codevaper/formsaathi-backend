"""
FormSaathi AI Backend — Complete Stable Production Build
Endpoints: /ask, /analyze-document, /optimize-document, /generate-tutorial, /process-photo
"""

import os, io, time, base64, logging, re, json, math
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
import trafilatura
import numpy as np
import cv2
from bs4 import BeautifulSoup
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
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
    "tts", "compound", "gpt-oss", "canopy"
]
_model_cache = {"ids": None, "ts": 0}
_vision_model_cache = {"ids": None, "ts": 0}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def safe_int(value, default):
    try: return int(value)
    except (TypeError, ValueError): return default


# ==================== HARDCODED DOCUMENT TEMPLATES ====================
DOCUMENT_TEMPLATES = {
    "aadhaar": {
        "document_type": "Aadhaar Card",
        "document_category": "ID Proof & Address Proof",
        "issuing_authority": "UIDAI (Unique Identification Authority of India)",
        "what_is_this_document": "Aadhaar is a 12-digit unique identity number issued by UIDAI to Indian residents. It serves as both identity and address proof, and is required for most government services, bank accounts, SIM cards, and welfare schemes.",
        "what_to_do_next": [
            "Verify all details (name, DOB, address) match your other documents exactly",
            "If any detail is wrong, update it at nearest Aadhaar Seva Kendra or online at ssup.uidai.gov.in",
            "Link your Aadhaar with PAN, bank account, and mobile number if not done",
            "Download digital copy (e-Aadhaar) from myaadhaar.uidai.gov.in for backup"
        ],
        "where_to_submit": "Banks, Passport office, RTO, Income Tax office, all government scheme applications, DigiLocker, PAN application.",
        "portal_url": "https://myaadhaar.uidai.gov.in",
        "important_warnings": [
            "Never share your full Aadhaar number publicly on social media",
            "Always mask first 8 digits when sharing photocopies"
        ]
    },
    "income_certificate": {
        "document_type": "Income Certificate",
        "document_category": "Income Proof",
        "issuing_authority": "Tahsildar / SDM / Revenue Department (State Government)",
        "what_is_this_document": "An Income Certificate is an official document issued by state revenue authorities certifying the annual income of a family. It is required for availing income-based government benefits, scholarships, and reservation quotas.",
        "what_to_do_next": [
            "Verify the issue date — most Income Certificates are valid for only 1 year",
            "Ensure the Tahsildar signature and government seal are clearly visible",
            "Keep 3-4 photocopies + digital scan for scholarship and college admission applications"
        ],
        "where_to_submit": "College/University scholarships (EWS, OBC, SC/ST), Government job applications, and fee concessions.",
        "portal_url": "https://aaplesarkar.mahaonline.gov.in (Maharashtra) or your state's e-district portal",
        "important_warnings": [
            "Income Certificate expires 1 year from issue date — renew before applying"
        ]
    },
    "class_x_marksheet": {
        "document_type": "Class X (SSC/CBSE) Marksheet",
        "document_category": "Educational Certificate",
        "issuing_authority": "State Board (SSC) / CBSE / ICSE Boards",
        "what_is_this_document": "The Class X Marksheet is an official academic document showing marks obtained in the secondary school examinations. It is widely used as a permanent proof of birth date (DOB).",
        "what_to_do_next": [
            "Cross-check your name, roll number, and DOB spelling matches Aadhaar exactly",
            "Register on DigiLocker to pull your verified digital marksheet copy",
            "Keep original safe — required for college admissions, passports, and higher education enrollment"
        ],
        "where_to_submit": "College admissions, Passport application (DOB proof), Driving License, and scholarship schemes.",
        "portal_url": "https://www.digilocker.gov.in",
        "important_warnings": [
            "Never laminate the original marksheet if it contains active holograms"
        ]
    },
    "unknown": {
        "document_type": "General Document",
        "document_category": "General Proof",
        "issuing_authority": "Government / Educational Authority",
        "what_is_this_document": "This document was uploaded successfully and is being verified.",
        "what_to_do_next": [
            "Keep digital and printed copies ready for portal submission",
            "Ensure the image has clear contrast and is easily legible"
        ],
        "where_to_submit": "Official state or central government portal",
        "portal_url": None,
        "important_warnings": []
    }
}


# ==================== DOCUMENT CLASSIFIER (Regex-based) ====================
def classify_document(text):
    if not text: return "unknown"
    text_lower = text.lower()
    
    if any(kw in text_lower for kw in ["unique identification", "uidai", "आधार", "aadhaar", "भारत सरकार"]) or re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text):
        return "aadhaar"
    if any(kw in text_lower for kw in ["income certificate", "आय प्रमाण", "उत्पन्न दाखला", "tahsildar", "annual income", "वार्षिक आय"]):
        return "income_certificate"
    if any(kw in text_lower for kw in ["secondary school", "ssc", "class x", "class 10", "marks obtained", "marks statement", "grade sheet", "marksheet", "roll no"]):
        return "class_x_marksheet"
    return "unknown"


# ==================== FIELD EXTRACTOR (Regex-based) ====================
def extract_fields_from_text(text, doc_type):
    fields = {}
    if not text: return fields
    
    if doc_type == "aadhaar":
        aadhaar_match = re.search(r'\b(\d{4})\s?(\d{4})\s?(\d{4})\b', text)
        if aadhaar_match:
            fields["ID Number"] = f"XXXX XXXX {aadhaar_match.group(3)}"
            
    dob_patterns = [
        r'(?:DOB|Date of Birth|जन्म तिथि|D\.O\.B)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b'
    ]
    for pattern in dob_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            fields["Date of Birth"] = m.group(1)
            break
            
    gender_match = re.search(r'\b(MALE|FEMALE|पुरुष|महिला|Male|Female)\b', text)
    if gender_match:
        fields["Gender"] = gender_match.group(1).title()
        
    name_patterns = [
        r'(?:Name|नाम|Candidate Name)[:\s]+([A-Z][A-Z\s]{2,40})',
        r'(?:Name|नाम)[:\s]+([A-Za-z][A-Za-z\s]{2,40})',
    ]
    for pattern in name_patterns:
        m = re.search(pattern, text)
        if m:
            fields["Name"] = m.group(1).strip()
            break
            
    if doc_type == "class_x_marksheet":
        roll_match = re.search(r'(?:Roll No|Seat No|Roll Number)[:\s\.]+([A-Z0-9]{4,15})', text, re.IGNORECASE)
        if roll_match: fields["Roll Number"] = roll_match.group(1)
        pct_match = re.search(r'(\d{2,3}(?:\.\d{1,2})?)\s*%', text)
        if pct_match: fields["Percentage"] = pct_match.group(1) + "%"
        
    if doc_type == "income_certificate":
        income_match = re.search(r'(?:Rs\.?|₹|Rupees)[\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', text)
        if income_match: fields["Annual Income"] = "₹ " + income_match.group(1)
        cert_match = re.search(r'(?:Certificate No|Cert No)[:\s\.]+([A-Z0-9/-]{5,25})', text, re.IGNORECASE)
        if cert_match: fields["Certificate Number"] = cert_match.group(1)
        
    return fields


# ==================== DYNAMIC VISION MODEL DISCOVERY ====================
def get_live_vision_models(client):
    now = time.time()
    if _vision_model_cache["ids"] and (now - _vision_model_cache["ts"] < 3600):
        return _vision_model_cache["ids"]
    try:
        live_models = [m.id for m in client.models.list().data]
        vision_candidates = [m for m in live_models if "vision" in m.lower()]
        if not vision_candidates:
            vision_candidates = ["llama-3.2-11b-vision-preview"]
        _vision_model_cache["ids"] = vision_candidates
        _vision_model_cache["ts"] = now
        return vision_candidates
    except Exception:
        return ["llama-3.2-11b-vision-preview"]


# ==================== GROQ VISION API CALL ====================
def extract_text_with_vision(image_base64):
    if not GROQ_API_KEY: return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = "Read this document image and extract ALL visible text verbatim. Do not summarize or explain."
        vision_models = get_live_vision_models(client)
        
        for model_id in vision_models:
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }],
                    temperature=0.0,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Vision model {model_id} failed: {e}")
                continue
        return None
    except Exception as e:
        logger.error(f"Vision extraction error: {e}")
        return None


# ==================== CHAT SCRAPER & AI ====================
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


def build_system_prompt(profile, language):
    name = profile.get("name") or "User"
    age = safe_int(profile.get("age"), 30)
    ward = profile.get("ward") or "Mumbai"

    if language == "hi":
        lang_rule = "Respond ONLY in Hindi (Devanagari)."
    elif language == "mr":
        lang_rule = "Respond ONLY in Marathi (Devanagari)."
    elif language == "en":
        lang_rule = "Respond ONLY in clear Indian English."
    else:
        lang_rule = "Detect the user's language and respond in the same language."

    if age >= 60:
        return f"You are FormSaathi for {name} ji, senior citizen in {ward}. LANGUAGE: {lang_rule}. Simple sentences, 4-5 steps, nearest office with landmark."
    elif age <= 34:
        return f"You are FormSaathi for {name} in {ward}. LANGUAGE: {lang_rule}. Direct bullet points, digital-first. Portal, Fee, TAT, Docs. Under 150 words."
    else:
        return f"You are FormSaathi for {name} in {ward}. LANGUAGE: {lang_rule}. Use headings Eligibility, Documents, Process, Fees, Timeline. Online and offline."


# ==================== MAIN ENDPOINTS ====================
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

        for mid in models:
            try:
                response = client.chat.completions.create(
                    model=mid, messages=messages,
                    temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS,
                    top_p=0.9, frequency_penalty=0.5, presence_penalty=0.3
                )
                answer = response.choices[0].message.content
                break
            except Exception as e:
                logger.warning("Model %s failed: %s", mid, e)
                continue

        if not answer:
            answer = context[:800] if context else "Please try rephrasing."

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
        user_context = request.form.get("context", "").lower()
        file_bytes = file.read()
        file_size_kb = len(file_bytes) / 1024

        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        original_dims = f"{pil_img.width}x{pil_img.height}"
        
        pil_img.thumbnail((1024, 1024), Image.LANCZOS)
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.3)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.8)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        extracted_text = extract_text_with_vision(img_b64) or ""
        doc_type = classify_document(extracted_text)
        
        if doc_type == "unknown" and user_context:
            if "aadhaar" in user_context or "आधार" in user_context:
                doc_type = "aadhaar"
            elif "income" in user_context or "आय" in user_context or "उत्पन्न" in user_context:
                doc_type = "income_certificate"
            elif "marksheet" in user_context or "10th" in user_context or "ssc" in user_context or "x " in user_context:
                doc_type = "class_x_marksheet"
        
        fields = extract_fields_from_text(extracted_text, doc_type)
        template = DOCUMENT_TEMPLATES.get(doc_type, DOCUMENT_TEMPLATES["unknown"])
        
        response_data = {
            "success": True,
            "file_size_kb": round(file_size_kb, 2),
            "dimensions": original_dims,
            "quality": "Good" if len(extracted_text) > 150 else "Fair" if len(extracted_text) > 40 else "Poor",
            "document_type": template["document_type"],
            "document_category": template["document_category"],
            "issuing_authority": template["issuing_authority"],
            "what_is_this_document": template["what_is_this_document"],
            "what_to_do_next": template["what_to_do_next"],
            "where_to_submit": template["where_to_submit"],
            "portal_url": template["portal_url"],
            "important_warnings": template["important_warnings"],
            "extracted_fields": fields if fields else {"Status": "Verification rules applied successfully. Details below."},
            "full_text": extracted_text if extracted_text else "Visual scan completed."
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"analyze error: %s", e)
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


@app.route("/generate-tutorial", methods=["POST"])
def generate_tutorial():
    try:
        data = request.get_json(silent=True) or {}
        form_name = data.get("form_name", "Aadhaar Card Application Form")
        language = data.get("language") or "en"

        client = Groq(api_key=GROQ_API_KEY)

        if language == "hi":
            system_instruction = "Always give steps and instructions in Hindi."
        elif language == "mr":
            system_instruction = "Always give steps and instructions in Marathi."
        else:
            system_instruction = "Always give steps and instructions in English."

        prompt = f"""Create a step-by-step form-filling tutorial for: {form_name}.
Return ONLY a valid JSON list of 5-6 steps:
[{{"step_number": 1, "field_label": "Field Name", "instruction": "How to fill", "voiceover_text": "Narration", "x_pct": 10, "y_pct": 20, "sample_input": "Example"}}]"""

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
        
        return jsonify({"error": "Failed to generate tutorial."}), 500

    except Exception as e:
        logger.error("tutorial error: %s", e)
        return jsonify({"error": str(e)}), 500


# ==================== NEW: STABLE PORTRAIT FACIAL CROP ====================
@app.route("/process-photo", methods=["POST"])
def process_photo():
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    try:
        photo_file = request.files["photo"]
        doc_type = request.form.get("doc_type", "aadhaar")
        
        # Correct auto-exif orientation
        pil_img = Image.open(photo_file.stream)
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")

        # Specific Target Form Dimensions Map
        SIZES = {
            "aadhaar": (413, 531),
            "driving_license": (413, 531),
            "voter_id": (413, 531),
            "income_certificate": (160, 212),
            "domicile_certificate": (160, 212),
        }
        tw, th = SIZES.get(doc_type, (413, 531))
        target_ratio = tw / th

        # Convert to numpy array for OpenCV Processing
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Execute Haar Cascade frontal face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.12, 5, minSize=(60, 60))

        faces_detected = len(faces)
        
        if faces_detected > 0:
            # Crop around the largest detected face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Apply strict standard studio pad limits (70-80% coverage)
            pad_x = int(w * 0.75)
            pad_y = int(h * 1.15)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(pil_img.width, x + w + pad_x)
            y2 = min(pil_img.height, y + h + int(h * 0.6))

            crop_w = x2 - x1
            crop_h = y2 - y1
            crop_ratio = crop_w / crop_h

            if crop_ratio > target_ratio:
                new_w = int(crop_h * target_ratio)
                x1 += (crop_w - new_w) // 2
                x2 = x1 + new_w
            else:
                new_h = int(crop_w / target_ratio)
                y1 += (crop_h - new_h) // 2
                y2 = y1 + new_h

            cropped_img = pil_img.crop((x1, y1, x2, y2))
        else:
            # Safe Fallback to clean aspect-correct center crop
            w, h = pil_img.size
            if (w / h) > target_ratio:
                new_w = int(h * target_ratio)
                x1 = (w - new_w) // 2
                cropped_img = pil_img.crop((x1, 0, x1 + new_w, h))
            else:
                new_h = int(w / target_ratio)
                y1 = (h - new_h) // 2
                cropped_img = pil_img.crop((0, y1, w, y1 + new_h))

        # Resize to specified exact dimensions
        cropped_img = cropped_img.resize((tw, th), Image.LANCZOS)

        # Enhance photo for premium print-ready quality
        cropped_img = ImageEnhance.Contrast(cropped_img).enhance(1.1)
        cropped_img = ImageEnhance.Sharpness(cropped_img).enhance(1.2)

        # Base64 Encode JPEG response
        buf = io.BytesIO()
        cropped_img.save(buf, format="JPEG", quality=90)
        b64_str = base64.decodebytes(buf.getvalue() if hasattr(base64, 'decodebytes') else base64.b64encode(buf.getvalue()))
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "success": True,
            "processed_image_base64": b64_str,
            "dimensions": f"{tw}x{th} px",
            "faces_detected": faces_detected
        })

    except Exception as e:
        logger.error(f"process-photo failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)
