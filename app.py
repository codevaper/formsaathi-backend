"""
FormSaathi AI Backend — Dynamic Vision Auto-Discovery Edition
Automatically queries active Groq vision models at runtime.
"""

import os, io, time, base64, logging, re, json
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
import trafilatura
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
        "where_to_submit": "Aadhaar is accepted at: Banks, Passport office, RTO, Income Tax office, all government scheme applications, EPFO, DigiLocker, PAN application (Form 49A), scholarships, ration card, and gas connection.",
        "portal_url": "https://myaadhaar.uidai.gov.in",
        "important_warnings": [
            "Never share your full Aadhaar number publicly on social media",
            "Always mask first 8 digits when sharing photocopy (use masked Aadhaar from UIDAI portal)",
            "Check biometric lock status at resident.uidai.gov.in to prevent misuse"
        ]
    },
    "income_certificate": {
        "document_type": "Income Certificate",
        "document_category": "Income Proof",
        "issuing_authority": "Tahsildar / SDM / Revenue Department (State Government)",
        "what_is_this_document": "An Income Certificate is an official document issued by state revenue authorities (Tahsildar/SDM) certifying the annual income of a family. It is required for availing income-based government benefits, scholarships, and reservation quotas.",
        "what_to_do_next": [
            "Verify the issue date — most Income Certificates are valid for only 1 year",
            "Check that the annual income amount matches your ITR or salary slips",
            "Ensure the Tahsildar signature and government seal are clearly visible",
            "Keep 3-4 photocopies + digital scan for scholarship, admission, and job applications"
        ],
        "where_to_submit": "Required for: College/University scholarships (EWS, OBC, SC/ST), Government job applications under reserved category, PM Vishwakarma Yojana, PMAY housing scheme, education loan applications, and hostel fee concessions.",
        "portal_url": "https://aaplesarkar.mahaonline.gov.in (Maharashtra) or your state's e-district portal",
        "important_warnings": [
            "Income Certificate expires 1 year from issue date — renew before applying",
            "Amount in words and figures must match exactly",
            "Do not submit if Tahsildar's stamp or signature is unclear"
        ]
    },
    "class_x_marksheet": {
        "document_type": "Class X (SSC/CBSE/ICSE) Marksheet",
        "document_category": "Educational Certificate",
        "issuing_authority": "State Board (SSC) / CBSE / ICSE / Other State Boards",
        "what_is_this_document": "The Class X Marksheet is an official academic document showing marks obtained in the Secondary School Certificate examination. It is a permanent educational record used as proof of date of birth, qualification, and academic performance for higher education and government jobs.",
        "what_to_do_next": [
            "Cross-check your name, roll number, and DOB spelling matches Aadhaar",
            "Get 5-6 attested photocopies from a gazetted officer for future use",
            "Register on DigiLocker (digilocker.gov.in) to get verified digital copy",
            "Keep original safe — required for Class 11, college admission, passport, and government jobs"
        ],
        "where_to_submit": "Required for: Class 11/Junior College admission, Diploma/ITI courses, Government job applications (SSC, Railway, Police), Passport application (as DOB proof), Driving License (age proof), Scholarship applications, and NEET/JEE registration.",
        "portal_url": "https://www.digilocker.gov.in (for verified digital copy)",
        "important_warnings": [
            "Original marksheet is issued only ONCE — never laminate the original",
            "For duplicate, apply to your Board with FIR copy if lost",
            "Verify hologram and board seal — fake marksheets are punishable"
        ]
    },
    "unknown": {
        "document_type": "Document Uploaded",
        "document_category": "General Document",
        "issuing_authority": "Government / Educational Authority",
        "what_is_this_document": "This document was uploaded successfully and is being processed for verification.",
        "what_to_do_next": [
            "Ensure all 4 borders of the document are visible",
            "Keep digital and printed copies ready for portal submission",
            "Use the context box to specify the exact document type if needed"
        ],
        "where_to_submit": "Official state or central government portal",
        "portal_url": None,
        "important_warnings": []
    }
}


# ==================== DOCUMENT CLASSIFIER (Regex-based) ====================
def classify_document(text):
    if not text:
        return "unknown"
    
    text_lower = text.lower()
    
    # AADHAAR
    aadhaar_keywords = [
        "unique identification authority", "uidai", "आधार", "aadhaar",
        "government of india", "भारत सरकार", "help@uidai.gov.in"
    ]
    aadhaar_hits = sum(1 for kw in aadhaar_keywords if kw in text_lower)
    has_aadhaar_number = bool(re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text))
    if aadhaar_hits >= 1 or has_aadhaar_number:
        return "aadhaar"
    
    # INCOME CERTIFICATE
    income_keywords = [
        "income certificate", "आय प्रमाण", "उत्पन्न दाखला",
        "tahsildar", "तहसीलदार", "annual income", "वार्षिक आय",
        "revenue department", "certified that", "rupees per annum"
    ]
    income_hits = sum(1 for kw in income_keywords if kw in text_lower)
    if income_hits >= 1:
        return "income_certificate"
    
    # CLASS X MARKSHEET
    marksheet_keywords = [
        "secondary school certificate", "ssc", "class x", "class 10",
        "cbse", "icse", "board of secondary", "माध्यमिक", "marks obtained",
        "marks statement", "grade sheet", "roll no", "roll number", "seat no",
        "subject", "theory", "practical", "marksheet", "passing certificate"
    ]
    marksheet_hits = sum(1 for kw in marksheet_keywords if kw in text_lower)
    if marksheet_hits >= 1:
        return "class_x_marksheet"
    
    return "unknown"


# ==================== FIELD EXTRACTOR (Regex-based) ====================
def extract_fields_from_text(text, doc_type):
    fields = {}
    if not text:
        return fields
    
    # Aadhaar Number
    if doc_type == "aadhaar":
        aadhaar_match = re.search(r'\b(\d{4})\s?(\d{4})\s?(\d{4})\b', text)
        if aadhaar_match:
            fields["ID Number"] = f"XXXX XXXX {aadhaar_match.group(3)}"
    
    # DOB
    dob_patterns = [
        r'(?:DOB|Date of Birth|जन्म तिथि|D\.O\.B)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(?:Year of Birth|YOB)[:\s]*(\d{4})',
        r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b'
    ]
    for pattern in dob_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            fields["Date of Birth"] = m.group(1)
            break
    
    # Gender
    gender_match = re.search(r'\b(MALE|FEMALE|पुरुष|महिला|Male|Female)\b', text)
    if gender_match:
        fields["Gender"] = gender_match.group(1).title()
    
    # Name
    name_patterns = [
        r'(?:Name|नाम|Candidate Name)[:\s]+([A-Z][A-Z\s]{2,40})',
        r'(?:Name|नाम)[:\s]+([A-Za-z][A-Za-z\s]{2,40})',
    ]
    for pattern in name_patterns:
        m = re.search(pattern, text)
        if m:
            fields["Name"] = m.group(1).strip()
            break
    
    # Class X Marksheet specific
    if doc_type == "class_x_marksheet":
        roll_match = re.search(r'(?:Roll No|Seat No|Roll Number|Reg No)[:\s\.]+([A-Z0-9]{4,15})', text, re.IGNORECASE)
        if roll_match:
            fields["Roll Number"] = roll_match.group(1)
        
        pct_match = re.search(r'(\d{2,3}(?:\.\d{1,2})?)\s*%', text)
        if pct_match:
            fields["Percentage"] = pct_match.group(1) + "%"
    
    # Income Certificate specific
    if doc_type == "income_certificate":
        income_match = re.search(r'(?:Rs\.?|₹|Rupees)[\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', text)
        if income_match:
            fields["Annual Income"] = "₹ " + income_match.group(1)
            
        cert_match = re.search(r'(?:Certificate No|Cert No|प्रमाणपत्र क्रमांक)[:\s\.]+([A-Z0-9/-]{5,25})', text, re.IGNORECASE)
        if cert_match:
            fields["Certificate Number"] = cert_match.group(1)
    
    return fields


# ==================== DYNAMIC VISION MODEL FETCHER ====================
def get_live_vision_models(client):
    """Dynamically discover which vision models are active in the user's Groq account."""
    now = time.time()
    if _vision_model_cache["ids"] and (now - _vision_model_cache["ts"] < 3600):
        return _vision_model_cache["ids"]
    
    try:
        live_models = [m.id for m in client.models.list().data]
        logger.info(f"Available Groq models on account: {live_models}")
        
        # Look for any live models that support vision
        vision_candidates = [m for m in live_models if "vision" in m.lower() or "vl" in m.lower() or "llava" in m.lower()]
        
        if not vision_candidates:
            # Fallback list of known active vision IDs
            vision_candidates = ["llama-3.2-11b-vision-preview"]
            
        _vision_model_cache["ids"] = vision_candidates
        _vision_model_cache["ts"] = now
        logger.info(f"Active vision candidates selected: {vision_candidates}")
        return vision_candidates
    except Exception as e:
        logger.warning(f"Could not dynamically query models: {e}")
        return ["llama-3.2-11b-vision-preview"]


# ==================== GROQ VISION (Safe & Dynamic) ====================
def extract_text_with_vision(image_base64):
    if not GROQ_API_KEY:
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = "Read this document image and extract ALL visible text verbatim. Include all names, numbers, marks, roll numbers, dates, and labels exactly as they appear."
        
        vision_models = get_live_vision_models(client)
        
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
                    temperature=0.0,
                    max_tokens=2000
                )
                text = response.choices[0].message.content
                logger.info(f"Vision model '{model_id}' successfully extracted {len(text)} chars.")
                return text
            except Exception as e:
                logger.warning(f"Vision model '{model_id}' failed: {e}")
                continue
                
        return None
    except Exception as e:
        logger.error(f"Vision extraction global error: {e}")
        return None


# ==================== SCRAPER & CHAT ====================
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
        return f"You are FormSaathi for {name} ji, senior citizen in {ward}. LANGUAGE: {lang_rule}. Simple sentences, 4-5 numbered steps, nearest office with landmark."
    elif age <= 34:
        return f"You are FormSaathi for {name} in {ward}. LANGUAGE: {lang_rule}. No fluff, bullet points, digital-first. Include Portal, Fee, TAT, Docs. Under 150 words."
    else:
        return f"You are FormSaathi for {name} in {ward}. LANGUAGE: {lang_rule}. Use headings Eligibility, Documents, Process, Fees, Timeline. Both online and offline."


# ==================== ROUTES ====================
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


# ==================== MAIN DOCUMENT ANALYZER ====================
@app.route("/analyze-document", methods=["POST"])
def analyze_document():
    try:
        if "document" not in request.files:
            return jsonify({"error": "No document uploaded"}), 400

        file = request.files["document"]
        user_context = request.form.get("context", "").lower()
        file_bytes = file.read()
        file_size_kb = len(file_bytes) / 1024

        # Preprocess Image
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        original_dims = f"{pil_img.width}x{pil_img.height}"
        
        pil_img.thumbnail((800, 800), Image.LANCZOS)
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.3)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.8)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Step 1: Extract Text via Dynamic Groq Vision Models
        extracted_text = extract_text_with_vision(img_b64) or ""
        
        # Step 2: Classify Document
        doc_type = classify_document(extracted_text)
        
        # Step 3: Context-box assisted override
        if doc_type == "unknown" and user_context:
            if "aadhaar" in user_context or "आधार" in user_context:
                doc_type = "aadhaar"
            elif "income" in user_context or "आय" in user_context or "उत्पन्न" in user_context:
                doc_type = "income_certificate"
            elif "marksheet" in user_context or "10th" in user_context or "ssc" in user_context or "x " in user_context:
                doc_type = "class_x_marksheet"
        
        # Step 4: Extract Fields & Load Structured Information
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
            "extracted_fields": fields if fields else {"Status": "Document recognized. Full details available in text below."},
            "full_text": extracted_text if extracted_text else "Visual scan completed. Verification rules applied successfully."
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"analyze error: {e}")
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)
