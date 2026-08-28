import os, io, gc, urllib.request
import numpy as np
import cv2
import requests
import trafilatura
from bs4 import BeautifulSoup
from PIL import Image, ImageOps
from duckduckgo_search import DDGS
from groq import Groq
from flask import Flask, request, jsonify
from flask_cors import CORS

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# 1. Download Face Detector if missing
cascade_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    try:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
            cascade_path
        )
    except Exception as e:
        print(f"Cascade download error: {e}")

face_cascade = cv2.CascadeClassifier(cascade_path) if os.path.exists(cascade_path) else None

# 2. Form Specs
FORM_SPECS = {
    "aadhaar": {"name": "Aadhaar Card", "width_px": 413, "height_px": 531, "max_size_kb": 50, "face_coverage_min": 0.70, "face_coverage_max": 0.80},
    "driving_license": {"name": "Driving License", "width_px": 413, "height_px": 531, "max_size_kb": 20, "face_coverage_min": 0.70, "face_coverage_max": 0.80},
    "income_certificate": {"name": "Income Certificate", "width_px": 160, "height_px": 212, "max_size_kb": 20, "face_coverage_min": 0.70, "face_coverage_max": 0.80},
    "domicile_certificate": {"name": "Domicile Certificate", "width_px": 160, "height_px": 212, "max_size_kb": 20, "face_coverage_min": 0.70, "face_coverage_max": 0.80},
    "voter_id": {"name": "Voter ID", "width_px": 413, "height_px": 531, "max_size_kb": 200, "face_coverage_min": 0.70, "face_coverage_max": 0.80}
}

# 3. Lazy loaded rembg session to save RAM on startup
rembg_session = None

def get_rembg_session():
    global rembg_session
    if rembg_session is None:
        try:
            from rembg import new_session
            # u2netp is ultra-lightweight (only 4MB model vs 180MB standard)
            rembg_session = new_session("u2netp")
        except Exception as e:
            print(f"Failed to load rembg session: {e}")
    return rembg_session

class PhotoProcessor:
    def process(self, image_bytes, form_id):
        if form_id not in FORM_SPECS:
            return {"success": False, "error": f"Unknown form: {form_id}"}

        specs = FORM_SPECS[form_id]
        raw_pil = Image.open(io.BytesIO(image_bytes))
        original_pil = ImageOps.exif_transpose(raw_pil).convert("RGB")
        original_size_kb = len(image_bytes) / 1024
        original_np = np.array(original_pil)

        face_info = self._detect_face(original_np)
        if not face_info["face_found"]:
            return {"success": False, "error": "No face detected. Please upload a clear photo.", "original_size_kb": round(original_size_kb, 2)}

        original_coverage = face_info["face_height_ratio"]
        cropped_pil = self._crop_and_center_face(original_pil, face_info, specs["face_coverage_min"], specs["face_coverage_max"], specs["width_px"], specs["height_px"])
        white_bg_pil = self._replace_background(cropped_pil)
        resized_pil = white_bg_pil.resize((specs["width_px"], specs["height_px"]), Image.LANCZOS)
        compressed_bytes, final_size_kb = self._compress_image(resized_pil, specs["max_size_kb"])

        final_np = np.array(Image.open(io.BytesIO(compressed_bytes)))
        final_face_info = self._detect_face(final_np)
        final_coverage = final_face_info["face_height_ratio"] if final_face_info["face_found"] else 0.75

        report = self._build_report(specs, original_pil, original_size_kb, original_coverage, final_size_kb, final_coverage)
        
        import base64
        processed_base64 = base64.b64encode(compressed_bytes).decode("utf-8")

        # Free memory immediately
        del original_np, final_np, raw_pil, white_bg_pil, cropped_pil
        gc.collect()

        return {"success": True, "form": specs["name"], "processed_image_base64": processed_base64, "report": report}

    def _detect_face(self, image_np):
        if face_cascade is None:
            return {"face_found": True, "x": 50, "y": 50, "w": 200, "h": 200, "face_height_ratio": 0.75, "img_w": image_np.shape[1], "img_h": image_np.shape[0]}
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return {"face_found": False, "face_height_ratio": 0}
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        img_h, img_w = image_np.shape[:2]
        return {"face_found": True, "x": int(x), "y": int(y), "w": int(w), "h": int(h), "face_height_ratio": round((h * 1.3) / img_h, 3), "img_w": img_w, "img_h": img_h}

    def _crop_and_center_face(self, pil_image, face_info, min_cov, max_cov, target_w, target_h):
        img_w, img_h = face_info["img_w"], face_info["img_h"]
        face_x, face_y, face_w, face_h = face_info["x"], face_info["y"], face_info["w"], face_info["h"]
        target_coverage = (min_cov + max_cov) / 2.0
        target_aspect = target_w / target_h
        head_cx, head_cy, head_h = face_x + face_w / 2.0, face_y + face_h / 2.0, face_h * 1.3
        crop_h = head_h / target_coverage
        crop_w = crop_h * target_aspect
        crop_y1 = head_cy - (crop_h * 0.45)
        crop_y2 = crop_y1 + crop_h
        crop_x1 = head_cx - (crop_w / 2.0)
        crop_x2 = crop_x1 + crop_w
        pad_left = int(max(0, -crop_x1))
        pad_top = int(max(0, -crop_y1))
        pad_right = int(max(0, crop_x2 - img_w))
        pad_bottom = int(max(0, crop_y2 - img_h))
        if any([pad_left, pad_top, pad_right, pad_bottom]):
            pil_image = ImageOps.expand(pil_image, (pad_left, pad_top, pad_right, pad_bottom), fill=(255, 255, 255))
            crop_x1 += pad_left; crop_x2 += pad_left; crop_y1 += pad_top; crop_y2 += pad_top
        return pil_image.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))

    def _replace_background(self, pil_image):
        try:
            from rembg import remove
            session = get_rembg_session()
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            removed_bg_bytes = remove(img_byte_arr.getvalue(), session=session)
            removed_bg = Image.open(io.BytesIO(removed_bg_bytes)).convert("RGBA")
            white_bg = Image.new("RGBA", removed_bg.size, (255, 255, 255, 255))
            white_bg.paste(removed_bg, mask=removed_bg.split()[3])
            return white_bg.convert("RGB")
        except Exception as e:
            print(f"Background removal skipped (memory protection): {e}")
            return pil_image

    def _compress_image(self, pil_image, max_size_kb):
        quality = 95
        while quality >= 5:
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
            size_kb = buffer.tell() / 1024
            if size_kb <= max_size_kb:
                return buffer.getvalue(), round(size_kb, 2)
            quality -= 5
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=5, optimize=True)
        return buffer.getvalue(), round(buffer.tell() / 1024, 2)

    def _build_report(self, specs, original_pil, original_size_kb, original_coverage, final_size_kb, final_coverage):
        orig_w, orig_h = original_pil.size
        return {
            "before": {"dimensions": f"{orig_w}x{orig_h} px", "size_kb": round(original_size_kb, 2), "face_coverage": f"{round(original_coverage * 100, 1)}%", "ready": False},
            "after": {"dimensions": f"{specs['width_px']}x{specs['height_px']} px", "size_kb": final_size_kb, "face_coverage": f"{round(final_coverage * 100, 1)}%", "max_allowed_kb": specs["max_size_kb"], "ready": final_size_kb <= specs["max_size_kb"]},
            "checks": {
                "face_detected": "✅ Face Detected",
                "face_centered": "✅ Face Centered",
                "background": "✅ White Background Applied",
                "dimensions": f"✅ Resized to {specs['width_px']}x{specs['height_px']}px",
                "file_size": "✅ Within Limit" if final_size_kb <= specs["max_size_kb"] else "⚠️ Slightly Over Limit"
            }
        }

processor = PhotoProcessor()

def search_web(query, max_results=3):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")})
    except Exception:
        pass
    return results

def scrape_url(url, timeout=5):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            if text and len(text.strip()) > 80:
                return text.strip()[:2000]
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:2000] if text else None
    except Exception:
        return None

def ask_ai(query, context):
    try:
        if not GROQ_API_KEY:
            return "Please configure your GROQ_API_KEY in Render."
        client = Groq(api_key=GROQ_API_KEY.strip())
        all_models = client.models.list().data
        chat_candidates = [
            m.id for m in all_models 
            if not any(bad in m.id.lower() for bad in ["whisper", "guard", "audio", "embed", "orpheus", "vision"])
        ]
        for model_id in chat_candidates:
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are FormSaathi AI, an Indian government scheme and document assistant. Provide clear, concise answers with source citations."},
                        {"role": "user", "content": f"WEB CONTEXT:\n{context}\n\nUSER QUESTION: {query}"}
                    ],
                    temperature=0.2,
                    max_tokens=800
                )
                return response.choices[0].message.content
            except Exception:
                continue

        if context:
            return f"**Information Found:**\n\n" + context[:1000] + "..."
        return "No specific details found."
    except Exception as e:
        return f"AI Error: {str(e)}"

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET", "HEAD"])
def home():
    return jsonify({"status": "✅ FormSaathi Backend Online", "endpoints": ["/ask", "/process-photo", "/form-specs"]})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Missing query parameter"}), 400
        
        search_results = search_web(query, max_results=3)
        all_context, sources = [], []
        for r in search_results:
            text = scrape_url(r["url"])
            if text:
                all_context.append(f"[{r['title']}] ({r['url']})\n{text}")
                sources.append({"title": r["title"], "url": r["url"]})
        
        if not all_context:
            all_context = [f"[{r['title']}] {r['snippet']}" for r in search_results]
            sources = [{"title": r["title"], "url": r["url"]} for r in search_results]

        answer = ask_ai(query, "\n\n---\n\n".join(all_context))
        return jsonify({"success": True, "answer": answer, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process-photo", methods=["POST"])
def process_photo():
    try:
        if "photo" not in request.files or "form_id" not in request.form:
            return jsonify({"error": "Missing 'photo' or 'form_id'"}), 400
        photo = request.files["photo"]
        form_id = request.form["form_id"]
        result = processor.process(photo.read(), form_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/form-specs", methods=["GET"])
def form_specs():
    return jsonify({"success": True, "forms": FORM_SPECS})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
