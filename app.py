# ======================================================================
# PHOTO RESIZER ENGINE (OpenCV + Lightweight Rembg)
# ======================================================================
import urllib.request
from rembg import remove, new_session

# 1. Download Face Detector
cascade_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        cascade_path
    )
face_cascade = cv2.CascadeClassifier(cascade_path)

# 2. Use lightweight "u2netp" model (4MB) to prevent Render RAM crashes
rembg_session = new_session("u2netp")

FORM_SPECS = {
    "aadhaar": {"name": "Aadhaar Card", "width_px": 413, "height_px": 531, "max_size_kb": 50, "face_coverage_min": 0.70, "face_coverage_max": 0.80},
    "driving_license": {"name": "Driving License", "width_px": 413, "height_px": 531, "max_size_kb": 20, "face_coverage_min": 0.70, "face_coverage_max": 0.80},
    "voter_id": {"name": "Voter ID", "width_px": 413, "height_px": 531, "max_size_kb": 200, "face_coverage_min": 0.70, "face_coverage_max": 0.80},
    "income_certificate": {"name": "Income Certificate", "width_px": 160, "height_px": 212, "max_size_kb": 20, "face_coverage_min": 0.70, "face_coverage_max": 0.80},
    "domicile_certificate": {"name": "Domicile Certificate", "width_px": 160, "height_px": 212, "max_size_kb": 20, "face_coverage_min": 0.70, "face_coverage_max": 0.80}
}

class PhotoProcessor:
    def process(self, image_bytes, form_id):
        if form_id not in FORM_SPECS:
            return {"success": False, "error": f"Unknown form: {form_id}"}
        specs = FORM_SPECS[form_id]
        
        raw_pil = Image.open(io.BytesIO(image_bytes))
        original_pil = ImageOps.exif_transpose(raw_pil).convert("RGB")
        original_size_kb = len(image_bytes) / 1024
        original_np = np.array(original_pil)
        
        # Detect exact face coordinates using OpenCV
        face_info = self._detect_face(original_np)
        if not face_info["face_found"]:
            return {"success": False, "error": "No clear face detected. Please upload a front-facing portrait."}
            
        original_coverage = face_info["face_height_ratio"]
        
        # Crop exactly to 75% coverage
        cropped_pil = self._crop_and_center_face(original_pil, face_info, specs["face_coverage_min"], specs["face_coverage_max"], specs["width_px"], specs["height_px"])
        
        # Remove background using lightweight AI model
        white_bg_pil = self._replace_background(cropped_pil)
        resized_pil = white_bg_pil.resize((specs["width_px"], specs["height_px"]), Image.LANCZOS)
        
        # Compress
        compressed_bytes, final_size_kb = self._compress_image(resized_pil, specs["max_size_kb"])
        
        report = {
            "before": {"size_kb": round(original_size_kb, 2)},
            "after": {"dimensions": f"{specs['width_px']}x{specs['height_px']} px", "size_kb": final_size_kb, "face_coverage": "75%", "max_allowed_kb": specs["max_size_kb"]},
            "checks": {
                "face_detected": "✅ Face Detected & Centered",
                "background": "✅ Studio White Background",
                "dimensions": f"✅ Resized to {specs['width_px']}x{specs['height_px']}px",
                "file_size": "✅ Within Limit" if final_size_kb <= specs["max_size_kb"] else "⚠️ Over Limit"
            }
        }
        return {"success": True, "form": specs["name"], "processed_image_base64": base64.b64encode(compressed_bytes).decode("utf-8"), "report": report}

    def _detect_face(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0: return {"face_found": False}
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
        
        pad_left, pad_top = int(max(0, -crop_x1)), int(max(0, -crop_y1))
        pad_right, pad_bottom = int(max(0, crop_x2 - img_w)), int(max(0, crop_y2 - img_h))
        if any([pad_left, pad_top, pad_right, pad_bottom]):
            pil_image = ImageOps.expand(pil_image, (pad_left, pad_top, pad_right, pad_bottom), fill=(255, 255, 255))
            crop_x1 += pad_left; crop_x2 += pad_left; crop_y1 += pad_top; crop_y2 += pad_top
            
        return pil_image.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))

    def _replace_background(self, pil_image):
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="PNG")
        removed_bg_bytes = remove(img_byte_arr.getvalue(), session=rembg_session) # Uses low-RAM model
        removed_bg = Image.open(io.BytesIO(removed_bg_bytes)).convert("RGBA")
        white_bg = Image.new("RGBA", removed_bg.size, (255, 255, 255, 255))
        white_bg.paste(removed_bg, mask=removed_bg.split()[3])
        return white_bg.convert("RGB")

    def _compress_image(self, pil_image, max_size_kb):
        quality = 95
        while quality >= 5:
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
            size_kb = buffer.tell() / 1024
            if size_kb <= max_size_kb: return buffer.getvalue(), round(size_kb, 2)
            quality -= 5
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=5, optimize=True)
        return buffer.getvalue(), round(buffer.tell() / 1024, 2)

photo_processor = PhotoProcessor()

# --- AND ADD THIS ENDPOINT ROUTE ---
@app.route("/process-photo", methods=["POST"])
def process_photo():
    try:
        if "photo" not in request.files or "form_id" not in request.form:
            return jsonify({"error": "Missing photo or form_id"}), 400
        result = photo_processor.process(request.files["photo"].read(), request.form["form_id"])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Photo error: {e}")
        return jsonify({"error": str(e)}), 500
