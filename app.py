# Ultra-Fast Document Context Endpoint (Uses ~150 tokens)
@app.route("/summarize-doc", methods=["POST", "OPTIONS"])
def summarize_doc():
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        data = request.get_json(silent=True) or {}
        extracted_text = (data.get("text") or "").strip()
        
        if not extracted_text:
            return jsonify({"error": "No text provided"}), 400

        # Safer prompt format that won't break with weird OCR characters
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

        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
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
            # Fallback if the AI just outputs plain text instead of JSON
            summary = {
                "document_type": "Official Document",
                "description": "Analyzed document based on extracted text.",
                "actionable_steps": raw_response[:200] # Provide the raw response as the next step
            }

        return jsonify({"success": True, "summary": summary})
        
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        return jsonify({"error": "Failed to analyze document text."}), 500
