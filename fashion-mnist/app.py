import os
import time
import base64
import tempfile
import requests
from flask import Flask, render_template, request, jsonify

# Import local Predictor as fallback when Cog container is not running
try:
    from predict import Predictor
    local_predictor = Predictor()
    local_predictor.setup()
except Exception as e:
    print(f"[WARN] Local predictor initialization notice: {e}")
    local_predictor = None

app = Flask(__name__)

# Cog container endpoint (e.g., http://localhost:5000/predictions)
COG_URL = os.getenv("COG_URL", "http://localhost:5000/predictions")
COG_BASE_URL = COG_URL.rsplit("/predictions", 1)[0]


@app.route("/")
def index():
    """Render the Garment Classifier web page with side-by-side comparison."""
    return render_template("index.html", cog_url=COG_URL)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Check if the Cog inference container is reachable."""
    try:
        health_endpoint = f"{COG_BASE_URL}/health-check"
        response = requests.get(health_endpoint, timeout=2)
        if response.status_code == 200:
            data = response.json()
            return jsonify({
                "status": "online",
                "mode": "Cog Container",
                "cog_status": data.get("status", "READY"),
                "endpoint": COG_URL
            })
    except Exception:
        pass

    # If Cog container is not reachable, fall back to local PyTorch inference
    if local_predictor is not None:
        return jsonify({
            "status": "online",
            "mode": "Local PyTorch Engine",
            "cog_status": "READY",
            "endpoint": "Direct In-Process Model"
        })

    return jsonify({
        "status": "offline",
        "mode": "Disconnected",
        "message": f"Cannot connect to Cog container at {COG_URL}"
    })


@app.route("/classify", methods=["POST"])
def classify():
    """
    Classify uploaded garment image with both Baseline and HPO models.
    Supports both Cog Container HTTP API and Local PyTorch engine.
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    image_bytes = file.read()
    if len(image_bytes) == 0:
        return jsonify({"success": False, "error": "Uploaded image file is empty."}), 400

    start_time = time.time()
    cog_connected = False
    output_data = None

    # 1. Try Cog Container if reachable
    try:
        mime_type = file.mimetype or "image/jpeg"
        b64_encoded = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_encoded}"

        payload = {"input": {"image": data_uri}}
        cog_response = requests.post(
            COG_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if cog_response.status_code == 200:
            res_json = cog_response.json()
            if res_json.get("status") == "succeeded":
                output_data = res_json.get("output")
                cog_connected = True
    except Exception:
        cog_connected = False

    # 2. Fall back to Local Predictor if Cog container is not running
    if output_data is None:
        if local_predictor is None:
            return jsonify({
                "success": False,
                "error": "Inference engine unavailable. Ensure Cog container or local weights are loaded."
            }), 500

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(image_bytes)

        try:
            output_data = local_predictor.predict(image=tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    elapsed = time.time() - start_time

    # Format side-by-side results
    def format_model_result(model_key):
        model_info = output_data.get(model_key, {})
        probs = model_info.get("probabilities", {})
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        breakdown = [
            {
                "class": c,
                "percentage": f"{p * 100:.2f}%",
                "percent_num": round(p * 100, 2)
            }
            for c, p in sorted_probs[:5]
        ]
        top_cls = model_info.get("top_class", "Unknown")
        conf_num = model_info.get("confidence", 0.0) * 100
        return {
            "top_class": top_cls,
            "confidence_pct": f"{conf_num:.2f}%",
            "confidence_num": round(conf_num, 2),
            "breakdown": breakdown
        }

    response_payload = {
        "success": True,
        "engine": "Cog Container" if cog_connected else "Local PyTorch Engine",
        "predict_time": f"{elapsed:.2f}s",
        "baseline": format_model_result("baseline"),
        "optimized": format_model_result("optimized"),
    }

    return jsonify(response_payload)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"Starting Garment Classifier Web App on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
