from flask import Flask, request, jsonify
from models import MODEL_REGISTRY

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models": sorted(MODEL_REGISTRY.keys())})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    model_name = data.get("model")
    inputs = data.get("inputs")

    if not model_name or not isinstance(model_name, str):
        return jsonify({"error": "field 'model' (str) is required"}), 400
    if inputs is None or not isinstance(inputs, dict):
        return jsonify({"error": "field 'inputs' (object) is required"}), 400

    runner = MODEL_REGISTRY.get(model_name)
    if runner is None:
        return (
            jsonify(
                {
                    "error": f"unknown model '{model_name}'",
                    "available_models": sorted(MODEL_REGISTRY.keys()),
                }
            ),
            400,
        )

    try:
        pred, meta = runner.predict(inputs)
        return jsonify({"model": model_name, "prediction": pred, "meta": meta or {}})
    except Exception as e:
        return jsonify({"error": f"inference failed: {type(e).__name__}: {e}"}), 400


if __name__ == "__main__":
    # dev only; container uses gunicorn
    app.run(host="0.0.0.0", port=8000, debug=True)
