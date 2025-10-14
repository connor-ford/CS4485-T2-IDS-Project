import os
import joblib
from flask import Flask, request, jsonify
from models import MODEL_REGISTRY
from utils.database_json import save_result_to_json

app = Flask(__name__)

# where artifacts live; override with IDSML_ARTIFACTS_DIR
ART_DIR = os.getenv(
    "IDSML_ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "artifacts")
)


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

        save_result_to_json({
            "model": model_name,
            "inputs": inputs,
            "prediction": pred,
            "meta": meta or {},
            "source": "frontend_form"
        })


        return jsonify({"model": model_name, "prediction": pred, "meta": meta or {}})
    except Exception as e:
        return jsonify({"error": f"inference failed: {type(e).__name__}: {e}"}), 400

@app.route("/history", methods=["GET"])
def history():
    with open(os.path.join("data", "results.json"), "r") as f:
        data = json.load(f)
    return jsonify(data)


@app.get("/schema")
def schema():
    """Return the active feature schema and class labels."""
    features_path = os.path.join(ART_DIR, "features.pkl")
    classes_path = os.path.join(ART_DIR, "classes.pkl")

    # fail fast with clear errors
    missing = [p for p in [features_path, classes_path] if not os.path.exists(p)]
    if missing:
        return (
            jsonify(
                {
                    "error": "schema artifacts not found",
                    "missing": missing,
                    "artifacts_dir": ART_DIR,
                }
            ),
            500,
        )

    features = joblib.load(features_path)
    classes = joblib.load(classes_path)
    return jsonify(
        {
            "artifacts_dir": ART_DIR,
            "features": features,  # ordered list
            "classes": classes,  # label order matches predict_proba indices
            "models": sorted(MODEL_REGISTRY.keys()),
        }
    )


@app.get("/schema/<what>")
def schema_part(what: str):
    if what not in ("features", "classes"):
        return (
            jsonify({"error": "use /schema, /schema/features, or /schema/classes"}),
            400,
        )
    path = os.path.join(ART_DIR, f"{what}.pkl")
    if not os.path.exists(path):
        return (
            jsonify({"error": f"{what}.pkl not found", "artifacts_dir": ART_DIR}),
            500,
        )
    return jsonify({what: joblib.load(path)})


if __name__ == "__main__":
    # dev only; container uses gunicorn
    app.run(host="0.0.0.0", port=8000, debug=True)
