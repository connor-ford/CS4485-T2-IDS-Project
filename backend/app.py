import os
import joblib
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from models import MODEL_REGISTRY

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('IDSML_DATABASE_URI', 'sqlite:///idsml.db')
db = SQLAlchemy(app)

# where artifacts live; override with IDSML_ARTIFACTS_DIR
ART_DIR = os.getenv(
    "IDSML_ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "artifacts")
)

#define models
class PredictionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50), nullable=False)
    input_data = db.Column(db.JSON, nullable=False)
    prediction_result = db.Column(db.JSON, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

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
        
        #log the prediction
        log_entry = PredictionLog(
            model_name=model_name,
            input_data=inputs,
            prediction_result=pred
        )
        try:
            db.session.add(log_entry)
            db.session.commit()
        
        except Exception as e:
            db.session.rollback()
            print(f"Failed to log prediction: {e}")    #log this to a file or alert the dev

        return jsonify({"model": model_name, "prediction": pred, "meta": meta or {}})
    except Exception as e:
        return jsonify({"error": f"inference failed: {type(e).__name__}: {e}"}), 400


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

@app.route("/logs", methods=["GET"])
def get_logs():
    logs = PredictionLog.query.all()
    return jsonify([{
        "id": log.id,
        "model_name": log.model_name,
        "input_data": log.input_data,
        "prediction_result": log.prediction_result,
        "timestamp": log.timestamp
    } for log in logs])

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    # dev only; container uses gunicorn
    app.run(host="0.0.0.0", port=8000, debug=True)
