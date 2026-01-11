from __future__ import annotations

import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from model_service_full_updated import FinanceMLService, init_db, fetch_latest_run, fetch_run_by_id

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)

# Ensure DB exists
init_db()

# Service loads dataset path from DATASET_PATH env var by default
service = FinanceMLService()

@app.get("/")
def root():
    # Serve dashboard
    return send_from_directory(BASE_DIR, "dashboard_updated.html")

@app.get("/dashboard")
def dashboard():
    return send_from_directory(BASE_DIR, "dashboard_updated.html")

@app.get("/api/latest")
def api_latest():
    latest = fetch_latest_run()
    if not latest:
        return jsonify({"error": "No runs saved yet. Call POST /api/predict first."}), 404
    return jsonify(latest)

@app.get("/api/run/<int:run_id>")
def api_run(run_id: int):
    run = fetch_run_by_id(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    return jsonify(run)

@app.post("/api/train")
def api_train():
    """Train the ML models and expose training metrics to the dashboard."""
    try:
        metrics = service.train()
        return jsonify({"ok": True, "model_metrics": metrics})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.post("/api/predict")
def api_predict():
    payload = request.get_json(force=True) or {}
    required = ["income","rent","groceries","transport","entertainment","investment","timely_loan_repayment","goal_achievement"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        run_id, out = service.predict_and_store(
            income=float(payload["income"]),
            rent=float(payload["rent"]),
            groceries=float(payload["groceries"]),
            transport=float(payload["transport"]),
            entertainment=float(payload["entertainment"]),
            investment=float(payload["investment"]),
            timely_loan_repayment=float(payload["timely_loan_repayment"]),
            goal_achievement=float(payload["goal_achievement"]),
            override_loan_repayment=float(payload["loan_repayment"]) if "loan_repayment" in payload else None,
        )
        return jsonify({"run_id": run_id, "output": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Windows (PowerShell):
    #   $env:DATASET_PATH = "C:\\path\\to\\data.csv"
    # macOS/Linux:
    #   export DATASET_PATH=/path/to/data.csv
    app.run(host="127.0.0.1", port=5000, debug=True)
