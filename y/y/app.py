from flask import Flask, send_file

app = Flask(__name__)

@app.route("/")
def landing():
    return send_file("zoro.html")

@app.route("/login")
def login():
    return send_file("index.html")

@app.route("/dashboard")
def dashboard():
    return send_file("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)
import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from model_service import FinanceMLService, init_db, fetch_latest_run

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

init_db()
service = FinanceMLService()

@app.get("/")
def home():
    return jsonify({"status": "ok", "message": "Finance ML API running"})

@app.get("/dashboard")
def dashboard():
    # IMPORTANT: put your dashboard.html here (same folder)
    return send_from_directory(BASE_DIR, "dashboard.html")

@app.post("/api/train")
def api_train():
    try:
        metrics = service.train()
        return jsonify({"status": "trained", "model_metrics": metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.post("/api/predict")
def api_predict():
    payload = request.get_json(force=True) or {}

    required = ["income","rent","groceries","transport","entertainment","investment","timely_loan_repayment","goal_achievement"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        run_id, output = service.predict_and_store(payload)
        return jsonify({"run_id": run_id, "output": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.get("/api/latest")
def api_latest():
    latest = fetch_latest_run()
    if not latest:
        return jsonify({"error": "No saved runs yet. Call POST /api/predict first."}), 404
    return jsonify(latest)

if __name__ == "__main__":
    # Tip: If DATASET_PATH is not visible in PowerShell, run:
    #   $env:DATASET_PATH = "C:\Users\YASH SHARMA\Downloads\data (1).csv"
    # then run python app.py
    app.run(host="127.0.0.1", port=5000, debug=True)
