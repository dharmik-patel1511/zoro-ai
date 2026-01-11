from model_service_full_updated import FinanceMLService, init_db, fetch_latest_run, fetch_run_by_id

import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from model_service_full_updated import FinanceMLService, init_db, fetch_latest_run, fetch_run_by_id

app = Flask(__name__)
CORS(app)

# Where this file lives (serve dashboard from same folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

init_db()

service = FinanceMLService()
# Train once at boot so dashboard can show something
try:
    metrics = service.train()
except Exception as e:
    metrics = {"error": str(e)}

@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "Finance ML API running",
        "model_metrics": metrics
    })

@app.get("/dashboard")
def dashboard():
    # Serve the updated dashboard html
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
    try:
        m = service.train()
        return jsonify({"status": "trained", "model_metrics": m})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.post("/api/predict")
def api_predict():
    payload = request.get_json(force=True) or {}

    # Required inputs
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
            override_loan_repayment=float(payload["loan_repayment"]) if "loan_repayment" in payload else None
        )
        return jsonify({"run_id": run_id, "output": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # IMPORTANT: set DATASET_PATH before running if you want real training
    # Windows example:
    #   setx DATASET_PATH "C:\Users\YASH SHARMA\Downloads\data (1).csv"
    # then reopen terminal and run: python app.py
    app.run(host="127.0.0.1", port=5000, debug=True)











@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    # The FinanceModel.predict method returns a dictionary
    result = finance_model.predict(
        income=data['income'],
        rent=data['rent'],
        groceries=data['groceries'],
        transport=data['transport'],
        entertainment=data['entertainment'],
        investment=data['investment'],
        timely_loan_repayment=data['timely_loan_repayment'],
        goal_achievement=data['goal_achievement']
    )
    return jsonify(result)