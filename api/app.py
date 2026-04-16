from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd
from flask import Flask, jsonify, request

# Add project root to import path for local package-style imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.etl_pipeline import PaymentDataPipeline
from models.routing_engine import PayRouteAI


app = Flask(__name__)
pipeline = PaymentDataPipeline()
engine = PayRouteAI(rules_path=str(ROOT / "config" / "routing_rules.json"))


def _seed_training_data() -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    rows = [
        {"timestamp": now - timedelta(minutes=10), "gateway": "Stripe", "success": 1, "latency_ms": 120},
        {"timestamp": now - timedelta(minutes=9), "gateway": "Stripe", "success": 1, "latency_ms": 130},
        {"timestamp": now - timedelta(minutes=8), "gateway": "Stripe", "success": 0, "latency_ms": 310},
        {"timestamp": now - timedelta(minutes=7), "gateway": "Razorpay", "success": 1, "latency_ms": 150},
        {"timestamp": now - timedelta(minutes=6), "gateway": "Razorpay", "success": 0, "latency_ms": 320},
        {"timestamp": now - timedelta(minutes=5), "gateway": "Razorpay", "success": 1, "latency_ms": 160},
        {"timestamp": now - timedelta(minutes=4), "gateway": "Paytm", "success": 1, "latency_ms": 180},
        {"timestamp": now - timedelta(minutes=3), "gateway": "Paytm", "success": 0, "latency_ms": 350},
        {"timestamp": now - timedelta(minutes=2), "gateway": "Paytm", "success": 1, "latency_ms": 170},
    ]
    return pd.DataFrame(rows)


def _bootstrap() -> None:
    train = _seed_training_data()
    engineered = pipeline.transform_features(train)
    engine.train_model(engineered)


_bootstrap()


@app.get("/health")
def health() -> tuple:
    return jsonify({"status": "ok", "service": "payroute-optimizer"}), 200


@app.post("/route_transaction")
def route_transaction() -> tuple:
    payload = request.get_json(silent=True) or {}
    payment_type = str(payload.get("payment_type", "UPI")).upper()

    # In production this snapshot would come from Snowflake/feature store.
    latest = pipeline.latest_gateway_snapshot(_seed_training_data())
    downtime = {"Stripe": 5.0, "Razorpay": 8.0, "Paytm": 3.0}

    result = engine.predict_best_gateway(
        candidate_features=latest,
        payment_type=payment_type,
        gateway_downtime=downtime,
    )

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
