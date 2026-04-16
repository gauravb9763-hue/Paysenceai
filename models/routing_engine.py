from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class PayRouteAI:
    """Hybrid scoring engine: Random Forest + Logistic Regression + rule filters."""

    def __init__(self, rules_path: str = "config/routing_rules.json") -> None:
        self.rules = self._load_rules(rules_path)
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lr = LogisticRegression(max_iter=1000)
        self.is_trained = False

    @staticmethod
    def _load_rules(rules_path: str) -> Dict:
        with Path(rules_path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def train_model(self, train_df: pd.DataFrame) -> None:
        """Train both models on engineered features and binary target success."""
        features = ["rolling_success", "latency_score", "latency_ms"]
        target = "success"

        if train_df.empty:
            raise ValueError("Training data is empty")
        if any(c not in train_df.columns for c in features + [target]):
            raise ValueError("Training frame missing required columns")

        x = train_df[features]
        y = train_df[target].astype(int)

        self.rf.fit(x, y)
        self.lr.fit(x, y)
        self.is_trained = True

    def predict_best_gateway(
        self,
        candidate_features: pd.DataFrame,
        payment_type: str,
        gateway_downtime: Dict[str, float] | None = None,
    ) -> Dict[str, object]:
        """Rank gateways and return top valid option with confidence and explainability."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train_model() first.")
        if candidate_features.empty:
            raise ValueError("candidate_features is empty")

        required = {"gateway", "rolling_success", "latency_score", "latency_ms"}
        if required.difference(candidate_features.columns):
            raise ValueError("candidate_features missing required columns")

        score_df = candidate_features.copy()
        x = score_df[["rolling_success", "latency_score", "latency_ms"]]

        rf_prob = self.rf.predict_proba(x)[:, 1]
        lr_prob = self.lr.predict_proba(x)[:, 1]

        # Weighted blend keeps RF expressiveness and LR calibration.
        score_df["hybrid_score"] = 0.7 * rf_prob + 0.3 * lr_prob
        filtered = self._apply_rule_filters(score_df, payment_type, gateway_downtime or {})

        if filtered.empty:
            return {
                "selected_gateway": None,
                "status": "no_valid_gateway",
                "confidence": 0.0,
                "reason": "All candidates filtered by static rules",
            }

        best_row = filtered.sort_values("hybrid_score", ascending=False).iloc[0]
        return {
            "selected_gateway": best_row["gateway"],
            "status": "optimized",
            "confidence": round(float(best_row["hybrid_score"]), 4),
            "considered_gateways": filtered["gateway"].tolist(),
        }

    def _apply_rule_filters(
        self,
        score_df: pd.DataFrame,
        payment_type: str,
        gateway_downtime: Dict[str, float],
    ) -> pd.DataFrame:
        allowed = set(self.rules.get("payment_type_priority", {}).get(payment_type, []))
        disabled = set(self.rules.get("disabled_gateways", []))
        max_downtime = float(self.rules.get("downtime_threshold_pct", 15))

        out = score_df.copy()
        if allowed:
            out = out[out["gateway"].isin(allowed)]

        if disabled:
            out = out[~out["gateway"].isin(disabled)]

        if gateway_downtime:
            out = out[out["gateway"].map(lambda g: gateway_downtime.get(g, 0.0) <= max_downtime)]

        return out.reset_index(drop=True)

    def get_log_summary_ai(self, failure_logs: List[str]) -> str:
        """Placeholder for LangChain/LLM summarization in production deployment."""
        if not failure_logs:
            return "No failure logs available."
        return (
            "Summary: "
            + f"Observed {len(failure_logs)} recent failures. "
            + "Top action: inspect latency spikes and issuer-bank specific declines."
        )
