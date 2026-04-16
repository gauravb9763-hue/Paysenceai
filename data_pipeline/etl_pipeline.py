from __future__ import annotations

import pandas as pd
import numpy as np


class PaymentDataPipeline:
    """ETL and feature engineering for gateway routing decisions."""

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build recency-aware routing features from raw transaction events.

        Expected columns:
        - timestamp
        - gateway
        - success (0/1)
        - latency_ms
        """
        required = {"timestamp", "gateway", "success", "latency_ms"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp", "gateway", "success", "latency_ms"])
        out = out.sort_values("timestamp")

        # Rolling success approximates short-term gateway health.
        out["rolling_success"] = out.groupby("gateway")["success"].transform(
            lambda s: s.rolling(window=3, min_periods=1).mean()
        )

        # Optional smoother recency metric for future model versions.
        out["ewm_success"] = out.groupby("gateway")["success"].transform(
            lambda s: s.ewm(span=5, adjust=False).mean()
        )

        # Penalize slow response paths to preserve API latency SLA.
        out["latency_score"] = np.where(out["latency_ms"] > 300, 0.0, 1.0)

        return out

    def latest_gateway_snapshot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return one latest feature row per gateway for online routing."""
        transformed = self.transform_features(df)
        idx = transformed.groupby("gateway")["timestamp"].idxmax()
        snapshot = transformed.loc[idx, ["gateway", "rolling_success", "ewm_success", "latency_score", "latency_ms"]]
        return snapshot.reset_index(drop=True)
