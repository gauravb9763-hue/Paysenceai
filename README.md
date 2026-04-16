# PaySense AI Engine — Payment Routing Optimizer

> **Intelligent payment routing optimization system powered by ML (Hybrid Scoring) + GenAI-ready hooks**

This repository is a **recruiter-friendly, runnable reference implementation** of the PaySense-style routing engine concept.
It shows the end-to-end *flow* (features → scoring → rule filters → API response) in a way that’s easy to read.

### TL;DR (What happens on every transaction)
1. Take transaction metadata (amount, payment type, issuer…)
2. Pull the **latest gateway health snapshot** (recent success trend + latency)
3. Score each gateway using **Hybrid ML** (Random Forest + Logistic Regression)
4. Apply **hard safety rules** (disabled gateways, downtime threshold, payment-type priority)
5. Return `{ selected_gateway, confidence }` within the latency budget

### What’s included in this repo
- A runnable Flask API: [api/app.py](api/app.py)
- Feature engineering pipeline: [data_pipeline/etl_pipeline.py](data_pipeline/etl_pipeline.py)
- Hybrid routing engine + rule filters: [models/routing_engine.py](models/routing_engine.py)
- Config-driven routing constraints: [config/routing_rules.json](config/routing_rules.json)

---

## Section 1: Context (Brief)

### One-Paragraph Description
**PaySense AI Engine** is an AI-powered payment routing platform designed to maximize transaction success rates by routing each transaction to the **best-performing gateway/terminal at that moment**. The system uses a **hybrid ML approach (Random Forest + Logistic Regression)** to predict success probability using recent gateway health signals (recency success + latency), then applies **static safety rules** (downtime threshold, payment-type eligibility, priority constraints) before returning the final routing decision. It is built to be deployed behind a **Flask REST API** and extended with a **GenAI layer** (placeholder hook in this repo) for failure log summarization and outage investigation.

### Primary Technical Constraints
- **Latency requirement**: routing decision in **<200ms** on the online path
- **High volume**: large concurrent throughput (routing is on critical payment path)
- **Data freshness**: gateway health must reflect **recent behavior**, not only historical averages
- **Multi-gateway complexity**: performance varies by **payment type**, **issuer**, **region**, and **time-of-day**

---

## Section 2: Technical Implementation (Detailed)

### Architecture Diagram

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                    PAYSENSE AI ENGINE — HIGH LEVEL FLOW                       │
└──────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
   │   Razorpay   │      │    Stripe    │      │    Paytm     │
   │   Gateway    │      │   Gateway    │      │   Gateway    │
   └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
          │                     │                     │
          └───────────────┬─────┴─────────────┬──────┘
                          ▼                   ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│  DATA INGESTION (events/logs/metrics)                                         │
│  - success/failure events                                                     │
│  - latency metrics                                                           │
│  - gateway health flags / downtime                                            │
└──────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ETL + FEATURE ENGINEERING (Python)                                           │
│  - rolling success (recency health)                                           │
│  - exponential weighted success (smoother recency)                            │
│  - latency score (penalize slow terminals)                                    │
│  Implemented in: data_pipeline/etl_pipeline.py                                │
└──────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  FEATURE STORE / DWH (production: Snowflake style)                            │
│  - latest per-gateway snapshot used by online inference                       │
└──────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  HYBRID ROUTING ENGINE (ML + rules)                                           │
│  - RandomForest: non-linear patterns                                          │
│  - LogisticRegression: probability calibration                                │
│  - rule filters: payment type eligibility, downtime threshold, disable list    │
│  Implemented in: models/routing_engine.py                                     │
└──────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  FLASK REST API                                                                │
│  POST /route_transaction  → { selected_gateway, confidence, considered_gateways }
│  Implemented in: api/app.py                                                    │
└──────────────────────────────────────────────────────────────────────────────┘
                          │
                          ├──────────────► Monitoring (production: Power BI)
                          └──────────────► Alerts/Drift checks (production)
```

### Architecture Explanation 
The system follows a layered design where **feature engineering** is separated from **online scoring**, enabling safe iteration on models without breaking API contracts. Online requests use the latest gateway snapshot, score candidates via hybrid ML, then apply rule-based safety constraints to return a low-latency decision.

---

### Project File Structure

```text
Paysenceai/
  api/
    app.py
  models/
    routing_engine.py
  data_pipeline/
    etl_pipeline.py
  config/
    routing_rules.json
  requirements.txt
  README.md
```

### File Descriptions

| File | Purpose | Key pieces |
|------|---------|------------|
| [api/app.py](api/app.py) | Flask REST API | `/route_transaction`, `/health`, bootstrapped demo training |
| [models/routing_engine.py](models/routing_engine.py) | Core routing logic | `PayRouteAI.train_model()`, `predict_best_gateway()`, rule filters, GenAI hook placeholder |
| [data_pipeline/etl_pipeline.py](data_pipeline/etl_pipeline.py) | Feature engineering | `transform_features()`, `latest_gateway_snapshot()` |
| [config/routing_rules.json](config/routing_rules.json) | Static constraints | downtime threshold, payment-type priority list, disabled gateways |
| [requirements.txt](requirements.txt) | Dependencies | Flask, Pandas, NumPy, scikit-learn |

---

### Code Walk-through (Critical Function)

**Critical function:** `transform_features()` in [data_pipeline/etl_pipeline.py](data_pipeline/etl_pipeline.py)

```python
def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
```

**Why this function matters :**
- Payment routing fails mostly when we use stale health stats; this function makes health **recency-aware**.
- It converts raw events into **stable, model-friendly signals** (success trend + latency penalty).
- It keeps online scoring fast because inference reads a small, clean feature set.

---

### Data Flow (Key Operation: `POST /route_transaction`)

```text
(1) Client → API
    POST /route_transaction
    { amount, payment_type, issuer_bank, ... }

(2) API builds candidate feature snapshot
    - In production: fetch from Snowflake/feature store
    - In this repo: uses a small seeded dataset for a runnable demo

(3) Hybrid scoring
    - RandomForest success probability
    - LogisticRegression success probability
    - hybrid_score = 0.7 * RF + 0.3 * LR

(4) Rule filters (safety)
    - payment-type allowed gateways
    - downtime threshold
    - disabled gateways list

(5) Response
    { selected_gateway, confidence, considered_gateways }

(6) Feedback loop (production concept)
    success/failure + latency events update gateway health features
```

---

## Section 3: Technical Decisions (Core)

### Decision 1: Hybrid ML (Random Forest + Logistic Regression)
**Chosen because:**
- Random Forest captures non-linear behavior (gateway interactions, latency cliffs)
- Logistic Regression provides more stable probability calibration

**Alternatives considered:** XGBoost, single-model RF, deep learning ranking

**Trade-off:** Slightly more moving parts, but improved stability and debuggability.

---

### Decision 2: Rules + ML (not ML-only)
**Chosen because:** payments need hard constraints:
- “this gateway is disabled”
- “this payment type is not allowed”
- “downtime above threshold”

**Trade-off:** Sometimes a high-scoring gateway is blocked by rules, but system safety is higher.

---

### Scaling Bottleneck & Mitigation
**Bottleneck:** online feature computation under high TPS (recency metrics become heavy if computed per request).

**Mitigation strategy (production-ready pattern):**
- Precompute gateway health every N seconds in a background job
- Cache the latest per-gateway snapshot (Redis-like)
- Online path only does: *fetch cached features → score → rule filter*

---

## Section 4: Learning & Iteration (Concise)

### Technical Mistake & Learning
**Mistake:** deployed early version without strong drift/behavior monitoring; routing quality dropped when gateway latency patterns shifted.

**Learning:** production ML needs continuous monitoring (feature distribution, confidence deltas) and retraining triggers based on thresholds.

### What I Would Do Differently Today
Replace small rolling windows with **exponential decay** features for smoother recency weighting during bursty failures.

---

## Tech Stack Summary

| Category | Tech |
|---|---|
| Language | Python, SQL (conceptually for ETL / DWH queries) |
| ML/AI | scikit-learn (RandomForest, LogisticRegression) |
| Data | Pandas, NumPy |
| API | Flask REST |
| Config | JSON rules |
| Cloud (production reference) | Azure ML + DWH (Snowflake-style) |
| Monitoring (production reference) | Power BI + drift/alerting |
| GenAI (production reference) | LangChain-style summarization (hook placeholder in this repo) |

## Key Outcomes (Project results / targets)

| Metric | Value |
|---|---:|
| Payment success rate lift | **+4% to +6%** (project outcome) |
| Online routing latency | **<200ms** (SLA target) |
| High-volume readiness | designed for millions/day (architecture intent) |
| Routing output | gateway + confidence + considered set |

---

## Quick Start

### Requirements
- Python 3.10+

### Run

```bash
py -m pip install -r requirements.txt
python api/app.py
```

### Test

```bash
curl -X POST http://localhost:5000/route_transaction \
  -H "Content-Type: application/json" \
  -d '{"amount": 5000, "payment_type": "UPI", "issuer_bank": "HDFC"}'
```

Example response:

```json
{
  "selected_gateway": "Razorpay",
  "status": "optimized",
  "confidence": 0.986,
  "considered_gateways": ["Paytm", "Razorpay", "Stripe"]
}
```

---

## GenAI Note (Honest + Extendable)
This repo includes a **placeholder hook** `get_log_summary_ai()` in [models/routing_engine.py](models/routing_engine.py). In a production system, this would be implemented with a secure LLM setup (e.g., LangChain + managed model endpoint) to summarize failure logs and help engineers debug gateway incidents faster.

---

## Author
Gaurav Bhurale — AI/ML Engineer (PaySense AI Engine / Real-time Payment Routing)
