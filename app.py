import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except Exception:
    USE_XGB = False

load_dotenv()

app = Flask(__name__)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini = genai.GenerativeModel("gemini-2.5-flash")

AVG_ANNUAL_REVENUE = 2400
AVG_ORDER_VALUE    = 200
COST_PER_INTERVENTION = 15

# Segment recovery rates (inverted vs intuition — Critical is hardest to retain)
RECOVERY_RATES = {"Critical": 0.15, "High": 0.25, "Medium": 0.35}

FEATURE_COLS = [
    "Tenure", "SatisfactionScore", "Complain",
    "NumberOfDeviceRegistered", "DaySinceLastOrder",
    "CashbackAmount", "OrderAmountHikeFromlastYear",
    "HourSpendOnApp", "NumberOfAddress", "OrderCount",
    "CouponUsed", "CityTier", "WarehouseToHome",
]

FEATURE_LABELS = {
    "DaySinceLastOrder":           "Days since last order",
    "SatisfactionScore":           "Satisfaction score",
    "Complain":                    "Complaint filed",
    "OrderAmountHikeFromlastYear": "Order value drop YoY",
    "CashbackAmount":              "Cashback not redeemed",
    "Tenure":                      "Customer tenure",
    "NumberOfDeviceRegistered":    "Devices registered",
    "HourSpendOnApp":              "App engagement",
    "NumberOfAddress":             "Addresses saved",
    "OrderCount":                  "Order frequency",
    "CouponUsed":                  "Coupon usage",
    "CityTier":                    "City tier",
    "WarehouseToHome":             "Delivery distance",
}


# ── Dataset ────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n=5630):
    """
    Generate synthetic data with enough noise that the trained model
    produces a spread of churn probabilities — not a bimodal 0/1 spike.
    Target at-risk distribution: ~15% Critical, ~25% High, ~60% Medium.
    """
    np.random.seed(42)
    churn = np.random.binomial(1, 0.165, n)

    # Base signals — weak correlation with moderate noise so model produces spread
    tenure_base = np.where(churn,
        np.random.gamma(2.5, 3, n).clip(0, 30),
        np.random.gamma(5, 5, n).clip(0, 60))
    tenure_base += np.random.normal(0, 4, n)  # noise
    tenure_base = tenure_base.clip(0, 60).astype(int)

    sat_base = np.where(churn,
        np.random.choice([1,2,3,4,5], n, p=[0.18,0.27,0.28,0.18,0.09]),
        np.random.choice([1,2,3,4,5], n, p=[0.06,0.12,0.24,0.33,0.25]))

    complain_base = np.where(churn,
        np.random.binomial(1, 0.28, n),
        np.random.binomial(1, 0.10, n))

    days_base = np.where(churn,
        np.clip(np.random.normal(28, 12, n), 5, 90),
        np.clip(np.random.normal(9,  7,  n), 0, 45))
    days_base = days_base.round().astype(int)

    cashback_base = np.where(churn,
        np.random.uniform(60, 180, n),
        np.random.uniform(90, 280, n))
    cashback_base += np.random.normal(0, 25, n)
    cashback_base = cashback_base.clip(20, 350).round(2)

    order_hike = np.where(churn,
        np.random.normal(2, 12, n),
        np.random.normal(15, 10, n))
    order_hike = order_hike.clip(-20, 50).round(1)

    order_count = np.where(churn,
        np.clip(np.random.normal(4, 3, n), 1, 15),
        np.clip(np.random.normal(10, 5, n), 2, 25))
    order_count = order_count.round().astype(int)

    return pd.DataFrame({
        "CustomerID":                  range(10001, 10001 + n),
        "Churn":                       churn,
        "Tenure":                      tenure_base,
        "SatisfactionScore":           sat_base,
        "Complain":                    complain_base,
        "NumberOfDeviceRegistered":    np.random.randint(1, 7, n),
        "DaySinceLastOrder":           days_base,
        "CashbackAmount":              cashback_base,
        "OrderAmountHikeFromlastYear": order_hike,
        "HourSpendOnApp":              np.random.randint(0, 6, n),
        "NumberOfAddress":             np.random.randint(1, 8, n),
        "OrderCount":                  order_count,
        "CouponUsed":                  np.random.randint(0, 10, n),
        "CityTier":                    np.random.choice([1, 2, 3], n, p=[0.45, 0.30, 0.25]),
        "WarehouseToHome":             np.random.randint(5, 40, n),
    })


def load_dataset():
    for path in ["data/E Commerce.xlsx", "data/E_Commerce.xlsx", "data/ecommerce.xlsx"]:
        if os.path.exists(path):
            try:
                for sheet in ["E Comm", 0]:
                    try:
                        df = pd.read_excel(path, sheet_name=sheet)
                        if "Churn" in df.columns:
                            df = df.dropna(subset=["Churn"])
                            df["Churn"] = df["Churn"].astype(int)
                            if "CustomerID" not in df.columns:
                                df.insert(0, "CustomerID", range(10001, 10001 + len(df)))
                            print(f"  Loaded real dataset from {path} — {len(df):,} rows")
                            return df
                    except Exception:
                        continue
            except Exception:
                pass
    print("  Real dataset not found — using synthetic data")
    return generate_synthetic_data()


# ── Model ──────────────────────────────────────────────────────────────────────

def train_model(df):
    available = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available].fillna(df[available].median())
    y = df["Churn"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # max_depth=3 for XGB (depth=4 overfit synthetic data → bimodal 0/1 probabilities)
    if USE_XGB:
        model = XGBClassifier(n_estimators=100, max_depth=3, random_state=42,
                              eval_metric="logloss", learning_rate=0.08,
                              subsample=0.8, colsample_bytree=0.8)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42,
                                       n_jobs=-1, min_samples_leaf=8)

    model.fit(X_train, y_train)
    importances = dict(zip(available, model.feature_importances_))
    return model, importances, available


# ── Dashboard computation ──────────────────────────────────────────────────────

def compute_dashboard(df, model, importances, features):
    X = df[features].fillna(df[features].median())
    scores = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["churn_score"] = scores

    high_risk = df[df["churn_score"] >= 0.50].copy()
    total     = len(df)
    hr_count  = len(high_risk)

    # Composite urgency score: blends churn probability with observable signals
    # so the displayed number visibly explains the segment label.
    # Formula: churn×0.50 + inactivity×0.25 + complaint×0.15 + low_sat×0.10
    def _urgency(row):
        days_norm = min(float(row.get("DaySinceLastOrder", 0)) / 90.0, 1.0)
        complaint = float(row.get("Complain", 0))
        sat       = float(row.get("SatisfactionScore", 3))
        low_sat   = 1.0 - (sat - 1.0) / 4.0  # sat 1→1.0, sat 5→0.0
        return round(float(row["churn_score"]) * 0.50 + days_norm * 0.25
                     + complaint * 0.15 + low_sat * 0.10, 3)

    if hr_count > 0:
        high_risk["urgency_score"] = high_risk.apply(_urgency, axis=1)
        p85 = float(high_risk["urgency_score"].quantile(0.85))
        p60 = float(high_risk["urgency_score"].quantile(0.60))

        def _seg(u):
            if u >= p85: return "Critical"
            if u >= p60: return "High"
            return "Medium"

        high_risk["segment"] = high_risk["urgency_score"].apply(_seg)
    else:
        high_risk["urgency_score"] = 0.0
        def _seg(_): return "Medium"

    seg_data = {}
    for seg, lo, hi in [
        ("Critical", p85 if hr_count > 0 else 0.85, 2.0),
        ("High",     p60 if hr_count > 0 else 0.60, p85 if hr_count > 0 else 0.85),
        ("Medium",   0.50, p60 if hr_count > 0 else 0.60),
    ]:
        sdf   = high_risk[high_risk["segment"] == seg]
        count = len(sdf)
        seg_data[seg.lower()] = {
            "count":         count,
            "revenue":       count * AVG_ANNUAL_REVENUE,
            "avg_score":     round(float(sdf["urgency_score"].mean()), 2) if count else 0.0,
            "recovery_rate": RECOVERY_RATES[seg],
        }

    revenue_at_risk = hr_count * AVG_ANNUAL_REVENUE

    top_drivers = [
        {"feature": FEATURE_LABELS.get(f, f), "impact": round(float(v), 4)}
        for f, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    actions = {
        "Critical": "Priority call + 20% discount",
        "High":     "Email + cashback offer",
        "Medium":   "Re-engagement campaign",
    }

    at_risk_rows = []
    for _, row in high_risk.sort_values("urgency_score", ascending=False).iterrows():
        seg = row["segment"]
        at_risk_rows.append({
            "customer_id":        int(row.get("CustomerID", int(row.name) + 10001)),
            "churn_score":        round(float(row["churn_score"]), 3),
            "urgency_score":      round(float(row["urgency_score"]), 3),
            "revenue_at_risk":    AVG_ANNUAL_REVENUE,
            "days_inactive":      int(row.get("DaySinceLastOrder", 0)),
            "satisfaction_score": int(row.get("SatisfactionScore", 3)),
            "complain":           int(row.get("Complain", 0)),
            "recommended_action": actions.get(seg, "Re-engagement campaign"),
            "segment":            seg,
            "tenure":             int(row.get("Tenure", 0)),
            "cashback_amount":    round(float(row.get("CashbackAmount", 0)), 0),
            "order_hike":         round(float(row.get("OrderAmountHikeFromlastYear", 0)), 1),
        })

    exec_cost    = hr_count * COST_PER_INTERVENTION
    crit_count   = seg_data["critical"]["count"]
    disc_cost    = crit_count * AVG_ORDER_VALUE * 0.20
    total_invest = exec_cost + disc_cost
    rev_critical = crit_count * AVG_ANNUAL_REVENUE * RECOVERY_RATES["Critical"]
    rev_high     = seg_data["high"]["count"]   * AVG_ANNUAL_REVENUE * RECOVERY_RATES["High"]
    rev_medium   = seg_data["medium"]["count"] * AVG_ANNUAL_REVENUE * RECOVERY_RATES["Medium"]
    rev_total    = rev_critical + rev_high + rev_medium
    roi          = round(rev_total / total_invest, 1) if total_invest > 0 else 0.0

    return {
        "summary": {
            "total_customers":  total,
            "high_risk_count":  hr_count,
            "revenue_at_risk":  revenue_at_risk,
            "default_assumptions": {
                "avg_annual_revenue":         AVG_ANNUAL_REVENUE,
                "exec_cost_per_intervention": COST_PER_INTERVENTION,
                "critical_discount_pct":      20,
                "avg_order_value":            AVG_ORDER_VALUE,
            },
            "segments": seg_data,
            "scenario": {
                "investment":          {"execution": round(exec_cost), "discounts": round(disc_cost), "total": round(total_invest)},
                "revenue_recoverable": {"critical": round(rev_critical), "high": round(rev_high), "medium": round(rev_medium), "total": round(rev_total)},
                "roi":                 roi,
            },
        },
        "top_churn_drivers":  top_drivers,
        "customers_at_risk":  at_risk_rows,
        "urgency_weights": {
            "churn_prob": 0.50, "inactivity": 0.25, "complaint": 0.15, "low_sat": 0.10
        },
    }


# ── Startup ────────────────────────────────────────────────────────────────────

print("Loading dataset…")
DF = load_dataset()

print("Training model…")
MODEL, IMPORTANCES, FEATURES = train_model(DF)
MODEL_NAME = "XGBoost" if USE_XGB else "Random Forest"
print(f"  {MODEL_NAME} trained on {len(FEATURES)} features")

DASHBOARD_CACHE = compute_dashboard(DF, MODEL, IMPORTANCES, FEATURES)
s    = DASHBOARD_CACHE["summary"]
segs = s["segments"]
print(f"  Dashboard ready — {s['high_risk_count']:,} at-risk | "
      f"Critical: {segs['critical']['count']} | "
      f"High: {segs['high']['count']} | "
      f"Medium: {segs['medium']['count']}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard")
def dashboard():
    return jsonify(DASHBOARD_CACHE)


@app.route("/api/scenario", methods=["POST"])
def scenario():
    data        = request.get_json() or {}
    avg_rev     = float(data.get("avg_annual_revenue", AVG_ANNUAL_REVENUE))
    exec_cost_u = float(data.get("exec_cost", COST_PER_INTERVENTION))
    disc_pct    = float(data.get("discount_pct", 20)) / 100

    segs         = DASHBOARD_CACHE["summary"]["segments"]
    crit_count   = segs["critical"]["count"]
    high_count   = segs["high"]["count"]
    med_count    = segs["medium"]["count"]
    total_at_risk = DASHBOARD_CACHE["summary"]["high_risk_count"]

    execution  = round(total_at_risk * exec_cost_u)
    discounts  = round(crit_count * AVG_ORDER_VALUE * disc_pct)
    total_inv  = execution + discounts

    rev_crit = round(crit_count * avg_rev * RECOVERY_RATES["Critical"])
    rev_high = round(high_count * avg_rev * RECOVERY_RATES["High"])
    rev_med  = round(med_count  * avg_rev * RECOVERY_RATES["Medium"])
    rev_total = rev_crit + rev_high + rev_med

    roi = round(rev_total / total_inv, 1) if total_inv > 0 else 0.0

    return jsonify({
        "investment":          {"execution": execution, "discounts": discounts, "total": total_inv},
        "revenue_recoverable": {"critical": rev_crit, "high": rev_high, "medium": rev_med, "total": rev_total},
        "roi":                 roi,
        "segment_counts":      {"critical": crit_count, "high": high_count, "medium": med_count},
        "segment_rates":       {"critical": RECOVERY_RATES["Critical"], "high": RECOVERY_RATES["High"], "medium": RECOVERY_RATES["Medium"]},
    })


@app.route("/api/intervene", methods=["POST"])
def intervene():
    data    = request.get_json()
    cid     = data.get("customer_id")
    segment = data.get("segment", "High")

    customer = next((c for c in DASHBOARD_CACHE["customers_at_risk"] if c["customer_id"] == cid), None)
    if not customer:
        return jsonify({"error": "Customer not found"}), 404

    prompt = f"""You are a customer retention specialist for an e-commerce company.
Write a warm, personalized 2-sentence retention message for this at-risk customer.
Be specific to their situation. Include a concrete offer. No subject line, no greeting — just the 2 sentences.

Customer #{cid}:
- Churn risk score: {customer['churn_score']:.0%} ({segment} segment)
- Days since last order: {customer['days_inactive']}
- Satisfaction score: {customer['satisfaction_score']} / 5
- Filed a complaint: {'Yes' if customer['complain'] else 'No'}
- Tenure: {customer['tenure']} months as customer
- Available cashback: ${customer['cashback_amount']:.0f}
- Recommended action: {customer['recommended_action']}"""

    response = gemini.generate_content(prompt)
    return jsonify({"message": response.text.strip()})


@app.route("/api/explain", methods=["POST"])
def explain():
    data   = request.get_json() or {}
    raw_id = str(data.get("customer_id", "")).replace("#", "").strip()

    try:
        cid = int(raw_id)
    except ValueError:
        return jsonify({"error": "Invalid customer ID"}), 400

    customer = next((c for c in DASHBOARD_CACHE["customers_at_risk"] if c["customer_id"] == cid), None)
    if not customer:
        return jsonify({"error": f"Customer #{cid} not found in at-risk list"}), 404

    c = customer
    prompt = f"""You are a customer success AI briefing a retention agent before a call.
In 2–3 sentences, explain plainly why Customer #{cid} is at risk of leaving, which signals are driving the score, and what the agent should prioritize when contacting them.
Be specific, use the data below, and do not repeat the numbers verbatim — interpret them.

Customer #{cid} profile:
- Churn risk score: {c['churn_score']:.0%} ({c['segment']} segment)
- Days since last order: {c['days_inactive']} days
- Satisfaction score: {c['satisfaction_score']} / 5
- Complaint filed: {'Yes' if c['complain'] else 'No'}
- Tenure: {c['tenure']} months
- Cashback not redeemed: ${c['cashback_amount']:.0f}
- Order value trend YoY: {c.get('order_hike', 0):+.1f}%
- Recommended action: {c['recommended_action']}

Write the explanation, then on a new line write "ACTION:" followed by 1 sentence of specific guidance for the agent opening the conversation."""

    response = gemini.generate_content(prompt)
    raw_text = response.text.strip()

    if "ACTION:" in raw_text:
        parts       = raw_text.split("ACTION:", 1)
        explanation = parts[0].strip()
        action_ctx  = parts[1].strip()
    else:
        explanation = raw_text
        action_ctx  = c["recommended_action"]

    signals = []
    if c["days_inactive"] > 20:
        signals.append({"label": "Days since last order", "value": f"{c['days_inactive']}d inactive", "severity": "critical" if c["days_inactive"] > 35 else "high"})
    if c["complain"]:
        signals.append({"label": "Complaint filed", "value": "Yes — unresolved", "severity": "critical"})
    if c["satisfaction_score"] <= 2:
        signals.append({"label": "Satisfaction score", "value": f"{c['satisfaction_score']} / 5 stars", "severity": "critical"})
    elif c["satisfaction_score"] == 3:
        signals.append({"label": "Satisfaction score", "value": f"{c['satisfaction_score']} / 5 stars", "severity": "high"})
    if c["cashback_amount"] > 0:
        signals.append({"label": "Cashback not redeemed", "value": f"${c['cashback_amount']:.0f} sitting unused", "severity": "medium"})

    return jsonify({
        "customer_id":        f"#{cid}",
        "segment":            c["segment"],
        "churn_score":        c["churn_score"],
        "explanation":        explanation,
        "signals":            signals[:4],
        "recommended_action": c["recommended_action"],
        "action_context":     action_ctx,
    })


@app.route("/query", methods=["POST"])
def query():
    data     = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    s       = DASHBOARD_CACHE["summary"]
    segs    = s["segments"]
    drivers = DASHBOARD_CACHE["top_churn_drivers"]

    context = f"""E-Commerce Churn Prevention — Live Model Context

Dataset: {s['total_customers']:,} customers analyzed
Model: {MODEL_NAME} classifier
At-risk customers: {s['high_risk_count']:,} ({s['high_risk_count']/s['total_customers']:.1%} of base)
Revenue at risk: ${s['revenue_at_risk']:,.0f} (annualized)

Risk segments:
  Critical: {segs['critical']['count']} customers — ${segs['critical']['revenue']:,} revenue at risk — 15% recovery rate
  High:     {segs['high']['count']} customers — ${segs['high']['revenue']:,} revenue at risk — 25% recovery rate
  Medium:   {segs['medium']['count']} customers — ${segs['medium']['revenue']:,} revenue at risk — 35% recovery rate

Top churn drivers (model feature importance):
{chr(10).join(f"  {d['feature']}: {d['impact']:.1%}" for d in drivers)}

Financial assumptions: ${AVG_ANNUAL_REVENUE:,} avg annual revenue · recovery rates vary by segment · ${COST_PER_INTERVENTION}/intervention · ${AVG_ORDER_VALUE} avg order value."""

    prompt = f"""You are an AI business analyst briefing a VP of Customer Success.
Answer in 3–5 sentences. Be specific with numbers. Be actionable and direct.

DATA:
{context}

QUESTION: {question}

ANSWER:"""

    response = gemini.generate_content(prompt)
    sources  = [
        f"Model: {MODEL_NAME} — {s['total_customers']:,} customers",
        "Dataset: E-Commerce Churn (Kaggle / Synthetic)",
        "Scope: active customers ≤90 days",
    ]
    return jsonify({"answer": response.text.strip(), "sources": sources})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
