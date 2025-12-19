# =============================
# Imports
# =============================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
from fastapi.middleware.cors import CORSMiddleware


# =============================
# Create FastAPI app
# =============================

app = FastAPI(
    title="Explainable AI Loan Approval API",
    description="Predicts loan approval with explanations and counterfactual advice",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Load trained model ONCE
# =============================

model = joblib.load("loan_model.pkl")

# =============================
# Load background data for SHAP
# =============================

df = pd.read_csv("loan_prepared.csv")

FEATURES = [
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "Total_Assets"
]

X_background = df[FEATURES]

# Create SHAP explainer once
explainer = shap.Explainer(model, X_background)

# =============================
# Feature name mapping
# =============================

FEATURE_NAMES = {
    "income_annum": "Annual Income",
    "loan_amount": "Loan Amount",
    "loan_term": "Loan Term",
    "cibil_score": "CIBIL Score",
    "Total_Assets": "Total Assets"
}

# =============================
# Input schema
# =============================

class LoanApplication(BaseModel):
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    total_assets: float

# =============================
# Helper: predict decision
# =============================

def predict_approval(row: pd.DataFrame):
    prob = model.predict_proba(row)[0][1]
    decision = "Approved" if prob >= 0.5 else "Rejected"
    return decision, prob

# =============================
# SHAP → English explanation
# =============================

def explain_prediction(sample, shap_values):
    explanations = []

    for feature, shap_value in zip(sample.columns, shap_values.values[0]):
        feature_name = FEATURE_NAMES.get(feature, feature)
        magnitude = abs(shap_value)

        if magnitude >= 0.15:
            strength = "strongly"
        elif magnitude >= 0.05:
            strength = "moderately"
        else:
            strength = "slightly"

        direction = "increased" if shap_value > 0 else "decreased"

        explanations.append(
            f"{feature_name} {strength} {direction} the probability of loan approval."
        )

    return explanations

# =============================
# Counterfactual advice
# =============================

def generate_counterfactual_advice(sample):
    advice = []

    original_decision, _ = predict_approval(sample)
    if original_decision == "Approved":
        return ["Loan is already approved. No changes required."]

    # Try improving CIBIL score
    for score in [650, 680, 700, 720]:
        temp = sample.copy()
        temp["cibil_score"] = score
        decision, _ = predict_approval(temp)
        if decision == "Approved":
            advice.append(
                f"Increasing the CIBIL score to {score} could change the decision to Approved."
            )
            break

    # Try increasing income
    base_income = sample["income_annum"].values[0]
    for factor in [1.2, 1.5, 2.0]:
        temp = sample.copy()
        temp["income_annum"] = base_income * factor
        decision, _ = predict_approval(temp)
        if decision == "Approved":
            advice.append(
                f"Increasing annual income by approximately {int((factor - 1) * 100)}% could improve approval chances."
            )
            break

    # Try reducing loan amount
    base_loan = sample["loan_amount"].values[0]
    for factor in [0.9, 0.8, 0.7]:
        temp = sample.copy()
        temp["loan_amount"] = base_loan * factor
        decision, _ = predict_approval(temp)
        if decision == "Approved":
            advice.append(
                f"Reducing the loan amount by approximately {int((1 - factor) * 100)}% could lead to approval."
            )
            break

    if not advice:
        advice.append(
            "No single small change was sufficient to flip the decision. Multiple factors may need improvement."
        )

    return advice

# =============================
# Root endpoint
# =============================

@app.get("/")
def root():
    return {"status": "running", "message": "Explainable AI Loan API is running"}

# =============================
# PREDICT ENDPOINT (FINAL)
# =============================

@app.post("/predict")
def predict_loan(application: LoanApplication):
    """
    This endpoint:
    1. Runs ML prediction
    2. Explains decision using SHAP
    3. Generates counterfactual advice
    """

    # Convert input to DataFrame
    sample = pd.DataFrame([{
        "income_annum": application.income_annum,
        "loan_amount": application.loan_amount,
        "loan_term": application.loan_term,
        "cibil_score": application.cibil_score,
        "Total_Assets": application.total_assets
    }])

    # Prediction
    decision, probability = predict_approval(sample)

    # SHAP explanation
    shap_values = explainer(sample)
    explanation = explain_prediction(sample, shap_values)

    # Counterfactual advice
    advice = generate_counterfactual_advice(sample)

    # Final response
    return {
        "decision": decision,
        "approval_probability": round(probability, 2),
        "explanation": explanation,
        "counterfactual_advice": advice
    }
