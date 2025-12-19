import pandas as pd
import shap
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("loan_model.pkl")

# -----------------------------
# Load prepared dataset
# -----------------------------
df = pd.read_csv("loan_prepared.csv")

# -----------------------------
# Select features
# -----------------------------
features = [
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "Total_Assets"
]

X = df[features]

# -----------------------------
# Create SHAP explainer
# -----------------------------
explainer = shap.Explainer(model, X)

# -----------------------------
# Feature name mapping
# -----------------------------
FEATURE_NAMES = {
    "income_annum": "Annual Income",
    "loan_amount": "Loan Amount",
    "loan_term": "Loan Term",
    "cibil_score": "CIBIL Score",
    "Total Assets": "Total Assets"
}

# -----------------------------
# SHAP → English explanation
# -----------------------------
def explain_prediction(sample, shap_values):
    explanations = []

    for feature, shap_value in zip(sample.columns, shap_values.values[0]):
        feature_name = FEATURE_NAMES.get(feature, feature)

        abs_value = abs(shap_value)

        if abs_value >= 0.15:
            strength = "strongly"
        elif abs_value >= 0.05:
            strength = "moderately"
        else:
            strength = "slightly"

        if shap_value > 0:
            direction = "increased"
        else:
            direction = "decreased"

        sentence = (
            f"{feature_name} {strength} "
            f"{direction} the probability of loan approval."
        )

        explanations.append(sentence)

    return explanations

# -----------------------------
# Prediction helper
# -----------------------------
def predict_approval(model, row):
    prob = model.predict_proba(row)[0][1]
    decision = "Approved" if prob >= 0.5 else "Rejected"
    return decision, prob

# -----------------------------
# Counterfactual advice generator
# -----------------------------
def generate_counterfactual_advice(model, sample):
    advice = []

    original_decision, original_prob = predict_approval(model, sample)

    if original_decision == "Approved":
        return ["Loan is already approved. No changes required."]

    # 1️⃣ Improve CIBIL score
    temp = sample.copy()
    for new_score in [650, 680, 700, 720]:
        temp["cibil_score"] = new_score
        decision, prob = predict_approval(model, temp)
        if decision == "Approved":
            advice.append(
                f"Increasing the CIBIL score to {new_score} could change the decision to Approved."
            )
            break

    # 2️⃣ Increase income
    temp = sample.copy()
    base_income = sample["income_annum"].values[0]
    for factor in [1.2, 1.5, 2.0]:
        temp["income_annum"] = base_income * factor
        decision, prob = predict_approval(model, temp)
        if decision == "Approved":
            advice.append(
                f"Increasing annual income by approximately {int((factor - 1) * 100)}% could improve approval chances."
            )
            break

    # 3️⃣ Reduce loan amount
    temp = sample.copy()
    base_loan = sample["loan_amount"].values[0]
    for factor in [0.9, 0.8, 0.7]:
        temp["loan_amount"] = base_loan * factor
        decision, prob = predict_approval(model, temp)
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

# -----------------------------
# Explain ONE sample
# -----------------------------
sample_index = 0
sample = X.iloc[[sample_index]]

decision, prob = predict_approval(model, sample)

shap_values = explainer(sample)

print("\n📌 Applicant Data:")
print(sample)

print(f"\n📊 Model Decision: {decision} (Approval Probability: {prob:.2f})")

print("\n🧠 Explanation in Plain English:")
for line in explain_prediction(sample, shap_values):
    print("-", line)

print("\n🛠 Counterfactual Advice:")
for line in generate_counterfactual_advice(model, sample):
    print("-", line)

# -----------------------------
# Visual explanation
# -----------------------------
shap.plots.waterfall(shap_values[0])
