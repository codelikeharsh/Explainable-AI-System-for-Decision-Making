import pandas as pd
import shap


# -----------------------------
# 1. Load prepared dataset
# -----------------------------
df = pd.read_csv("loan_prepared.csv")

# -----------------------------
# 2. Select features (X)
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
# 3. Select target (y)
# -----------------------------
y = df["Loan_Status"]

from sklearn.model_selection import train_test_split

# -----------------------------
# 5. Split data into train & test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LogisticRegression

# -----------------------------
# 6. Train Logistic Regression model
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

model.fit(X_train, y_train)

print("\n✅ Logistic Regression model trained successfully")
# -----------------------------
# 7. Make predictions
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("\nSample Predictions (first 5):")
for i in range(5):
    print(
        f"Predicted: {y_pred[i]}, "
        f"Probability Approved: {y_prob[i][1]:.2f}, "
        f"Actual: {y_test.iloc[i]}"
    )
#evaluting accuracy here 
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

import joblib

# -----------------------------
# Save trained model
# -----------------------------
joblib.dump(model, "loan_model.pkl")

print("✅ Model saved as loan_model.pkl")

