🏦 Loan Decision Intelligence
Explainable AI System for Transparent Loan Approval

A full-stack Explainable AI application that predicts loan approval decisions and explains why each decision was made, along with actionable counterfactual advice.

Built using FastAPI, Logistic Regression, SHAP, and React, this system focuses on transparency, auditability, and real-world usability, rather than just accuracy.

🚀 Key Features :

✅ Loan Approval Prediction with probability score
🔍 Explainable AI (SHAP) – feature-level reasoning in plain English
🔁 Counterfactual Advice – minimal changes required to flip the decision
📊 Confidence Visualization (progress bar)
📄 Downloadable PDF Decision Report
🌓 Dark / Light Mode (monochrome, enterprise UI)
🧾 Audit Log View – decision history for traceability
🌐 REST API Backend (FastAPI)
🖥️ Professional React Frontend

🧠 Why This Project Matters

Most ML projects stop at:

“The model predicts Approved / Rejected.”

This project goes further by answering:

Why was this decision made?
Which factors helped or hurt the decision?
What should change to get approval next time?

This is how real AI systems are built in finance, risk, and compliance-heavy domains.

🏗️ System Architecture:
User (React UI)
      |
      |  JSON Request
      v
FastAPI Backend
      |
      |--> Logistic Regression Model
      |--> SHAP Explainer
      |--> Counterfactual Engine
      |
      v
JSON Response
(Decision + Probability + Explanation + Advice)

📊 Model Details

Model: Logistic Regression

Reason:
Interpretable
Probability-based
Industry-standard baseline for credit risk

Features Used:
Annual Income
Loan Amount
Loan Term
CIBIL Score
Total Assets
Target:
Loan Approval (Approved / Rejected)

🔍 Explainability (SHAP)

For each prediction, the system generates:
Feature contributions (positive or negative)
Human-readable explanations like:
“CIBIL Score strongly increased the probability of loan approval.”

This ensures transparent and auditable AI decisions.

🔁 Counterfactual Advice

Instead of just explaining rejection, the system answers:
“What is the smallest change needed to get approval?”

Examples:

Increase CIBIL score to 700
Reduce loan amount by 20%
Increase annual income by 50%

⚠️ These are model-based insights, not financial guarantees.

🖥️ Frontend (React)

Minimal black & white enterprise UI
Full-screen layout (no boxed cards)
Dark mode for professional dashboards
Clear separation of:
Decision
Explanation
Counterfactual advice
PDF export for reports
Audit log for decision history

🌐 Backend (FastAPI)

REST API with input validation (Pydantic)
Loads ML model once at startup
Stateless, fast, and production-ready

Endpoints:
POST /predict → returns decision + explanations