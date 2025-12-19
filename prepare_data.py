import pandas as pd

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("loan.csv")

# -----------------------------
# 2. Clean column names
# (removes leading/trailing spaces)
# -----------------------------
df.columns = df.columns.str.strip()
df["loan_status"] = df["loan_status"].str.strip()

# -----------------------------
# 3. Create Total Assets
# -----------------------------
df["Total_Assets"] = (
    df["residential_assets_value"]
    + df["commercial_assets_value"]
    + df["luxury_assets_value"]
    + df["bank_asset_value"]
)

# -----------------------------
# 4. Convert loan_status to numeric
# -----------------------------
df["Loan_Status"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

# -----------------------------
# 5. Check if any values failed to map
# -----------------------------
if df["Loan_Status"].isnull().any():
    print("⚠️ Warning: Some loan_status values could not be mapped")
    print(df["loan_status"].unique())

# -----------------------------
# 6. Print distribution
# -----------------------------
print("Loan Status Distribution:")
print(df["Loan_Status"].value_counts())

# -----------------------------
# 7. Save cleaned dataset
# -----------------------------
df.to_csv("loan_prepared.csv", index=False)

print("\n✅ Dataset prepared successfully as loan_prepared.csv")
