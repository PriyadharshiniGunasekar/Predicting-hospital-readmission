import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import joblib
import os

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("data/hospital_readmissions_30k.csv")

# ==============================
# 2. Preprocessing & Feature Engineering
# ==============================
if "patient_id" in df.columns:
    df = df.drop(columns=["patient_id"])

# --- Blood Pressure Split ---
if "blood_pressure" in df.columns:
    df[["systolic_bp", "diastolic_bp"]] = df["blood_pressure"].str.split("/", expand=True)
    df["systolic_bp"] = pd.to_numeric(df["systolic_bp"], errors="coerce")
    df["diastolic_bp"] = pd.to_numeric(df["diastolic_bp"], errors="coerce")
    df = df.drop(columns=["blood_pressure"])

# Convert Yes/No to 1/0
for col in ["diabetes", "hypertension"]:
    if col in df.columns and df[col].dtype == "object":
        df[col] = df[col].map({"Yes": 1, "No": 0})

# Encode gender (Male=1, Female=0, Other=2)
if "gender" in df.columns and df["gender"].dtype == "object":
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0, "Other": 2})

# Encode discharge_destination
if "discharge_destination" in df.columns and df["discharge_destination"].dtype == "object":
    le = LabelEncoder()
    df["discharge_destination"] = le.fit_transform(df["discharge_destination"])

# Handle missing values
df = df.fillna(0)

# --- Feature Engineering ---
df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
df["high_bp_flag"] = np.where((df["systolic_bp"] > 140) | (df["diastolic_bp"] > 90), 1, 0)
df["comorbidity_index"] = df[["diabetes", "hypertension"]].sum(axis=1)

df["bmi_category"] = pd.cut(
    df["bmi"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]
).astype(int)

df["discharge_risk"] = df["discharge_destination"].apply(
    lambda x: 1 if x in [1, 2] else 0
)

df["meds_per_day"] = df["medication_count"] / (df["length_of_stay"] + 1)
df["meds_per_comorbidity"] = df["medication_count"] / (df["comorbidity_index"] + 1)

# --- Clinical Features ---
df["age_risk_flag"] = np.where(df["age"] >= 60, 1, 0)
df["metabolic_syndrome"] = np.where(
    (df["diabetes"] == 1) & (df["hypertension"] == 1) & (df["bmi"] >= 30), 1, 0
)

# ==============================
# 3. Define Features & Target
# ==============================
target = "readmitted_30_days"
df[target] = df[target].map({"Yes": 1, "No": 0})
X = df.drop(columns=[target])
y = df[target]

# ==============================
# 4. Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 5. Scale Features
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, "models/feature_names.pkl")

# ==============================
# 6. Train XGBoost with Class Weights
# ==============================
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

# ==============================
# 7. Calibrate Probabilities
# ==============================
calibrated_model = CalibratedClassifierCV(xgb_model, method="isotonic", cv=5)
calibrated_model.fit(X_train, y_train)

# ==============================
# 8. Save Model
# ==============================
joblib.dump(calibrated_model, "models/xgb_model.pkl")

print("✅ Training complete. Model + scaler + feature_names saved in models/")
