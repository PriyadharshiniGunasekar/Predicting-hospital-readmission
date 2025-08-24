import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
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

# Convert Yes/No to 1/0 for categorical flags
for col in ["diabetes", "hypertension"]:
    if col in df.columns and df[col].dtype == "object":
        df[col] = df[col].map({"Yes": 1, "No": 0})

# Encode gender (Male=1, Female=0)
if "gender" in df.columns and df["gender"].dtype == "object":
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

# Encode discharge_destination
if "discharge_destination" in df.columns and df["discharge_destination"].dtype == "object":
    le = LabelEncoder()
    df["discharge_destination"] = le.fit_transform(df["discharge_destination"])

# Handle missing values
df = df.fillna(0)

# --- Feature Engineering ---
if "systolic_bp" in df.columns and "diastolic_bp" in df.columns:
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["high_bp_flag"] = np.where((df["systolic_bp"] > 140) | (df["diastolic_bp"] > 90), 1, 0)

if "diabetes" in df.columns and "hypertension" in df.columns:
    df["comorbidity_index"] = df[["diabetes", "hypertension"]].sum(axis=1)
else:
    df["comorbidity_index"] = 0

if "bmi" in df.columns:
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]
    ).astype(int)
else:
    df["bmi_category"] = 0

if "discharge_destination" in df.columns:
    df["discharge_risk"] = df["discharge_destination"].apply(
        lambda x: 1 if x in [1, 2] else 0  # adjust if label encoding differs
    )
else:
    df["discharge_risk"] = 0

if "medication_count" in df.columns and "length_of_stay" in df.columns:
    df["meds_per_day"] = df["medication_count"] / (df["length_of_stay"] + 1)
else:
    df["meds_per_day"] = 0

if "medication_count" in df.columns and "comorbidity_index" in df.columns:
    df["meds_per_comorbidity"] = df["medication_count"] / (df["comorbidity_index"] + 1)
else:
    df["meds_per_comorbidity"] = 0

# ==============================
# 3. Define Features & Target
# ==============================
target = "readmitted_30_days"
df[target] = df[target].map({"Yes": 1, "No": 0})
X = df.drop(columns=[target])
y = df[target]

# ==============================
# 4. Balance Dataset using SMOTE
# ==============================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ==============================
# 5. Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# ==============================
# 6. Scale Features
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# ==============================
# 7. Train XGBoost Classifier
# ==============================
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 8. Save Model
# ==============================
joblib.dump(model, "models/xgb_model.pkl")

print("✅ Training complete. Balanced XGBoost model saved as models/xgb_model.pkl")
