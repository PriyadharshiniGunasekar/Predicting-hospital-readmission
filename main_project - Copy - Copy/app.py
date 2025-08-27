import os, io, csv, datetime, smtplib, joblib, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from dotenv import load_dotenv
from email.mime.text import MIMEText
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# PDF libs
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from transformers import pipeline

try:
    # Lightweight summarization model
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    print("✅ Summarizer loaded successfully")
except Exception as e:
    summarizer = None
    print("⚠️ Summarizer not available:", e)

app = Flask(__name__)
app.secret_key = "supersecret"   # Needed for flash messages

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =====================
# Load Model + Encoders
# =====================
try:
    model = joblib.load("models/xgboost_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    num_cols = joblib.load("models/num_cols.pkl")
    le_gender = joblib.load("models/le_gender.pkl")
    le_discharge = joblib.load("models/le_discharge.pkl")
    le_bmi = joblib.load("models/le_bmi.pkl")
    print("✅ Models loaded successfully")
except Exception as e:
    print("❌ Missing model/scaler/encoder files:", e)

# =====================
# Email Setup
# =====================
load_dotenv()
FROM_EMAIL = os.getenv("EMAIL_USER", "")
EMAIL_PASS = os.getenv("EMAIL_PASS", "")

def send_email(to_email: str, subject: str, body: str) -> str:
    if not FROM_EMAIL or not EMAIL_PASS:
        return "Email disabled: set EMAIL_USER and EMAIL_PASS in .env"
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = FROM_EMAIL
        msg["To"] = to_email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            print("📧 Gmail login with:", FROM_EMAIL, EMAIL_PASS)
            server.login(FROM_EMAIL, EMAIL_PASS)
            server.send_message(msg)
        return "Sent"
    except Exception as e:
        return f"Failed: {e}"
# =====================
# Hugging Face Summary Generator
# =====================
def generate_summary(patient_id, doctor_name, features_dict, prob, risk, context_notes=None):
    age = features_dict.get("age", "N/A")
    bmi = features_dict.get("bmi", "N/A")
    chol = features_dict.get("cholesterol", "N/A")
    los = features_dict.get("length_of_stay", "N/A")

    context = f" Additional notes: {context_notes}." if context_notes else ""

    if risk == "High":
        action = "Immediate post-discharge planning and close monitoring are strongly recommended."
    elif risk == "Medium":
        action = "Consider scheduling follow-up and reviewing medication adherence."
    else:
        action = "Routine follow-up may suffice, but continue monitoring for changes."

    base_summary = (
        f"Patient {patient_id} under Dr. {doctor_name} has readmission probability {prob:.2f} "
        f"with risk categorized as {risk}. Age: {age}, BMI: {bmi}, cholesterol: {chol}, "
        f"length of stay: {los}.{context} {action}"
    )
    if summarizer:
        try:
            final_summary = summarizer(base_summary, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
            return final_summary
        except Exception as e:
            return base_summary + f" (⚠️ Summarizer failed: {e})"
    else:
        return base_summary + " (ℹ️ Summarizer not configured)"

        
# =====================
# PDF Helpers
# =====================
def _probability_bar(prob: float) -> bytes:
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(4.5, 0.6))
    
    # ✅ Color based on risk thresholds
    if prob >= 0.30:
        color = '#ff4757'   # red for High
    elif prob >= 0.15:
        color = '#ffa502'   # orange/amber for Medium
    else:
        color = '#2ed573'   # green for Low
    
    ax.barh([0], [prob], height=0.6, color=color, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Probability", fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_patient_report(patient_id, doctor_name, doctor_email, features_dict, prob, risk, summary_text="") -> bytes:
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/Patient_{patient_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=42, rightMargin=42)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Hospital Readmission Risk Report", styles["Title"]))
    story.append(Spacer(1, 12))

    pt_tbl = Table([
        ["Patient ID", patient_id],
        ["Doctor Name", doctor_name],
        ["Doctor Email", doctor_email],
        ["Report Generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ], colWidths=[120, 350])
    pt_tbl.setStyle(TableStyle([("BOX", (0,0), (-1,-1), 0.5, colors.grey)]))
    story.append(pt_tbl)
    story.append(Spacer(1, 12))

    if risk == "High":
        color = colors.red
    elif risk == "Medium":
        color = colors.orange
    else:
        color = colors.green

    sum_tbl = Table([["Predicted Risk", risk], ["Probability", f"{prob:.2f}"]], colWidths=[120, 350])
    sum_tbl.setStyle(TableStyle([("TEXTCOLOR", (1,0), (1,0), color),("BOX", (0,0), (-1,-1), 0.5, colors.grey)]))
    story.append(sum_tbl)
    story.append(Spacer(1, 12))

    try:
        story.append(Image(io.BytesIO(_probability_bar(prob)), width=5.2*inch, height=0.8*inch))
    except:
        pass

    story.append(Spacer(1, 12))
    story.append(Paragraph("Clinical Recommendations", styles["Heading3"]))
    story.append(Paragraph("• " + " • ".join([
        "Immediate follow-up if High risk",
        "Routine monitoring if Medium risk",
        "Standard discharge if Low risk"
    ]), styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Clinical Summary", styles["Heading3"]))
    story.append(Paragraph(summary_text, styles["Normal"]))

    rows = [["Feature", "Value"]] + [[k.replace("_", " ").title(), str(v)] for k, v in features_dict.items()]
    feat_tbl = Table(rows, colWidths=[200, 270])
    feat_tbl.setStyle(TableStyle([("BOX", (0,0), (-1,-1), 0.5, colors.grey)]))
    story.append(Paragraph("Model Inputs", styles["Heading3"]))
    story.append(feat_tbl)

    doc.build(story)
    with open(filename, "wb") as f:
        f.write(buf.getvalue())
    buf.seek(0)
    return buf.getvalue()


# =====================
# Analytics Functions
# =====================
def get_dashboard_stats():
    """Get statistics for dashboard"""
    if not os.path.exists(LOG_FILE):
        return {
            'total_predictions': 0,
            'high_risk_count': 0,
            'low_risk_count': 0,
            'high_risk_percentage': 0,
            'avg_age': 0,
            'avg_length_stay': 0
        }
    
    df = safe_read_log()
    return {
    'total_predictions': len(df),
    'high_risk_count': len(df[df['risk'] == 'High']),
    'medium_risk_count': len(df[df['risk'] == 'Medium']),
    'low_risk_count': len(df[df['risk'] == 'Low']),
    'high_risk_percentage': round((len(df[df['risk'] == 'High']) / len(df)) * 100, 1) if len(df) > 0 else 0,
    'avg_age': round(df['age'].mean(), 1) if len(df) > 0 else 0,
    'avg_length_stay': round(df['length_of_stay'].mean(), 1) if len(df) > 0 else 0,
    'recent_predictions': df.tail(5).to_dict('records') if len(df) > 0 else []
    }

def create_risk_distribution_chart():
    """Create risk distribution pie chart"""
    if not os.path.exists(LOG_FILE):
        return json.dumps({}, cls=PlotlyJSONEncoder)
    
    df = safe_read_log()
    risk_counts = df['risk'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
    labels=risk_counts.index,
    values=risk_counts.values,
    hole=0.4,
    marker_colors=['#ff4757', '#ffa502', '#2ed573']  # High, Medium, Low
)])

    
    fig.update_layout(
        title="Risk Distribution",
        font=dict(size=14),
        showlegend=True
    )
    
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def create_age_risk_chart():
    """Create age vs risk scatter plot"""
    if not os.path.exists(LOG_FILE):
        return json.dumps({}, cls=PlotlyJSONEncoder)
    
    df = safe_read_log()

    
    fig = px.scatter(
    df, x='age', y='probability',
    color='risk',
    color_discrete_map={
        'High': '#ff4757',
        'Medium': '#ffa502',
        'Low': '#2ed573'
    },
    title='Age vs Risk Probability',
    labels={'probability': 'Risk Probability', 'age': 'Age'}
)

    
    fig.update_layout(font=dict(size=12))
    
    return json.dumps(fig, cls=PlotlyJSONEncoder)
def classify_risk(prob: float, override_flag: bool = False) -> str:
    """Convert probability + override into High/Medium/Low risk classification."""
    if override_flag: 
        return "High"
    if prob >= 0.30:
        return "High"
    elif prob >= 0.15:
        return "Medium"
    return "Low"
       

# =====================
# Logging
# =====================
def safe_read_log():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=LOG_HEADER)
    df = pd.read_csv(LOG_FILE, on_bad_lines="skip")
    # Ensure all expected columns exist
    for col in LOG_HEADER:
        if col not in df.columns:
            df[col] = ""
    
    # ✅ Clean up risk labels (remove spaces, standardize case)
    if "risk" in df.columns:
        df["risk"] = df["risk"].astype(str).str.strip().str.title()
        # Only allow valid values
        df.loc[~df["risk"].isin(["High", "Medium", "Low"]), "risk"] = "Low"

    # ✅ Normalize timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].astype(str).str.replace(" ", "_").str.replace(":", "-"),
            format="%Y-%m-%d_%H-%M-%S",
            errors="coerce"
        )
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d_%H-%M-%S")
    # ✅ Force numeric columns
    for col in ["age", "probability", "length_of_stay"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[LOG_HEADER]
LOG_FILE = "prediction_log.csv"
LOG_HEADER = [
    "timestamp", "patient_id", "doctor_name", "doctor_email",
    "age","gender","cholesterol","bmi","diabetes","hypertension",
    "medication_count","length_of_stay","discharge_destination",
    "systolic_bp","diastolic_bp","pulse_pressure","high_bp_flag",
    "comorbidity_index","bmi_category","discharge_risk",
    "meds_per_day","meds_per_comorbidity",
    "probability","risk","override_applied","override_reasons","email_status","summary"
]

def log_prediction_row(row: list):
    newfile = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        if newfile:
            w.writerow(LOG_HEADER)
        w.writerow(row)

# =====================
# Routes
# =====================

@app.route("/")
def home():
    """Landing page with navigation"""
    return render_template("home.html")
@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Prediction form page"""
    if request.method == "POST":
        try:
            form = request.form
            patient_id = form["patient_id"]
            age = int(form["age"])
            gender = form["gender"]
            bp = form["blood_pressure"]
            cholesterol = float(form["cholesterol"])
            bmi = float(form["bmi"])
            diabetes = 1 if form["diabetes"] == "Yes" else 0
            hypertension = 1 if form["hypertension"] == "Yes" else 0
            medication_count = int(form["medication_count"])
            length_of_stay = int(form["length_of_stay"])
            discharge = form["discharge_destination"].replace(" ", "_")
            doctor_name = form.get("doctor_name", "").strip()
            doctor_email = form.get("doctor_email", "").strip()

            try:
                systolic_bp, diastolic_bp = map(float, bp.split("/"))
            except:
                flash("⚠️ Invalid BP format, use systolic/diastolic", "danger")
                return redirect(url_for("predict"))

            # ------------------
            # Prepare Data
            # ------------------
            data = {
                "age": age,
                "gender": le_gender.transform([gender])[0],
                "cholesterol": cholesterol,
                "bmi": bmi,
                "diabetes": diabetes,
                "hypertension": hypertension,
                "medication_count": medication_count,
                "length_of_stay": length_of_stay,
                "discharge_destination": le_discharge.transform([discharge])[0],
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
            }
            data["pulse_pressure"] = systolic_bp - diastolic_bp
            data["high_bp_flag"] = 1 if systolic_bp > 140 or diastolic_bp > 90 else 0
            data["comorbidity_index"] = diabetes + hypertension

            if bmi < 18.5: bmi_cat = "Underweight"
            elif bmi < 25: bmi_cat = "Normal"
            elif bmi < 30: bmi_cat = "Overweight"
            else: bmi_cat = "Obese"
            data["bmi_category"] = le_bmi.transform([bmi_cat])[0]

            data["discharge_risk"] = data["discharge_destination"]
            data["meds_per_day"] = medication_count / (length_of_stay + 1)
            data["meds_per_comorbidity"] = medication_count / (data["comorbidity_index"] + 1)

            input_df = pd.DataFrame([data]).reindex(columns=feature_names, fill_value=0)
            input_df[num_cols] = scaler.transform(input_df[num_cols])

            # ------------------
            # Prediction
            # ------------------
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0][1]

            # Clinical overrides
            override_flag, reasons = False, []
            if age >= 85:
                override_flag, reasons = True, reasons + ["Age ≥ 85"]
            if age >= 75 and data["comorbidity_index"] >= 2:
                override_flag, reasons = True, reasons + ["Age ≥ 75 with multiple comorbidities"]
            if length_of_stay >= 14:
                override_flag, reasons = True, reasons + ["Prolonged hospital stay ≥ 14 days"]
            if discharge in ["Nursing_Facility", "Rehab"]:
                override_flag, reasons = True, reasons + ["Discharge to nursing/rehab facility"]
            risk_text = classify_risk(prediction_proba, override_flag)

            # ------------------
            # Clinical Summary
            # ------------------
            pdf_features = {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in data.items()}
            summary_text = generate_summary(
                patient_id, doctor_name, pdf_features, float(prediction_proba), risk_text
            )
            report_bytes = generate_patient_report(
                patient_id=patient_id,
                doctor_name=doctor_name,
                doctor_email=doctor_email,
                features_dict=pdf_features,
                prob=float(prediction_proba),
                risk=risk_text,
                summary_text=summary_text   # ✅ Add this
            )

            # ------------------
            # Email sending (only for High risk)
            # ------------------
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            email_status = "Skipped"

            # ✅ Require doctor email for High risk
            if risk_text == "High":
                if not doctor_email:
                    email_status = "Skipped (no doctor email)"
                else:
                    try:
                        msg_body = f"""
                        Patient ID: {patient_id}
                        Probability: {prediction_proba:.2f}
                        Risk Level: {risk_text}

                        Override Applied: {override_flag}
                        Reasons: {', '.join(reasons) if reasons else 'None'}

                        🧠 Clinical Summary:
                        {summary_text}
                        """
                        msg = MIMEText(msg_body)
                        msg["Subject"] = f"Patient {patient_id} Risk Report"
                        msg["From"] = FROM_EMAIL
                        msg["To"] = doctor_email

                        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
                            server.login(FROM_EMAIL, EMAIL_PASS)
                            server.send_message(msg)
                        email_status = "Sent"
                    except Exception as e:
                        email_status = f"Failed: {e}"

            # ------------------
            # Log Row
            # ------------------
            log_row = [
                timestamp, patient_id, doctor_name, doctor_email,
                age, gender, cholesterol, bmi, diabetes, hypertension,
                medication_count, length_of_stay, discharge,
                systolic_bp, diastolic_bp, data["pulse_pressure"], data["high_bp_flag"],
                data["comorbidity_index"], data["bmi_category"], data["discharge_risk"],
                round(data["meds_per_day"],6), round(data["meds_per_comorbidity"],6),
                round(float(prediction_proba),6), risk_text, int(override_flag), "|".join(reasons),
                email_status, summary_text
            ]
            log_prediction_row(log_row)

            # ------------------
            # PDF Report
            # ------------------
            # ----------------------
            # Generate PDF Report (safe)
            # ----------------------
            try:
                _ = generate_patient_report(
                    patient_id=patient_id,
                    doctor_name=doctor_name,
                    doctor_email=doctor_email,
                    features_dict=pdf_features or {},   # ✅ correct param
                    prob=float(prediction_proba),
                    risk=risk_text,
                    summary_text="Generated automatically by MedPredict AI"
                )
            except Exception as e:
                print(f"⚠️ PDF generation failed: {e}")

            # ----------------------
            # Redirect to dashboard
            # ----------------------
            flash(f"Prediction successful! Risk: {risk_text}, Probability: {prediction_proba:.2f}", "success")
            return redirect(url_for("dashboard"))

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            flash("Prediction failed due to an internal error.", "danger")
            return redirect(url_for("predict"))

    return render_template("predict.html")


@app.route("/dashboard")
def dashboard():
    """Main dashboard with statistics and charts"""
    stats = get_dashboard_stats()
    return render_template("dashboard.html", stats=stats)
# ---------------------
# Dashboard JSON APIs
# ---------------------

@app.route("/api/risk-distribution")
def api_risk_distribution():
    if not os.path.exists(LOG_FILE):
        return jsonify({"data": []})

    df = safe_read_log()
    if df.empty:
        return jsonify({"data": []})

    # Clean risk labels
    df["risk"] = df["risk"].astype(str).str.strip().str.title()
    df.loc[~df["risk"].isin(["High", "Medium", "Low"]), "risk"] = "Low"

    risk_counts = df["risk"].value_counts().to_dict()

    data = [{
        "labels": list(risk_counts.keys()),
        "values": list(risk_counts.values()),
        "type": "pie",
        "name": "Risk Mix"
    }]
    return jsonify({"data": data})


@app.route("/api/age-risk")
def api_age_risk():
    if not os.path.exists(LOG_FILE):
        return jsonify({"data": []})

    df = safe_read_log()
    if df.empty:
        return jsonify({"data": []})

    # Force numeric
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df = df.dropna(subset=["age", "probability"])

    data = [{
        "x": df["age"].tolist(),
        "y": df["probability"].tolist(),
        "mode": "markers",
        "type": "scatter",
        "name": "Age vs Risk"
    }]
    return jsonify({"data": data})


@app.route("/api/predictions-timeline")
def api_predictions_timeline():
    if not os.path.exists(LOG_FILE):
        return jsonify({"data": []})

    df = safe_read_log()
    if df.empty:
        return jsonify({"data": []})

    # Safer timestamp parsing
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    grouped = df.groupby(["date", "risk"]).size().unstack(fill_value=0)

    traces = []
    for risk_level in ["High", "Medium", "Low"]:
        if risk_level in grouped.columns:
            traces.append({
                "x": grouped.index.astype(str).tolist(),
                "y": grouped[risk_level].tolist(),
                "type": "bar",
                "name": risk_level
            })

    return jsonify({"data": traces})



@app.route("/patients")
def patients():
    """Patient records page"""
    if not os.path.exists(LOG_FILE):
        flash("No predictions logged yet.", "info")
        return render_template("patients.html", rows=[])
    
    df = safe_read_log()

    # Sort by timestamp descending
    df = df.sort_values('timestamp', ascending=False)
    rows = df.to_dict(orient="records")
    return render_template("patients.html", rows=rows)


@app.route("/analytics")
def analytics():
    """Advanced analytics page"""
    return render_template("analytics.html")
@app.route("/download_pdf/<patient_id>/<timestamp>")
def download_pdf(patient_id, timestamp):
    if not os.path.exists(LOG_FILE):
        flash("No logs found.", "danger")
        return redirect(url_for("dashboard"))

    df = safe_read_log()

    def normalize(ts: str) -> str:
        return ts.replace(" ", "_").replace(":", "-")

    safe_ts = normalize(timestamp)
    df["ts_normalized"] = df["timestamp"].astype(str).apply(normalize)

    # ✅ Match against normalized timestamps instead of raw
    row = df[(df["patient_id"].astype(str) == str(patient_id)) & (df["ts_normalized"] == safe_ts)]

    if row.empty:
        print("❌ No matching record found!")
        print("Available timestamps for this patient:")
        print(df[df["patient_id"].astype(str) == str(patient_id)][["patient_id", "timestamp"]])
        flash("Prediction not found.", "warning")
        return redirect(url_for("dashboard"))

    record = row.iloc[0].to_dict()

    # Build features for PDF
    features = {
        k: record[k]
        for k in df.columns
        if k not in LOG_HEADER[:4] + [
            "probability", "risk", "override_applied",
            "override_reasons", "email_status"
        ]
    }

    pdf_bytes = generate_patient_report(
        patient_id=record["patient_id"],
        doctor_name=record["doctor_name"],
        doctor_email=record["doctor_email"],
        features_dict=features,
        prob=float(record["probability"]),
        risk=record["risk"]
    )

    # ✅ Human-readable filename
    return send_file(
        io.BytesIO(pdf_bytes),
        download_name=f"Patient_{record['patient_id']}_{record['timestamp']}_report.pdf",
        as_attachment=True
    )

@app.route("/delete_record/<patient_id>/<timestamp>")
def delete_record(patient_id, timestamp):
    """Delete a specific record"""
    if not os.path.exists(LOG_FILE):
        flash("No logs found.", "danger")
        return redirect(url_for("patients"))
    
    df = safe_read_log()


    # ✅ Normalize timestamps consistently
    def normalize(ts: str) -> str:
        return ts.replace(" ", "_").replace(":", "-")

    safe_ts = normalize(timestamp)
    df["ts_normalized"] = df["timestamp"].astype(str).apply(normalize)

    # Drop matching row
    df = df[~((df["patient_id"] == patient_id) & (df["ts_normalized"] == safe_ts))]

    # Save back
    df.drop(columns=["ts_normalized"], errors="ignore", inplace=True)
    df.to_csv(LOG_FILE, index=False)
    
    flash(f"Record for Patient {patient_id} deleted successfully.", "success")
    return redirect(url_for("patients"))
@app.route("/export_data")
def export_data():
    """Export all data as CSV"""
    if not os.path.exists(LOG_FILE):
        flash("No data to export.", "warning")
        return redirect(url_for("dashboard"))
    
    return send_file(LOG_FILE, 
                    download_name=f"hospital_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    as_attachment=True)

# =====================
# Run Flask
# =====================
if __name__ == "__main__":
    app.run(debug=True)