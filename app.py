from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from illness_logic import get_diagnosis, get_all_symptoms, check_emergency, get_medicine_details
from models import db, User, DiagnosisHistory
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()  # loads your .env file
# ── Specialist recommendation database ──────────────────────────────────────
SPECIALIST_MAP = {
    "migraine": "Neurologist",
    "epilepsy": "Neurologist",
    "parkinson": "Neurologist",
    "stroke": "Neurologist",
    "anxiety": "Psychiatrist",
    "depression": "Psychiatrist",
    "bipolar": "Psychiatrist",
    "schizophrenia": "Psychiatrist",
    "heart attack": "Cardiologist",
    "hypertension": "Cardiologist",
    "arrhythmia": "Cardiologist",
    "angina": "Cardiologist",
    "pneumonia": "Pulmonologist",
    "asthma": "Pulmonologist",
    "copd": "Pulmonologist",
    "tuberculosis": "Pulmonologist",
    "diabetes": "Endocrinologist",
    "thyroid": "Endocrinologist",
    "hypothyroidism": "Endocrinologist",
    "hyperthyroidism": "Endocrinologist",
    "arthritis": "Rheumatologist",
    "lupus": "Rheumatologist",
    "gout": "Rheumatologist",
    "kidney": "Nephrologist",
    "urinary": "Nephrologist",
    "cancer": "Oncologist",
    "tumor": "Oncologist",
    "skin": "Dermatologist",
    "acne": "Dermatologist",
    "psoriasis": "Dermatologist",
    "eczema": "Dermatologist",
    "eye": "Ophthalmologist",
    "vision": "Ophthalmologist",
    "ear": "Otolaryngologist",
    "nose": "Otolaryngologist",
    "throat": "Otolaryngologist",
    "dental": "Dentist",
    "tooth": "Dentist",
    "gum": "Dentist",
    "pregnancy": "Gynecologist",
    "menopause": "Gynecologist",
    "prostate": "Urologist",
}

def get_specialist(disease_name: str) -> str:
    """Return recommended specialist for a disease."""
    disease_lower = disease_name.lower()
    for keyword, specialist in SPECIALIST_MAP.items():
        if keyword in disease_lower:
            return specialist
    return "General Physician"
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///medibuddy.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # key is now hidden safely

with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    symptoms = get_all_symptoms()
    return render_template("index.html", all_symptoms=symptoms)


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not username or not password:
            flash("Please fill in all fields.", "error")
            return render_template("register.html")

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html")

        if User.query.filter_by(username=username).first():
            flash("That username is already taken.", "error")
            return render_template("register.html")

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        flash("Account created! Welcome to MediBuddy.", "success")
        return redirect(url_for("home"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(url_for("home"))

        flash("Incorrect username or password.", "error")
        return render_template("login.html")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You've been logged out.", "success")
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    """Health dashboard showing diagnosis history and health summary."""
    diagnoses = DiagnosisHistory.query.filter_by(user_id=current_user.id).order_by(
        DiagnosisHistory.created_at.desc()
    ).all()
    
    # Calculate health metrics
    total_diagnoses = len(diagnoses)
    last_checkup = diagnoses[0].created_at if diagnoses else None
    
    # Calculate health score (average confidence of top diagnoses, with recent bias)
    health_score = 0
    if diagnoses:
        recent = diagnoses[:5]  # last 5 diagnoses weighted more
        scores = [d.confidence for d in recent]
        health_score = round(sum(scores) / len(scores)) if scores else 0
        health_score = min(health_score, 95)  # cap at 95% for realism
    
    # Find most reported symptom
    most_common_symptom = None
    if diagnoses:
        symptom_count = {}
        for diag in diagnoses:
            symptoms = [s.strip() for s in diag.symptoms.split(",")]
            for sym in symptoms:
                symptom_count[sym] = symptom_count.get(sym, 0) + 1
        if symptom_count:
            most_common_symptom = max(symptom_count, key=symptom_count.get)
    
    # ── Chart 1: Most common diseases (pie chart) ──
    disease_count = {}
    for diag in diagnoses:
        disease = diag.top_disease
        disease_count[disease] = disease_count.get(disease, 0) + 1
    
    # Sort and limit to top 6 for readability
    top_diseases = sorted(disease_count.items(), key=lambda x: x[1], reverse=True)[:6]
    disease_labels = [d[0] for d in top_diseases]
    disease_data = [d[1] for d in top_diseases]
    
    # ── Chart 2: Monthly diagnosis trend (bar chart) ──
    from datetime import datetime, timedelta
    from collections import defaultdict
    
    monthly_count = defaultdict(int)
    for diag in diagnoses:
        month_key = diag.created_at.strftime("%Y-%m")  # Format: "2026-01"
        monthly_count[month_key] += 1
    
    # Get last 6 months
    sorted_months = sorted(monthly_count.items())[-6:]
    month_labels = [m[0] for m in sorted_months]
    month_data = [m[1] for m in sorted_months]
    
    return render_template(
        "dashboard.html",
        user=current_user,
        diagnoses=diagnoses,
        total_diagnoses=total_diagnoses,
        health_score=health_score,
        last_checkup=last_checkup,
        most_common_symptom=most_common_symptom,
        disease_labels=disease_labels,
        disease_data=disease_data,
        month_labels=month_labels,
        month_data=month_data,
    )

@app.route("/result", methods=["POST"])
def result():
    symptoms = request.form.getlist("symptoms")
    severities = request.form.getlist("severity")
    diagnosis_list = get_diagnosis(symptoms, severities)
    emergency_symptoms = check_emergency(symptoms)
    
    # Save diagnosis to database if user is logged in
    if current_user.is_authenticated and diagnosis_list:
        top_disease, _, _, _, confidence = diagnosis_list[0]
        diagnosis_record = DiagnosisHistory(
            user_id=current_user.id,
            symptoms=", ".join(symptoms),
            top_disease=top_disease,
            confidence=confidence
        )
        db.session.add(diagnosis_record)
        db.session.commit()
    
    return render_template(
        "result.html",
        symptoms=symptoms,
        severities=severities,
        diagnoses=diagnosis_list,
        emergency_symptoms=emergency_symptoms,
        get_medicine_details=get_medicine_details,
        # Find specialist (add this line in the return statement)
        specialist = get_specialist(diagnosis_list[0][0]) if diagnosis_list else "General Physician"
    )


@app.route("/disease-info", methods=["POST"])
def disease_info():
    data = request.get_json()
    disease = data.get("disease", "").strip()
    symptoms = data.get("symptoms", [])

    if not disease:
        return jsonify({"info": "No disease specified."})

    prompt = (
        f"The patient has been matched with: {disease}.\n"
        f"Their reported symptoms were: {', '.join(symptoms)}.\n\n"
        "Please provide a clear, friendly medical overview using EXACTLY this structure:\n\n"
        "**What is it?**\n"
        "2-3 sentences explaining what this condition is.\n\n"
        "**Why does it happen?**\n"
        "2-3 sentences on the common causes.\n\n"
        "**Common symptoms**\n"
        "A short bullet list of 4-5 key symptoms.\n\n"
        "**What to do**\n"
        "2-3 practical next steps the patient can take.\n\n"
        "**When to see a doctor**\n"
        "1-2 sentences on warning signs that need urgent care.\n\n"
        "Keep it simple, warm, and easy to understand. No jargon."
    )

    try:
        response = _client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=600,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are MediBuddy AI, a warm medical information assistant. "
                        "Explain medical conditions in plain, friendly language. "
                        "Always end with: '⚠️ This is for informational purposes only — please consult a doctor for proper diagnosis and treatment.'"
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
        info = response.choices[0].message.content
    except Exception as e:
        info = f"Could not load disease information. ({e})"

    return jsonify({"info": info})


@app.route("/symptoms-list")
def symptoms_list():
    """JSON endpoint for autocomplete."""
    return jsonify(get_all_symptoms())

@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html")

@app.route("/chatbot-response", methods=["POST"])
@login_required
def chatbot_response():
    user_message = request.json.get("message", "")
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are MediBuddy AI Health Assistant. Provide helpful health information and advice. Always remind users to consult healthcare professionals for serious conditions."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
           model="llama-3.3-70b-versatile",       )
        
        response_text = chat_completion.choices[0].message.content
        return jsonify({"response": response_text})
    
    except Exception as e:
        print(f"Groq Error: {str(e)}")
        return jsonify({"error": str(e), "response": f"Sorry, I encountered an error: {str(e)}"}), 500
if __name__ == "__main__":
    app.run(debug=True)
