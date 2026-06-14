from flask import Flask, render_template, request, jsonify
from illness_logic import get_diagnosis, get_all_symptoms
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()  # loads your .env file

app = Flask(__name__)

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # key is now hidden safely


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    symptoms = get_all_symptoms()
    return render_template("index.html", all_symptoms=symptoms)


@app.route("/result", methods=["POST"])
def result():
    symptoms = request.form.getlist("symptoms")
    severities = request.form.getlist("severity")
    diagnosis_list = get_diagnosis(symptoms, severities)
    return render_template(
        "result.html",
        symptoms=symptoms,
        severities=severities,
        diagnoses=diagnosis_list,
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


@app.route("/chatbot-response", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"reply": "Please describe your symptoms."})

    messages = [
        {
            "role": "system",
            "content": (
                "You are MediBuddy AI, a friendly and knowledgeable medical assistant. "
                "Help users understand their symptoms and suggest possible conditions "
                "and general over-the-counter remedies available in India. "
                "Be empathetic, clear, and concise. "
                "Always end with: '⚠️ This is not a substitute for professional medical advice. Please consult a doctor.' "
                "Keep responses under 120 words."
            )
        }
    ]

    for entry in history[-8:]:
        role = "user" if entry["role"] == "user" else "assistant"
        messages.append({"role": role, "content": entry["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        response = _client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=200,
            messages=messages
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"I'm having trouble connecting right now. Please try again shortly. ({e})"

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
