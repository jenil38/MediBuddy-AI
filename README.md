## Live Demo
[https://medibuddy-ai.onrender.com](https://medibuddy-ai-8nse.onrender.com)
# 🩺 MediBuddy — AI Symptom Checker

> An intelligent medical symptom checker that suggests possible diagnoses and provides friendly health information powered by AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)
![Groq](https://img.shields.io/badge/AI-Groq%20LLaMA-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌟 Features

- **Symptom checker** — select symptoms and get ranked disease matches from a 300+ disease dataset
- **AI disease info** — click any diagnosis to get a plain-English explanation powered by LLaMA AI
- **MediBuddy chatbot** — conversational AI assistant for symptom guidance
- **Severity scoring** — rate how severe each symptom is for better accuracy
- **Medicine suggestions** — common over-the-counter treatments for matched conditions
- **Autocomplete search** — quickly find symptoms from 130+ options

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI | Groq API (LLaMA 3.1) |
| Data | Pandas, CSV dataset (300+ diseases) |
| Frontend | HTML, CSS, JavaScript |
| Matching | Fuzzy symptom matching (difflib) |

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/medibuddy.git
cd medibuddy
```

### 2. Install dependencies
```bash
pip install flask groq pandas python-dotenv
```

### 3. Set up your API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 4. Run the app
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

---

## 📁 Project Structure

```
medibuddy/
├── app.py                          # Flask routes and AI integration
├── illness_logic.py                # Symptom matching and diagnosis logic
├── Disease_and_symptoms_dataset.csv  # Medical dataset (300+ diseases)
├── templates/
│   ├── index.html                  # Home page - symptom selection
│   └── result.html                 # Results page - diagnoses
├── .env                            # API keys (not in GitHub)
├── .gitignore
└── README.md
```

---

## ⚠️ Disclaimer

MediBuddy is for **educational and informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified doctor for medical concerns.

---

## 👨‍💻 Author

Built by [Your Name] — [your LinkedIn or GitHub link]
```
## Dataset
Download from Kaggle and place in root folder as:
`Disease and symptoms dataset.csv`
