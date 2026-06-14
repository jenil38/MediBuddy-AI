import pandas as pd
import os
from difflib import get_close_matches

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH = os.path.join(_BASE_DIR, "Disease and symptoms dataset.csv")

try:
    _df = pd.read_csv(_CSV_PATH)
    _df.columns = _df.columns.str.strip().str.lower()
    _DISEASE_COL = "diseases"
    _SYMPTOM_COLS = [c for c in _df.columns if c != _DISEASE_COL]
    _DATASET_LOADED = True
except Exception as e:
    print(f"[illness_logic] WARNING: Could not load dataset: {e}")
    _df = None
    _SYMPTOM_COLS = []
    _DATASET_LOADED = False


def get_all_symptoms() -> list[str]:
    return sorted(_SYMPTOM_COLS)


def get_diagnosis(symptoms: list[str], severities: list[str]) -> list[tuple]:
    input_symptoms = [s.strip().lower() for s in symptoms if s.strip()]
    sev_map = {}
    for i, sym in enumerate(input_symptoms):
        try:
            sev_map[sym] = int(severities[i])
        except (IndexError, ValueError):
            sev_map[sym] = 1

    if _DATASET_LOADED and input_symptoms:
        results = _dataset_match(input_symptoms, sev_map)
        if results:
            return results

    return _rule_based_match(input_symptoms)


def _resolve_symptoms(input_symptoms: list[str]) -> dict[str, str]:
    resolved = {}
    for sym in input_symptoms:
        sym_lower = sym.strip().lower()
        if sym_lower in _df.columns:
            resolved[sym_lower] = sym_lower
            continue
        substr_hits = [col for col in _SYMPTOM_COLS if sym_lower in col or col in sym_lower]
        if substr_hits:
            resolved[sym_lower] = min(substr_hits, key=len)
            continue
        close = get_close_matches(sym_lower, _SYMPTOM_COLS, n=1, cutoff=0.6)
        if close:
            resolved[sym_lower] = close[0]
    return resolved


def _dataset_match(input_symptoms, sev_map):
    sym_map = _resolve_symptoms(input_symptoms)
    if not sym_map:
        return []

    scores = pd.Series(0.0, index=_df.index)
    for user_sym, col in sym_map.items():
        weight = sev_map.get(user_sym, 1)
        scores += _df[col].fillna(0) * weight

    _df["_score"] = scores
    top = (
        _df[_df["_score"] > 0]
        .sort_values("_score", ascending=False)
        .drop_duplicates(subset=[_DISEASE_COL])
        .head(5)
    )
    _df.drop(columns=["_score"], inplace=True)

    if top.empty:
        return []

    results = []
    for _, row in top.iterrows():
        disease = str(row[_DISEASE_COL]).strip().title()
        score = round(row["_score"], 1)
        medicines = _medicine_suggestions(disease)
        results.append((disease, medicines, score))

    return results


_RULES = {
    ("cough", "cold"):                  ("Common Cold",         ["Tab. Cetrizine 10mg (once at night)", "Tab. Paracetamol 500mg (if fever)", "Steam inhalation twice daily"]),
    ("cold", "sneeze"):                 ("Allergic Rhinitis",   ["Tab. Levocetirizine 5mg", "Nasal saline spray", "Avoid dust/allergens"]),
    ("fever", "body pain"):             ("Viral Fever",         ["Tab. Dolo 650 (3× daily)", "ORS sachets", "Rest and plenty of fluids"]),
    ("fever", "headache"):              ("Viral Infection",     ["Tab. Paracetamol 650mg", "ORS solution", "Tab. Azee (if prescribed)"]),
    ("headache",):                      ("Tension Headache",    ["Tab. Crocin 500mg", "Drink plenty of water", "Rest in a dark room"]),
    ("sore throat",):                   ("Throat Infection",    ["Tab. Azithromycin 500mg (3 days)", "Warm saline gargle", "Strepsils lozenges"]),
    ("chest pain", "cough"):            ("Bronchitis",          ["Syrup Benadryl 10ml 3×/day", "Tab. Dolo 650 (if fever)", "Steam inhalation"]),
    ("fever", "rash"):                  ("Dengue (Possible)",   ["Tab. Paracetamol (NO aspirin)", "Hydration essential", "⚠️ Consult a doctor immediately"]),
    ("vomiting", "diarrhea"):           ("Food Poisoning",      ["Tab. Ondem (vomiting)", "Tab. Eldoper (loose motions)", "ORS every 2 hrs"]),
    ("abdominal pain", "vomiting"):     ("Stomach Infection",   ["Tab. Cyclopam", "Tab. Pantop 40 (before meals)", "Electral for hydration"]),
    ("high fever", "chills"):           ("Malaria (Possible)",  ["Tab. Paracetamol", "⚠️ Consult doctor for blood test", "Avoid mosquito exposure"]),
    ("joint pain", "fever"):            ("Chikungunya",         ["Tab. Dolo 650", "Tab. Aceclo-SP (if severe)", "Hydration & rest"]),
    ("burning urine", "abdominal pain"):("UTI",                 ["Tab. Norflox-TZ (twice daily)", "Cranberry syrup", "Tab. Drotin for pain"]),
    ("fatigue", "fever", "muscle pain"):("Influenza",           ["Tab. Tamiflu (if prescribed)", "Dolo 650", "Warm fluids and rest"]),
    ("dry cough", "loss of taste"):     ("COVID-19 (Possible)", ["Tab. Dolo 650", "Zincovit Multivitamin", "⚠️ Isolate & consult a clinic"]),
    ("acidity", "chest burn"):          ("Acidity / GERD",      ["Tab. Pan 40 (before breakfast)", "Tab. Gelusil (after meals)", "Avoid spicy food"]),
    ("back pain", "leg pain"):          ("Sciatica",            ["Tab. Etoshine 90", "Hot-water compression", "Physiotherapy recommended"]),
}

def _rule_based_match(input_symptoms):
    matched = []
    for rule_syms, (illness, meds) in _RULES.items():
        if all(s in input_symptoms for s in rule_syms):
            matched.append((illness, meds, 0))
    return matched or [("Unknown Illness", ["Please consult a qualified doctor."], 0)]


# ── Comprehensive medicine database covering all 378 diseases ─────────────────
_MED_DB = {
    # Respiratory
    "common cold":              ["Tab. Cetrizine 10mg (once at night)", "Tab. Paracetamol 500mg (if fever)", "Steam inhalation twice daily"],
    "flu":                      ["Tab. Dolo 650 (3× daily)", "ORS sachets", "Rest and warm fluids"],
    "influenza":                ["Tab. Tamiflu (if prescribed)", "Tab. Dolo 650", "Warm fluids & rest"],
    "asthma":                   ["Salbutamol inhaler (rescue use)", "Budesonide inhaler (controller)", "Avoid smoke/dust triggers"],
    "pneumonia":                ["Tab. Amoxicillin-Clavulanate 625mg", "Tab. Azithromycin 500mg", "⚠️ Hospital evaluation needed"],
    "tuberculosis":             ["DOTS therapy (6-month course)", "⚠️ Refer to nearest DOTS centre", "Nutritional support essential"],
    "acute bronchitis":         ["Syrup Benadryl 10ml 3×/day", "Tab. Paracetamol 650mg (if fever)", "Steam inhalation & warm fluids"],
    "acute bronchiolitis":      ["Salbutamol nebulisation", "ORS & hydration", "⚠️ Paediatric review needed"],
    "acute bronchospasm":       ["Salbutamol inhaler 2 puffs immediately", "Tab. Prednisolone 10mg", "⚠️ Seek emergency care if severe"],
    "chronic obstructive pulmonary disease (copd)": ["Tiotropium inhaler (daily)", "Salbutamol inhaler (rescue)", "Pulmonary rehabilitation"],
    "whooping cough":           ["Tab. Azithromycin 500mg (5 days)", "Tab. Dolo 650 (if fever)", "Isolate & rest"],
    "croup":                    ["Dexamethasone syrup (as prescribed)", "Cool mist humidifier", "⚠️ Paediatric review if worsening"],
    "laryngitis":               ["Warm saline gargle", "Tab. Ibuprofen 400mg for pain", "Voice rest & hydration"],
    "tracheitis":               ["Tab. Amoxicillin 500mg (5 days)", "Steam inhalation", "Tab. Paracetamol 650mg (if fever)"],
    "pharyngitis":              ["Tab. Azithromycin 500mg (3 days)", "Warm saline gargle", "Strepsils lozenges"],
    "strep throat":             ["Tab. Amoxicillin 500mg (10 days)", "Warm saline gargle", "Tab. Paracetamol for pain"],
    "tonsillitis":              ["Tab. Amoxicillin 500mg (7 days)", "Warm saline gargle", "Tab. Ibuprofen 400mg for pain"],
    "sinusitis":                ["Tab. Amoxicillin 500mg (5 days)", "Nasal saline spray", "Tab. Cetrizine 10mg at night"],
    "acute sinusitis":          ["Tab. Amoxicillin-Clavulanate 625mg", "Nasal decongestant spray", "Steam inhalation"],
    "chronic sinusitis":        ["Nasal corticosteroid spray (Flonase)", "Tab. Levocetirizine 5mg", "Saline nasal rinse daily"],
    "nasal polyp":              ["Nasal corticosteroid spray", "Tab. Montelukast 10mg", "⚠️ ENT evaluation needed"],
    "pulmonary embolism":       ["⚠️ Emergency — call ambulance immediately", "Anticoagulation (hospital)", "Do not delay treatment"],
    "pulmonary fibrosis":       ["Pirfenidone (as prescribed)", "Oxygen therapy (if needed)", "⚠️ Pulmonologist review needed"],
    "pleural effusion":         ["⚠️ Hospital evaluation needed", "Diuretics (as prescribed)", "Treat underlying cause"],
    "pneumothorax":             ["⚠️ Emergency care needed immediately", "Oxygen therapy", "Hospital admission required"],
    "atelectasis":              ["Deep breathing exercises", "Chest physiotherapy", "⚠️ Doctor review needed"],
    "emphysema":                ["Tiotropium inhaler (daily)", "Salbutamol rescue inhaler", "Stop smoking — critical"],
    "acute respiratory distress syndrome (ards)": ["⚠️ ICU admission required", "Mechanical ventilation support", "Treat underlying cause"],
    "pulmonary hypertension":   ["Sildenafil (as prescribed)", "Diuretics (as prescribed)", "⚠️ Cardiologist review needed"],
    "sarcoidosis":              ["Tab. Prednisolone (as prescribed)", "Regular monitoring", "⚠️ Specialist review needed"],
    "interstitial lung disease":["Tab. Prednisolone (as prescribed)", "Oxygen therapy (if needed)", "⚠️ Pulmonologist needed"],
    "lung cancer":              ["⚠️ Oncologist evaluation urgently", "Chemotherapy/radiotherapy (specialist)", "Supportive care"],

    # Infections & Fever
    "dengue fever":             ["Tab. Paracetamol (NO aspirin/ibuprofen)", "ORS every 2 hrs", "⚠️ Hospital if platelet drops"],
    "malaria":                  ["Tab. Coartem/Lumerax (confirmed)", "Tab. Paracetamol", "⚠️ Blood test essential"],
    "typhoid fever":            ["Tab. Azithromycin 500mg (7 days)", "Tab. Cefixime 200mg", "Soft bland diet & hydration"],
    "chickenpox":               ["Calamine lotion (topical)", "Tab. Acyclovir 400mg (5 days)", "Avoid scratching; cut nails short"],
    "shingles (herpes zoster)": ["Tab. Acyclovir 800mg 5×/day (7 days)", "Tab. Pregabalin for nerve pain", "Calamine lotion on rash"],
    "mumps":                    ["Tab. Paracetamol 650mg (for pain/fever)", "Cold/warm compress on swelling", "Rest & soft diet"],
    "mononucleosis":            ["Tab. Paracetamol 650mg", "Rest (avoid contact sports)", "Adequate hydration"],
    "lyme disease":             ["Tab. Doxycycline 100mg (21 days)", "Tab. Amoxicillin 500mg (alternative)", "⚠️ Doctor confirmation needed"],
    "sepsis":                   ["⚠️ Emergency — hospital immediately", "IV antibiotics (hospital)", "ICU care required"],
    "meningitis":               ["⚠️ Medical emergency — call ambulance", "IV antibiotics (hospital)", "Do not delay treatment"],
    "encephalitis":             ["⚠️ Hospital admission urgently", "Tab. Acyclovir IV (hospital)", "Supportive care"],
    "cellulitis or abscess of mouth": ["Tab. Amoxicillin-Clavulanate 625mg", "Warm saline rinse", "⚠️ Dentist review needed"],
    "impetigo":                 ["Mupirocin cream (topical, 3×/day)", "Tab. Cefalexin 500mg (if widespread)", "Keep area clean & dry"],
    "scarlet fever":            ["Tab. Amoxicillin 500mg (10 days)", "Tab. Paracetamol for fever", "Calamine lotion for rash"],
    "cat scratch disease":      ["Tab. Azithromycin 500mg (5 days)", "Warm compress on lymph nodes", "Rest"],
    "toxoplasmosis":            ["Tab. Pyrimethamine + Sulfadiazine (as prescribed)", "⚠️ Doctor supervision needed", "Folic acid supplement"],
    "histoplasmosis":           ["Tab. Itraconazole 200mg (as prescribed)", "⚠️ Specialist review needed", "Rest & hydration"],
    "cryptococcosis":           ["⚠️ Hospital treatment required", "Antifungal IV (hospital)", "Specialist review"],
    "rocky mountain spotted fever": ["Tab. Doxycycline 100mg (7 days)", "⚠️ Doctor confirmation needed", "Rest & hydration"],
    "gas gangrene":             ["⚠️ Surgical emergency — hospital immediately", "IV antibiotics (hospital)", "Hyperbaric oxygen (hospital)"],
    "necrotizing fasciitis":    ["⚠️ Surgical emergency — hospital immediately", "IV antibiotics (hospital)", "Immediate surgery needed"],
    "postoperative infection":  ["Tab. Amoxicillin-Clavulanate 625mg", "Wound dressing & cleaning", "⚠️ Surgeon review needed"],

    # Gastrointestinal
    "gastroesophageal reflux disease (gerd)": ["Tab. Pan 40 (before breakfast)", "Tab. Gelusil (after meals)", "Avoid spicy/oily food & lying down after eating"],
    "gastritis":                ["Tab. Pantoprazole 40mg (before meals)", "Tab. Sucralfate (after meals)", "Avoid NSAIDs, alcohol & spicy food"],
    "gastroduodenal ulcer":     ["Tab. Pantoprazole 40mg (twice daily)", "Tab. Amoxicillin 1g + Clarithromycin 500mg (if H.pylori)", "Avoid NSAIDs & alcohol"],
    "indigestion":              ["Tab. Gelusil (after meals)", "Tab. Domperidone 10mg (before meals)", "Small frequent meals; avoid oily food"],
    "gastroenteritis":          ["ORS sachets every 2 hrs", "Tab. Ondem (vomiting)", "Tab. Eldoper (loose motions)"],
    "infectious gastroenteritis":["ORS sachets every 2 hrs", "Tab. Norflox-TZ (if bacterial)", "Tab. Ondem for vomiting"],
    "noninfectious gastroenteritis":["ORS sachets", "Tab. Domperidone 10mg", "Bland diet (rice, banana, toast)"],
    "irritable bowel syndrome": ["Tab. Mebeverine 135mg (before meals)", "High-fibre diet", "Stress management & regular exercise"],
    "crohn disease":            ["Tab. Mesalazine (as prescribed)", "Tab. Prednisolone (flare)", "⚠️ Gastroenterologist needed"],
    "ulcerative colitis":       ["Tab. Mesalazine 400mg (3×/day)", "Tab. Prednisolone (flare)", "⚠️ Gastroenterologist needed"],
    "appendicitis":             ["⚠️ Surgical emergency — hospital immediately", "Surgery (appendectomy)", "Do not take painkillers before evaluation"],
    "cholecystitis":            ["⚠️ Hospital evaluation needed", "Tab. Ciprofloxacin + Metronidazole", "NPO (nothing by mouth) until evaluated"],
    "gallstone":                ["Tab. Ursodeoxycholic acid (as prescribed)", "Low-fat diet", "⚠️ Surgeon evaluation needed"],
    "choledocholithiasis":      ["⚠️ Hospital evaluation needed", "ERCP procedure (hospital)", "Low-fat diet"],
    "pancreatitis":             ["⚠️ Hospital admission needed", "NPO + IV fluids (hospital)", "Tab. Pantoprazole 40mg"],
    "acute pancreatitis":       ["⚠️ Hospital admission needed", "IV fluids & pain management (hospital)", "Avoid alcohol & fatty foods"],
    "chronic pancreatitis":     ["Pancreatic enzyme supplements (Creon)", "Tab. Pantoprazole 40mg", "Strict alcohol avoidance"],
    "cirrhosis":                ["Tab. Propranolol (for varices prevention)", "Low-sodium diet & diuretics", "⚠️ Hepatologist needed urgently"],
    "hepatitis due to a toxin": ["⚠️ Stop all alcohol/medication immediately", "Tab. Liv 52 (liver support)", "⚠️ Hepatologist evaluation needed"],
    "viral hepatitis":          ["Rest & adequate hydration", "High-carb, low-fat diet", "⚠️ Hepatologist evaluation needed"],
    "jaundice":                 ["Tab. Liv 52 (liver support)", "High-carb, low-fat diet", "⚠️ Medical evaluation needed urgently"],
    "nonalcoholic liver disease (nash)": ["Weight loss programme", "Tab. Vitamin E (as prescribed)", "Low-fat, low-sugar diet"],
    "alcoholic liver disease":  ["⚠️ Stop alcohol immediately", "Tab. Thiamine 100mg", "Nutritional support & Liv 52"],
    "hepatic encephalopathy":   ["⚠️ Hospital admission needed", "Tab. Lactulose (reduces ammonia)", "Low-protein diet temporarily"],
    "liver cancer":             ["⚠️ Oncologist evaluation urgently", "Specialist treatment required", "Supportive care"],
    "intestinal obstruction":   ["⚠️ Surgical emergency — hospital immediately", "NPO + IV fluids (hospital)", "Surgery may be needed"],
    "diverticulitis":           ["Tab. Ciprofloxacin 500mg + Tab. Metronidazole 400mg", "Liquid diet during flare", "High-fibre diet long-term"],
    "colonic polyp":            ["⚠️ Colonoscopy needed", "High-fibre diet", "Regular colorectal screening"],
    "colorectal cancer":        ["⚠️ Oncologist evaluation urgently", "Surgery/chemotherapy (specialist)", "Supportive care"],
    "stomach cancer":           ["⚠️ Oncologist evaluation urgently", "Specialist treatment required", "Nutritional support"],
    "esophagitis":              ["Tab. Pantoprazole 40mg (twice daily)", "Tab. Sucralfate suspension", "Avoid spicy food & alcohol"],
    "hiatal hernia":            ["Tab. Pantoprazole 40mg (before meals)", "Elevate head of bed", "Avoid large meals"],
    "hemorrhoids":              ["Sitz bath (warm water) 3×/day", "Anovate cream (topical)", "High-fibre diet & plenty of water"],
    "anal fissure":             ["Sitz bath (warm water) 3×/day", "Lidocaine gel (topical)", "High-fibre diet & stool softeners"],
    "constipation":             ["Tab. Bisacodyl 5mg (at night)", "Isabgol (psyllium husk) with water", "High-fibre diet & 8 glasses of water/day"],
    "chronic constipation":     ["Lactulose syrup 15ml (twice daily)", "Isabgol (psyllium husk) with water", "High-fibre diet & adequate water"],
    "lactose intolerance":      ["Lactase enzyme tablets (with meals)", "Avoid dairy or use lactose-free milk", "Calcium supplement if needed"],
    "celiac disease":           ["Strict gluten-free diet (no wheat/barley/rye)", "Iron + folic acid supplements", "⚠️ Gastroenterologist follow-up"],
    "intestinal malabsorption": ["Nutritional supplements (as needed)", "Treat underlying cause", "⚠️ Gastroenterologist needed"],
    "dumping syndrome":         ["Small frequent meals (6/day)", "Avoid sugary/sweet foods", "Lie down 30 min after meals"],
    "volvulus":                 ["⚠️ Surgical emergency — hospital immediately", "Surgery required", "Do not delay treatment"],
    "intussusception":          ["⚠️ Surgical emergency — hospital immediately", "Enema reduction or surgery (hospital)", "Paediatric review urgently"],
    "gastrointestinal hemorrhage": ["⚠️ Emergency — hospital immediately", "IV fluids & blood transfusion (hospital)", "Endoscopy needed"],
    "esophageal varices":       ["⚠️ Emergency — hospital immediately", "Endoscopic banding (hospital)", "Tab. Propranolol for prevention"],

    # Cardiovascular
    "hypertension":             ["Tab. Amlodipine 5mg (daily)", "Low-sodium diet (< 2g/day)", "Regular BP monitoring & exercise"],
    "high blood pressure":      ["Tab. Amlodipine 5mg (daily)", "Low-sodium diet", "Regular BP monitoring"],
    "heart attack":             ["⚠️ Emergency — call ambulance immediately", "Aspirin 325mg (chew immediately)", "Do not drive — wait for ambulance"],
    "heart failure":            ["Tab. Furosemide 40mg (diuretic)", "Tab. Enalapril 5mg", "Low-sodium diet & fluid restriction"],
    "angina":                   ["Tab. Sorbitrate (under tongue during attack)", "Tab. Aspirin 75mg (daily)", "Tab. Atenolol 50mg (daily)"],
    "arrhythmia":               ["Tab. Metoprolol 25mg (as prescribed)", "Avoid caffeine & alcohol", "⚠️ Cardiologist evaluation needed"],
    "atrial fibrillation":      ["Tab. Warfarin (as prescribed)", "Tab. Metoprolol 25mg", "⚠️ Cardiologist evaluation needed"],
    "cardiac arrest":           ["⚠️ Emergency — CPR immediately", "Call ambulance — 108", "Defibrillation (hospital)"],
    "cardiomyopathy":           ["Tab. Carvedilol (as prescribed)", "Tab. Enalapril 5mg", "⚠️ Cardiologist needed"],
    "pericarditis":             ["Tab. Ibuprofen 400mg (3×/day)", "Tab. Colchicine 0.5mg", "Rest & avoid strenuous activity"],
    "endocarditis":             ["⚠️ Hospital admission needed", "IV antibiotics (hospital, 4-6 weeks)", "⚠️ Cardiologist needed urgently"],
    "myocarditis":              ["Tab. Ibuprofen 400mg (if prescribed)", "Rest (no exercise)", "⚠️ Cardiologist needed urgently"],
    "deep vein thrombosis (dvt)":["Tab. Rivaroxaban 15mg (as prescribed)", "Compression stockings", "⚠️ Doctor evaluation needed urgently"],
    "varicose veins":           ["Compression stockings (grade 2)", "Elevate legs when resting", "⚠️ Vascular surgeon evaluation"],
    "peripheral arterial disease":["Tab. Aspirin 75mg + Clopidogrel 75mg", "Stop smoking immediately", "⚠️ Vascular surgeon needed"],
    "hypertensive heart disease":["Tab. Amlodipine 5mg + Enalapril 5mg", "Strict low-sodium diet", "⚠️ Cardiologist follow-up"],
    "coronary atherosclerosis": ["Tab. Aspirin 75mg + Atorvastatin 40mg", "Heart-healthy low-fat diet", "⚠️ Cardiologist needed"],
    "ischemic heart disease":   ["Tab. Aspirin 75mg + Atorvastatin 40mg", "Tab. Metoprolol 25mg", "⚠️ Cardiologist evaluation needed"],
    "heart block":              ["⚠️ Cardiologist evaluation needed", "Pacemaker may be required", "Avoid strenuous activity"],
    "pulmonary congestion":     ["Tab. Furosemide 40mg", "Low-sodium diet & fluid restriction", "⚠️ Cardiologist needed"],
    "orthostatic hypotension":  ["Increase fluid & salt intake", "Rise slowly from sitting/lying", "Compression stockings if needed"],
    "raynaud disease":          ["Tab. Nifedipine 10mg (as prescribed)", "Keep hands/feet warm (gloves/socks)", "Avoid cold & stress triggers"],
    "vasculitis":               ["Tab. Prednisolone (as prescribed)", "⚠️ Rheumatologist evaluation needed", "Regular monitoring"],
    "venous insufficiency":     ["Compression stockings (grade 2)", "Elevate legs when resting", "Regular walking exercise"],

    # Diabetes & Endocrine
    "diabetes":                 ["Metformin 500mg (as prescribed)", "Monitor blood sugar daily", "Low-sugar, high-fibre diet"],
    "diabetic ketoacidosis":    ["⚠️ Emergency — hospital immediately", "IV insulin & fluids (hospital)", "Monitor electrolytes (hospital)"],
    "hypoglycemia":             ["15g fast-acting sugar (glucose tabs/juice) immediately", "Recheck sugar in 15 minutes", "⚠️ Consult doctor if recurrent"],
    "diabetic peripheral neuropathy": ["Tab. Pregabalin 75mg (twice daily)", "Optimise blood sugar control", "Methylcobalamin 1500mcg supplement"],
    "hypothyroidism":           ["Tab. Levothyroxine (as prescribed)", "Take on empty stomach in morning", "Regular TSH monitoring"],
    "hyperthyroidism":          ["Tab. Carbimazole (as prescribed)", "Tab. Propranolol 20mg (for symptoms)", "⚠️ Endocrinologist needed"],
    "diabetes insipidus":       ["Tab. Desmopressin (as prescribed)", "Adequate water intake", "⚠️ Endocrinologist needed"],
    "cushing syndrome":         ["⚠️ Endocrinologist evaluation needed", "Surgical treatment if tumour", "Regular monitoring"],
    "goiter":                   ["Iodine-rich diet (if deficiency)", "Tab. Levothyroxine (if hypothyroid)", "⚠️ Endocrinologist review"],
    "graves disease":           ["Tab. Carbimazole (as prescribed)", "Tab. Propranolol 20mg", "⚠️ Endocrinologist needed"],
    "hashimoto thyroiditis":    ["Tab. Levothyroxine (if hypothyroid)", "Regular TSH monitoring", "Anti-inflammatory diet"],
    "polycystic ovarian syndrome (pcos)": ["Tab. Metformin 500mg", "Tab. Clomiphene (if fertility needed)", "Regular exercise & weight management"],
    "gestational diabetes":     ["Blood sugar monitoring (4×/day)", "Insulin (if diet fails)", "Low-sugar diet; regular OB review"],
    "hypercalcemia":            ["⚠️ Hospital evaluation needed", "IV fluids (hospital)", "Treat underlying cause"],
    "hypocalcemia":             ["Tab. Calcium carbonate 500mg (3×/day)", "Vitamin D 60000 IU (weekly)", "High-calcium diet"],
    "hyperkalemia":             ["⚠️ Hospital evaluation needed (can affect heart)", "Kayexalate (as prescribed)", "Low-potassium diet"],
    "hypokalemia":              ["Tab. Potassium chloride (Slow K) as prescribed", "High-potassium foods (banana, spinach)", "⚠️ Doctor evaluation needed"],
    "hyponatremia":             ["⚠️ Hospital evaluation needed", "Fluid restriction or IV saline (hospital)", "Treat underlying cause"],
    "hypernatremia":            ["⚠️ Hospital evaluation needed", "Gradual rehydration (hospital)", "Treat underlying cause"],
    "hyperlipidemia":           ["Tab. Atorvastatin 10-40mg (daily)", "Low-fat, high-fibre diet", "Regular exercise"],
    "hypercholesterolemia":     ["Tab. Atorvastatin 10-40mg (daily)", "Low-fat diet & exercise", "Avoid trans fats"],
    "obesity":                  ["Structured low-calorie diet", "Regular aerobic exercise (150 min/week)", "Behavioural therapy support"],
    "vitamin d deficiency":     ["Vitamin D3 60000 IU (once weekly for 8 weeks)", "Sun exposure (15-20 min daily)", "Calcium 500mg supplement"],
    "vitamin b12 deficiency":   ["Methylcobalamin 1500mcg (daily)", "Tab. Neurobion Forte (daily)", "B12-rich foods (eggs, meat, dairy)"],
    "vitamin a deficiency":     ["Vitamin A 50000 IU (as prescribed)", "Vitamin A-rich foods (carrots, spinach)", "⚠️ Doctor supervision for high doses"],
    "iron deficiency anemia":   ["Ferrous sulphate 200mg (twice daily)", "Vitamin C (with iron for absorption)", "Iron-rich foods (leafy greens, meat)"],
    "anemia":                   ["Iron + Folic acid tablets (daily)", "Vitamin B12 if needed", "Iron-rich diet & adequate hydration"],
    "folate deficiency":        ["Folic acid 5mg (daily)", "Leafy greens, lentils, fortified foods", "⚠️ Check B12 also"],
    "scurvy":                   ["Vitamin C 500mg (3×/day for 2 weeks)", "Citrus fruits & vegetables daily", "Maintain adequate dietary vitamin C"],
    "magnesium deficiency":     ["Tab. Magnesium oxide 400mg (daily)", "Magnesium-rich foods (nuts, leafy greens)", "⚠️ Doctor supervision if severe"],

    # Neurological
    "migraine":                 ["Tab. Sumatriptan 50mg (at onset)", "Tab. Domperidone 10mg (for nausea)", "Rest in dark quiet room; cold compress"],
    "tension headache":         ["Tab. Ibuprofen 400mg or Paracetamol 1g", "Drink plenty of water", "Neck/shoulder massage & rest"],
    "epilepsy":                 ["Tab. Sodium valproate/Levetiracetam (as prescribed)", "⚠️ Never stop medication suddenly", "⚠️ Neurologist evaluation needed"],
    "stroke":                   ["⚠️ Emergency — call ambulance immediately (FAST test)", "⚠️ Every minute matters — go to hospital", "Thrombolysis if within 4.5 hrs (hospital)"],
    "transient ischemic attack":["⚠️ Hospital evaluation urgently (mini-stroke)", "Tab. Aspirin 300mg immediately", "⚠️ Neurologist needed same day"],
    "parkinson disease":        ["Tab. Levodopa-Carbidopa (as prescribed)", "Physiotherapy & exercise", "⚠️ Neurologist follow-up needed"],
    "alzheimer disease":        ["Tab. Donepezil 5mg (as prescribed)", "Cognitive exercises & social engagement", "⚠️ Neurologist follow-up needed"],
    "multiple sclerosis":       ["Interferon beta/Glatiramer (as prescribed)", "Tab. Baclofen (for spasticity)", "⚠️ Neurologist needed"],
    "meningitis":               ["⚠️ Emergency — call ambulance immediately", "IV antibiotics (hospital)", "Do not delay treatment"],
    "encephalitis":             ["⚠️ Hospital admission urgently", "Tab. Acyclovir (if viral, hospital)", "Supportive care"],
    "guillain barre syndrome":  ["⚠️ Hospital admission needed", "IVIG or plasmapheresis (hospital)", "Physiotherapy for recovery"],
    "amyotrophic lateral sclerosis (als)": ["Tab. Riluzole (as prescribed)", "Physiotherapy & speech therapy", "⚠️ Neurologist needed urgently"],
    "myasthenia gravis":        ["Tab. Pyridostigmine (as prescribed)", "Tab. Prednisolone (immunosuppression)", "⚠️ Neurologist needed"],
    "essential tremor":         ["Tab. Propranolol 40mg (daily)", "Tab. Primidone (alternative)", "⚠️ Neurologist evaluation"],
    "bell palsy":               ["Tab. Prednisolone 60mg (tapering over 10 days)", "Eye lubricant drops (if eye not closing)", "Physiotherapy for face muscles"],
    "trigeminal neuralgia":     ["Tab. Carbamazepine 200mg (twice daily)", "Tab. Gabapentin 300mg (alternative)", "⚠️ Neurologist needed"],
    "sciatica":                 ["Tab. Etoshine 90mg (daily)", "Hot/cold compress on lower back", "Physiotherapy & stretching exercises"],
    "carpal tunnel syndrome":   ["Wrist splint at night", "Tab. Ibuprofen 400mg for pain", "Physiotherapy; surgery if severe"],
    "restless leg syndrome":    ["Tab. Ropinirole (as prescribed)", "Iron supplement (if deficient)", "Avoid caffeine; warm leg baths"],
    "narcolepsy":               ["Tab. Modafinil (as prescribed)", "Scheduled short naps", "⚠️ Sleep specialist needed"],
    "neuralgia":                ["Tab. Pregabalin 75mg (twice daily)", "Tab. Amitriptyline 10mg (at night)", "Physiotherapy if needed"],
    "vertigo":                  ["Tab. Betahistine 16mg (3×/day)", "Tab. Stemetil (promethazine) for nausea", "Epley manoeuvre (with physiotherapist)"],
    "benign paroxysmal positional vertical (bppv)": ["Epley manoeuvre (physiotherapist)", "Tab. Betahistine 16mg (3×/day)", "Avoid sudden head movements"],
    "concussion":               ["Rest (physical & cognitive) for 48 hrs", "Tab. Paracetamol 500mg (no NSAIDs)", "⚠️ Doctor review if symptoms worsen"],
    "cerebral edema":           ["⚠️ Emergency — hospital immediately", "IV Mannitol (hospital)", "Treat underlying cause"],
    "intracerebral hemorrhage": ["⚠️ Emergency — hospital immediately", "BP control (hospital)", "Neurosurgery if indicated"],
    "subarachnoid hemorrhage":  ["⚠️ Emergency — call ambulance immediately", "Nimodipine (hospital)", "Neurosurgery needed"],
    "subdural hemorrhage":      ["⚠️ Emergency — hospital immediately", "Surgery if indicated", "CT scan urgently needed"],
    "hydrocephalus":            ["⚠️ Neurosurgical evaluation needed", "VP shunt placement (surgery)", "Regular monitoring"],
    "muscular dystrophy":       ["Physiotherapy & stretching", "Tab. Prednisolone (as prescribed)", "⚠️ Neurologist follow-up needed"],
    "myositis":                 ["Tab. Prednisolone (as prescribed)", "Physiotherapy", "⚠️ Rheumatologist needed"],

    # Mental Health
    "anxiety":                  ["Tab. Escitalopram 10mg (as prescribed)", "Breathing exercises & meditation", "Cognitive behavioural therapy (CBT)"],
    "depression":               ["Tab. Sertraline 50mg (as prescribed)", "Regular exercise & social engagement", "Psychotherapy & counselling"],
    "panic disorder":           ["Tab. Escitalopram 10mg (as prescribed)", "Tab. Clonazepam 0.5mg (short term)", "CBT therapy; breathing exercises"],
    "bipolar disorder":         ["Tab. Lithium (as prescribed)", "Tab. Valproate (as prescribed)", "⚠️ Psychiatrist evaluation needed"],
    "schizophrenia":            ["Risperidone/Olanzapine (as prescribed)", "⚠️ Psychiatrist evaluation needed", "Social support & rehabilitation"],
    "post-traumatic stress disorder (ptsd)": ["Tab. Sertraline (as prescribed)", "EMDR therapy", "Trauma-focused CBT"],
    "obsessive compulsive disorder (ocd)": ["Tab. Fluvoxamine (as prescribed)", "CBT with ERP therapy", "⚠️ Psychiatrist evaluation needed"],
    "insomnia":                 ["Tab. Melatonin 3-5mg (before bed)", "Sleep hygiene (fixed schedule, dark room)", "Avoid screens 1 hr before bed"],
    "primary insomnia":         ["Tab. Melatonin 5mg (at bedtime)", "CBT for insomnia (CBT-I)", "Avoid caffeine after 2pm"],
    "eating disorder":          ["⚠️ Psychiatrist & dietitian needed", "Nutritional rehabilitation", "Psychotherapy support"],
    "attention deficit hyperactivity disorder (adhd)": ["Tab. Methylphenidate (as prescribed)", "Behavioural therapy", "⚠️ Psychiatrist evaluation needed"],
    "autism":                   ["Behavioural therapy (ABA)", "Speech & language therapy", "⚠️ Paediatric specialist needed"],
    "dementia":                 ["Tab. Donepezil 5mg (as prescribed)", "Cognitive exercises & social engagement", "⚠️ Neurologist/psychiatrist needed"],
    "alcohol abuse":            ["Tab. Naltrexone/Disulfiram (as prescribed)", "De-addiction counselling", "⚠️ Psychiatrist evaluation needed"],
    "drug abuse":               ["De-addiction programme", "Naltrexone (as prescribed)", "⚠️ Psychiatrist evaluation needed"],
    "delirium":                 ["⚠️ Hospital evaluation needed", "Treat underlying cause", "Haloperidol (hospital, if agitated)"],
    "adjustment reaction":      ["Counselling & psychotherapy", "Tab. Escitalopram (if prescribed)", "Social support network"],
    "dysthymic disorder":       ["Tab. Sertraline 50mg (as prescribed)", "Psychotherapy (CBT)", "Regular exercise & social contact"],
    "social phobia":            ["Tab. Sertraline (as prescribed)", "CBT for social anxiety", "Gradual exposure therapy"],
    "conversion disorder":      ["Psychotherapy (CBT)", "Physiotherapy", "⚠️ Neurologist + psychiatrist needed"],
    "somatization disorder":    ["Psychotherapy (CBT)", "Tab. Amitriptyline (for pain)", "Regular GP review"],

    # Musculoskeletal
    "arthritis":                ["Tab. Etoricoxib 90mg (daily)", "Physiotherapy & gentle exercise", "Omega-3 supplements & weight management"],
    "osteoarthritis":           ["Tab. Etoricoxib 90mg (daily)", "Physiotherapy & weight management", "Glucosamine + Chondroitin supplement"],
    "rheumatoid arthritis":     ["Tab. Methotrexate (as prescribed)", "Tab. Hydroxychloroquine (as prescribed)", "⚠️ Rheumatologist needed"],
    "gout":                     ["Tab. Colchicine 0.5mg (during attack)", "Tab. Allopurinol 100mg (prevention)", "Avoid red meat, alcohol & purine-rich foods"],
    "ankylosing spondylitis":   ["Tab. Etoricoxib 90mg (daily)", "Physiotherapy & stretching (essential)", "⚠️ Rheumatologist needed"],
    "fibromyalgia":             ["Tab. Amitriptyline 10mg (at night)", "Regular gentle exercise (swimming, walking)", "Tab. Pregabalin 75mg (as prescribed)"],
    "osteoporosis":             ["Tab. Calcium 500mg + Vitamin D3 (daily)", "Tab. Alendronate 70mg (weekly)", "Weight-bearing exercises"],
    "chronic back pain":        ["Tab. Etoshine 90mg (daily)", "Physiotherapy & posture correction", "Hot compress & regular stretching"],
    "lumbago":                  ["Tab. Ibuprofen 400mg (3×/day with food)", "Hot compress on lower back", "Bed rest (short) + gentle walking"],
    "spondylosis":              ["Tab. Etoricoxib 90mg (daily)", "Physiotherapy & neck/back exercises", "Posture correction & ergonomic chair"],
    "spondylitis":              ["Tab. Etoricoxib 90mg (daily)", "Physiotherapy (essential)", "⚠️ Rheumatologist needed"],
    "herniated disk":           ["Tab. Ibuprofen 400mg + Tab. Muscle relaxant (Tizanidine)", "Physiotherapy", "⚠️ Orthopaedic review if severe"],
    "degenerative disc disease":["Tab. Etoricoxib 90mg (daily)", "Physiotherapy & core strengthening", "Weight management"],
    "spinal stenosis":          ["Tab. Pregabalin 75mg (for nerve pain)", "Physiotherapy", "⚠️ Orthopaedic/neurosurgeon if severe"],
    "carpal tunnel syndrome":   ["Wrist splint at night", "Tab. Ibuprofen 400mg for pain", "Physiotherapy; surgery if severe"],
    "bursitis":                 ["Tab. Ibuprofen 400mg (3×/day with food)", "Cold compress (acute) then warm", "Rest the affected joint"],
    "tendinitis":               ["Tab. Ibuprofen 400mg (3×/day)", "RICE (Rest, Ice, Compress, Elevate)", "Physiotherapy & eccentric exercises"],
    "lateral epicondylitis (tennis elbow)": ["Tab. Ibuprofen 400mg (3×/day)", "Tennis elbow strap/brace", "Physiotherapy & stretching"],
    "plantar fasciitis":        ["Tab. Ibuprofen 400mg (with food)", "Heel cushion insoles", "Stretching exercises (morning routine)"],
    "rotator cuff injury":      ["Tab. Etoricoxib 90mg (daily)", "Physiotherapy (essential)", "RICE protocol; surgery if severe"],
    "muscle spasm":             ["Tab. Tizanidine 2mg (as needed)", "Warm compress on affected area", "Stretching & hydration"],
    "polymyalgia rheumatica":   ["Tab. Prednisolone 15mg (as prescribed)", "Regular ESR/CRP monitoring", "⚠️ Rheumatologist needed"],
    "bone spur of the calcaneous":["Heel cushion insoles", "Tab. Ibuprofen 400mg (with food)", "Physiotherapy & stretching"],
    "knee ligament or meniscus tear": ["RICE protocol (Rest, Ice, Compress, Elevate)", "Tab. Etoricoxib 90mg (daily)", "⚠️ Orthopaedic evaluation needed"],
    "chronic knee pain":        ["Tab. Etoricoxib 90mg (daily)", "Physiotherapy & quadriceps strengthening", "Weight management"],
    "sprain or strain":         ["RICE protocol (Rest, Ice, Compress, Elevate)", "Tab. Ibuprofen 400mg (with food)", "Physiotherapy after 48 hrs"],
    "avascular necrosis":       ["⚠️ Orthopaedic evaluation needed", "Pain management (Tab. Etoricoxib)", "Surgery may be needed"],
    "scoliosis":                ["Physiotherapy & exercise", "Back brace (if prescribed)", "⚠️ Orthopaedic review needed"],
    "torticollis":              ["Tab. Diazepam 5mg (short term)", "Neck physiotherapy", "Heat therapy on neck muscles"],
    "trigger finger (finger disorder)": ["Corticosteroid injection (doctor)", "Splinting", "Surgery if not responding"],
    "de quervain disease":      ["Thumb spica splint", "Tab. Ibuprofen 400mg (with food)", "Corticosteroid injection (doctor)"],
    "adhesive capsulitis of the shoulder": ["Physiotherapy (essential)", "Tab. Ibuprofen 400mg", "Corticosteroid injection (doctor)"],

    # Skin
    "acne":                     ["Clindamycin gel (topical, twice daily)", "Tab. Doxycycline 100mg (if severe)", "Salicylic acid face wash (daily)"],
    "eczema":                   ["Hydrocortisone cream (topical)", "Moisturiser (Cetaphil/Vaseline) 3×/day", "Avoid soap; use mild cleanser"],
    "psoriasis":                ["Clobetasol cream (topical)", "Moisturiser (Vaseline) regularly", "⚠️ Dermatologist evaluation needed"],
    "contact dermatitis":       ["Hydrocortisone cream (topical)", "Tab. Cetrizine 10mg (at night)", "Identify & avoid the allergen/irritant"],
    "rosacea":                  ["Metronidazole gel (topical, twice daily)", "Broad-spectrum SPF 50 sunscreen daily", "Avoid alcohol, spicy food & sun"],
    "urticaria":                ["Tab. Cetrizine 10mg (daily)", "Tab. Ranitidine 150mg (additional antihistamine)", "Identify and avoid triggers"],
    "fungal infection of the skin": ["Clotrimazole cream (topical, 3×/day)", "Keep area dry & clean", "Tab. Fluconazole 150mg (if widespread)"],
    "athlete's foot":           ["Clotrimazole/Terbinafine cream (3×/day for 4 weeks)", "Keep feet dry; change socks daily", "Antifungal powder in shoes"],
    "onychomycosis":            ["Tab. Terbinafine 250mg (6-12 weeks)", "Antifungal nail lacquer", "Keep nails short & dry"],
    "scabies":                  ["Permethrin 5% cream (applied all over, washed off after 8 hrs)", "Repeat after 7 days", "Wash all clothes/bedding in hot water"],
    "lice":                     ["Permethrin 1% lotion (applied to hair)", "Fine-tooth comb to remove nits", "Wash bedding & clothing in hot water"],
    "impetigo":                 ["Mupirocin cream (topical, 3×/day for 7 days)", "Tab. Cefalexin 500mg (if widespread)", "Keep area clean; cover with dressing"],
    "cellulitis":               ["Tab. Cefalexin 500mg (4×/day for 7 days)", "Elevate the affected limb", "⚠️ Hospital if spreading rapidly"],
    "shingles (herpes zoster)": ["Tab. Acyclovir 800mg 5×/day (7 days)", "Calamine lotion on rash", "Tab. Pregabalin 75mg for nerve pain"],
    "cold sore":                ["Acyclovir cream (topical, 5×/day)", "Tab. Acyclovir 400mg (if frequent)", "Avoid touching lesion; wash hands"],
    "warts":                    ["Salicylic acid solution (daily)", "Cryotherapy (doctor)", "Imiquimod cream (as prescribed)"],
    "molluscum contagiosum":    ["Tretinoin cream (topical)", "Cryotherapy (doctor)", "Self-limiting — resolves in 6-12 months"],
    "seborrheic dermatitis":    ["Ketoconazole shampoo (twice weekly)", "Hydrocortisone cream (on face)", "Zinc pyrithione shampoo (daily)"],
    "dermatitis due to sun exposure": ["Cool compress immediately", "SPF 50 sunscreen daily", "Hydrocortisone cream (if inflamed)"],
    "actinic keratosis":        ["⚠️ Dermatologist evaluation needed", "5-Fluorouracil cream (as prescribed)", "Strict sun protection"],
    "melanoma":                 ["⚠️ Dermatologist urgently (any changing mole)", "Surgical excision (specialist)", "Oncologist referral"],
    "skin cancer":              ["⚠️ Dermatologist evaluation urgently", "Surgical treatment (specialist)", "Sun protection always"],
    "psoriasis":                ["Clobetasol cream (topical)", "Coal tar preparation", "⚠️ Dermatologist needed for systemic cases"],
    "lichen planus":            ["Clobetasol cream (topical)", "Tab. Prednisolone (if widespread)", "⚠️ Dermatologist evaluation"],
    "vitiligo":                 ["Tacrolimus 0.1% cream (topical)", "PUVA therapy (specialist)", "⚠️ Dermatologist evaluation needed"],
    "alopecia":                 ["Minoxidil 2-5% (topical daily)", "Tab. Finasteride (if prescribed)", "⚠️ Dermatologist evaluation"],
    "hidradenitis suppurativa": ["Tab. Clindamycin 300mg (twice daily)", "Tab. Rifampicin 300mg (as prescribed)", "⚠️ Dermatologist evaluation needed"],
    "pemphigus":                ["Tab. Prednisolone 60mg (as prescribed)", "⚠️ Dermatologist/hospital needed", "Wound care & dressings"],
    "decubitus ulcer":          ["Regular repositioning (every 2 hrs)", "Wound dressing (moisture-retentive)", "High-protein diet & pressure-relief mattress"],
    "burn":                     ["Cool running water for 20 minutes immediately", "Silver sulfadiazine cream (non-face burns)", "⚠️ Hospital if large/deep/face burn"],
    "frostbite":                ["Gradually rewarm (warm water 38-42°C)", "Do not rub affected area", "⚠️ Hospital for severe cases"],
    "diaper rash":              ["Zinc oxide cream (thick layer at each change)", "Air the area regularly", "Frequent nappy changes"],
    "intertrigo (skin condition)": ["Clotrimazole cream (if fungal)", "Keep area dry (cornstarch powder)", "Hydrocortisone cream (for inflammation)"],
    "pityriasis rosea":         ["Tab. Cetrizine 10mg (for itch)", "Calamine lotion (topical)", "Resolves in 6-8 weeks; avoid hot baths"],

    # Eyes
    "conjunctivitis":           ["Chloramphenicol eye drops (4×/day for 5 days)", "Warm saline eye wash", "Avoid touching or rubbing eyes"],
    "conjunctivitis due to allergy": ["Olopatadine eye drops (twice daily)", "Tab. Cetrizine 10mg at night", "Avoid known allergens; cold compress"],
    "conjunctivitis due to bacteria": ["Chloramphenicol eye drops (4×/day for 5 days)", "Warm saline eye wash", "Wash hands frequently"],
    "dry eye of unknown cause": ["Artificial tears (carboxymethylcellulose drops) 4×/day", "Omega-3 supplements", "Reduce screen time; use humidifier"],
    "stye":                     ["Warm compress (10 min, 4×/day)", "Chloramphenicol ointment at night", "Do not squeeze the stye"],
    "chalazion":                ["Warm compress (10 min, 4×/day)", "Chloramphenicol ointment", "See doctor if not resolving in 4 weeks"],
    "glaucoma":                 ["Timolol eye drops (as prescribed)", "⚠️ Ophthalmologist review needed", "Never miss eye drops"],
    "acute glaucoma":           ["⚠️ Eye emergency — hospital immediately", "Acetazolamide (hospital)", "Laser/surgery may be needed"],
    "cataract":                 ["⚠️ Surgical treatment needed (ophthalmologist)", "Wear anti-glare glasses (interim)", "Avoid prolonged sun exposure"],
    "macular degeneration":     ["⚠️ Ophthalmologist evaluation needed", "Anti-VEGF injections (specialist)", "Lutein/Zeaxanthin supplements"],
    "retinal detachment":       ["⚠️ Eye emergency — hospital immediately", "Surgical repair (specialist)", "Do not delay — can cause blindness"],
    "uveitis":                  ["Prednisolone eye drops (as prescribed)", "Tab. Prednisolone (if systemic)", "⚠️ Ophthalmologist needed urgently"],
    "blepharitis":              ["Warm compress + lid scrub (daily)", "Chloramphenicol ointment at night", "Omega-3 supplements"],
    "floaters":                 ["⚠️ Eye doctor check if new/sudden floaters", "Usually harmless if longstanding", "⚠️ Emergency if with flashes of light"],
    "subconjunctival hemorrhage":["Resolves on its own in 1-2 weeks", "Artificial tears for comfort", "⚠️ See doctor if recurrent or after trauma"],
    "corneal abrasion":         ["Chloramphenicol ointment (4×/day)", "Lubricating eye drops", "⚠️ Eye doctor review needed"],
    "foreign body in the eye":  ["⚠️ Do not rub the eye", "Flush with clean water for 10-15 minutes", "⚠️ Eye doctor urgently if not removed"],
    "diabetic retinopathy":     ["Strict blood sugar control", "⚠️ Ophthalmologist review (annually)", "Anti-VEGF injections (if needed)"],

    # Ear, Nose, Throat
    "otitis media":             ["Tab. Amoxicillin 500mg (5-7 days)", "Tab. Paracetamol for pain", "Warm compress on ear"],
    "acute otitis media":       ["Tab. Amoxicillin 500mg (7 days)", "Tab. Ibuprofen 400mg for pain", "Warm compress on ear"],
    "chronic otitis media":     ["Ofloxacin ear drops (twice daily)", "Keep ear dry", "⚠️ ENT evaluation needed"],
    "otitis externa (swimmer's ear)": ["Ciprofloxacin + Dexamethasone ear drops (7 days)", "Keep ear dry", "Do not use cotton buds"],
    "ear wax impaction":        ["Ear softening drops (olive oil/sodium bicarbonate)", "⚠️ ENT for professional removal", "Do not use cotton buds"],
    "tinnitus of unknown cause":["Tab. Betahistine 16mg (3×/day)", "Avoid loud noise", "⚠️ ENT evaluation needed"],
    "meniere disease":          ["Tab. Betahistine 16mg (3×/day)", "Low-salt diet", "⚠️ ENT/specialist evaluation needed"],
    "sensorineural hearing loss":["⚠️ ENT evaluation needed", "Hearing aids (if indicated)", "Avoid loud noise exposure"],
    "conductive hearing loss":  ["⚠️ ENT evaluation needed", "Treat underlying cause (wax, fluid)", "Hearing aids if needed"],
    "deviated nasal septum":    ["Nasal saline spray", "Tab. Cetrizine 10mg (if allergies)", "⚠️ ENT evaluation needed for surgery"],
    "allergic rhinitis":        ["Tab. Levocetirizine 5mg (daily)", "Nasal corticosteroid spray", "Avoid allergens (dust, pollen)"],
    "seasonal allergies (hay fever)": ["Tab. Cetrizine 10mg (daily during season)", "Nasal saline spray", "Avoid outdoor allergens during pollen season"],
    "food allergy":             ["Adrenaline auto-injector (EpiPen) if anaphylaxis risk", "Tab. Cetrizine 10mg for mild reactions", "Strictly avoid trigger foods"],
    "allergy":                  ["Tab. Cetrizine 10mg (daily)", "Identify and avoid allergens", "⚠️ Allergy testing if recurrent"],

    # Urinary
    "urinary tract infection":  ["Tab. Nitrofurantoin 100mg (5 days) or Norflox-TZ", "Drink 2-3 litres water daily", "Cranberry juice; proper hygiene"],
    "pyelonephritis":           ["Tab. Ciprofloxacin 500mg (7 days)", "Plenty of fluids", "⚠️ Hospital if fever/vomiting/severe"],
    "kidney stone":             ["Tab. Diclofenac 75mg IM/Tab. Tramadol (for pain)", "Drink 3+ litres water daily", "⚠️ Urologist if stone >5mm"],
    "cystitis":                 ["Tab. Nitrofurantoin 100mg (5 days)", "Drink plenty of water", "Avoid irritants (caffeine, alcohol)"],
    "chronic kidney disease":   ["Tab. Amlodipine (for BP control)", "Low-protein, low-potassium diet", "⚠️ Nephrologist evaluation needed"],
    "acute kidney injury":      ["⚠️ Hospital admission urgently", "IV fluids (hospital)", "Treat underlying cause urgently"],
    "kidney failure":           ["⚠️ Nephrologist evaluation urgently", "Dialysis (if indicated)", "Strict diet & fluid restriction"],
    "kidney cancer":            ["⚠️ Oncologist evaluation urgently", "Surgical treatment (specialist)", "Regular follow-up"],
    "nephrotic syndrome":       ["Tab. Prednisolone (as prescribed)", "Low-sodium diet", "⚠️ Nephrologist needed"],
    "hydronephrosis":           ["⚠️ Urologist evaluation needed", "Treat underlying obstruction", "Regular monitoring with ultrasound"],
    "benign prostatic hyperplasia (bph)": ["Tab. Tamsulosin 0.4mg (at night)", "Tab. Finasteride 5mg (as prescribed)", "Avoid alcohol, caffeine & diuretics at night"],
    "prostatitis":              ["Tab. Ciprofloxacin 500mg (4 weeks)", "Tab. Tamsulosin 0.4mg (for urinary symptoms)", "Warm sitz baths"],
    "prostate cancer":          ["⚠️ Oncologist evaluation urgently", "PSA & biopsy needed (specialist)", "Specialist treatment required"],
    "urethritis":               ["Tab. Doxycycline 100mg (7 days)", "Tab. Azithromycin 1g (single dose) for chlamydia", "⚠️ STI testing recommended"],
    "incontinence":             ["Pelvic floor exercises (Kegel exercises)", "Bladder training", "⚠️ Urologist evaluation needed"],

    # Reproductive & Gynaecological
    "pelvic inflammatory disease": ["Tab. Doxycycline 100mg + Metronidazole 400mg (14 days)", "Rest & avoid sexual activity", "⚠️ Gynaecologist evaluation needed"],
    "endometriosis":            ["Tab. Mefenamic acid 500mg (for pain)", "Combined oral contraceptive pill", "⚠️ Gynaecologist evaluation needed"],
    "polycystic ovarian syndrome (pcos)": ["Tab. Metformin 500mg", "Tab. Clomiphene (if fertility needed)", "Regular exercise & weight management"],
    "uterine fibroids":         ["Tab. Mefenamic acid 500mg (for pain/heavy periods)", "Tab. Norethisterone (to regulate periods)", "⚠️ Gynaecologist evaluation needed"],
    "ovarian cyst":             ["Tab. Mefenamic acid 500mg (for pain)", "Monitoring ultrasound in 6-8 weeks", "⚠️ Gynaecologist if large/persistent"],
    "ovarian torsion":          ["⚠️ Surgical emergency — hospital immediately", "Surgery needed urgently", "Do not delay treatment"],
    "ectopic pregnancy":        ["⚠️ Medical emergency — hospital immediately", "Surgery or Methotrexate (hospital)", "Immediate evaluation needed"],
    "vaginal yeast infection":  ["Clotrimazole pessary 500mg (single dose)", "Fluconazole 150mg Tab (single dose)", "Avoid tight clothing; wear cotton"],
    "vaginitis":                ["Metronidazole 400mg (5 days) for BV", "Clotrimazole cream (for yeast)", "⚠️ Swab test to confirm type"],
    "atrophic vaginitis":       ["Vaginal oestrogen cream (as prescribed)", "Lubricant gel for dryness", "⚠️ Gynaecologist evaluation needed"],
    "cervicitis":               ["Tab. Azithromycin 1g (single dose)", "Tab. Doxycycline 100mg (7 days)", "⚠️ STI testing recommended"],
    "menopause":                ["Hormone replacement therapy (as prescribed)", "Calcium 500mg + Vitamin D (daily)", "⚠️ Gynaecologist evaluation"],
    "premenstrual tension syndrome": ["Tab. Mefenamic acid 500mg (premenstrually)", "Evening primrose oil supplement", "Regular exercise & reduce salt/caffeine"],
    "endometrial hyperplasia":  ["Progesterone (as prescribed)", "⚠️ Gynaecologist evaluation needed", "Regular endometrial monitoring"],
    "cervical cancer":          ["⚠️ Oncologist evaluation urgently", "HPV vaccination for prevention", "Regular Pap smears for screening"],
    "ovarian cancer":           ["⚠️ Oncologist evaluation urgently", "Surgery + chemotherapy (specialist)", "CA-125 monitoring"],
    "uterine cancer":           ["⚠️ Oncologist evaluation urgently", "Surgery (hysterectomy — specialist)", "Specialist treatment required"],
    "breast cancer":            ["⚠️ Oncologist evaluation urgently", "Surgery/chemotherapy/radiotherapy", "Regular breast self-examination"],
    "breast cyst":              ["Monitoring (usually self-resolving)", "Fine needle aspiration if large", "⚠️ Breast surgeon if persistent"],
    "fibrocystic breast disease":["Evening primrose oil supplement", "Reduce caffeine & fat intake", "Well-fitted supportive bra"],
    "mastitis":                 ["Tab. Flucloxacillin 500mg (10 days)", "Continue breastfeeding (important)", "Warm compress & frequent feeding"],

    # Sexually Transmitted Infections
    "chlamydia":                ["Tab. Azithromycin 1g (single dose)", "Tab. Doxycycline 100mg (7 days) alternative", "⚠️ Partner treatment also needed"],
    "gonorrhea":                ["Ceftriaxone 500mg injection (single dose)", "Tab. Azithromycin 1g (single dose)", "⚠️ Partner treatment also needed"],
    "syphilis":                 ["Benzathine Penicillin G injection (as prescribed)", "⚠️ Doctor evaluation needed", "⚠️ Partner treatment also needed"],
    "genital herpes":           ["Tab. Acyclovir 400mg (3×/day for 7-10 days)", "Antiviral cream (topical)", "Avoid sexual contact during outbreaks"],
    "hpv":                      ["HPV vaccination (preventive)", "⚠️ Gynaecologist for Pap smear", "Regular cervical screening"],
    "human immunodeficiency virus infection (hiv)": ["Antiretroviral therapy (ART) as prescribed", "⚠️ Infectious disease specialist needed", "Regular CD4 & viral load monitoring"],
    "trichomonas infection":    ["Tab. Metronidazole 400mg (5 days)", "⚠️ Partner treatment also needed", "Avoid alcohol during treatment"],

    # Paediatric
    "chickenpox":               ["Calamine lotion (topical)", "Tab. Acyclovir 400mg (5 days)", "Avoid scratching; cut nails short"],
    "measles":                  ["Tab. Paracetamol 500mg for fever", "Vitamin A supplements", "⚠️ Isolate; doctor evaluation needed"],
    "mumps":                    ["Tab. Paracetamol 650mg (for pain/fever)", "Cold/warm compress on swelling", "Rest & soft diet"],
    "rubella":                  ["Tab. Paracetamol 500mg for fever", "Rest & hydration", "⚠️ Avoid pregnant women"],
    "whooping cough":           ["Tab. Azithromycin 500mg (5 days)", "Tab. Dolo 650 (if fever)", "Isolate & rest"],
    "neonatal jaundice":        ["⚠️ Paediatric evaluation needed", "Phototherapy (hospital if needed)", "Adequate feeding"],
    "febrile seizure":          ["⚠️ Medical evaluation urgently", "Tab. Paracetamol (for fever control)", "⚠️ Neurologist review if recurrent"],
    "croup":                    ["Dexamethasone syrup (as prescribed)", "Cool mist humidifier", "⚠️ Paediatric review if worsening"],

    # Blood Disorders
    "anaemia":                  ["Ferrous sulphate 200mg (twice daily)", "Folic acid 5mg (daily)", "Iron-rich diet (leafy greens, meat, lentils)"],
    "sickle cell anemia":       ["Hydroxyurea (as prescribed)", "Folic acid 5mg (daily)", "⚠️ Haematologist needed"],
    "sickle cell crisis":       ["⚠️ Hospital admission needed", "IV fluids + pain management (hospital)", "Oxygen therapy"],
    "thalassemia":              ["Folic acid 5mg (daily)", "⚠️ Haematologist evaluation needed", "Regular blood transfusions if needed"],
    "hemophilia":               ["Factor replacement therapy (specialist)", "⚠️ Haematologist needed", "Avoid aspirin/NSAIDs"],
    "thrombocytopenia":         ["⚠️ Haematologist evaluation needed", "Tab. Prednisolone (as prescribed)", "Avoid trauma & NSAIDs"],
    "leukemia":                 ["⚠️ Oncologist/Haematologist urgently", "Chemotherapy (specialist)", "Regular blood monitoring"],
    "lymphoma":                 ["⚠️ Oncologist evaluation urgently", "Chemotherapy/radiotherapy (specialist)", "Supportive care"],
    "multiple myeloma":         ["⚠️ Oncologist evaluation urgently", "Chemotherapy (specialist)", "Calcium + Vitamin D supplement"],
    "polycythemia vera":        ["⚠️ Haematologist evaluation needed", "Phlebotomy (doctor)", "Tab. Aspirin 75mg (daily)"],
    "hemolytic anemia":         ["Tab. Prednisolone (as prescribed)", "Folic acid 5mg (daily)", "⚠️ Haematologist needed"],
    "aplastic anemia":          ["⚠️ Haematologist urgently", "Immunosuppressive therapy (specialist)", "Bone marrow transplant if needed"],
    "von willebrand disease":   ["⚠️ Haematologist evaluation needed", "Desmopressin (as prescribed)", "Avoid aspirin/NSAIDs"],
    "deep vein thrombosis (dvt)":["Tab. Rivaroxaban (as prescribed)", "Compression stockings", "⚠️ Doctor evaluation needed urgently"],
    "pulmonary embolism":       ["⚠️ Emergency — hospital immediately", "Anticoagulation (hospital)", "Do not delay treatment"],

    # Other Common Conditions
    "covid-19":                 ["Tab. Dolo 650 (3× daily)", "Zincovit multivitamin (daily)", "⚠️ Isolate & consult clinic"],
    "allergy to animals":       ["Tab. Cetrizine 10mg (daily)", "Nasal corticosteroid spray", "Minimise animal contact; HEPA filter"],
    "heat exhaustion":          ["Move to cool area immediately", "ORS or cool water to drink", "Cool wet towels on skin"],
    "heat stroke":              ["⚠️ Medical emergency — call ambulance", "Cool patient immediately (ice packs to armpits/groin)", "IV fluids (hospital)"],
    "hypothermia":              ["⚠️ Medical emergency — hospital immediately", "Remove wet clothing; warm blankets", "Warm fluids if conscious"],
    "carbon monoxide poisoning":["⚠️ Emergency — get to fresh air immediately", "Call ambulance (000/108)", "100% oxygen therapy (hospital)"],
    "insect bite":              ["Tab. Cetrizine 10mg (for allergy)", "Calamine lotion or hydrocortisone cream", "⚠️ EpiPen if anaphylaxis risk"],
    "envenomation from spider or animal bite": ["⚠️ Hospital immediately", "Antivenom (hospital)", "Keep limb still and below heart level"],
    "foreign body in the throat":["⚠️ Heimlich manoeuvre if choking", "⚠️ Emergency if airway blocked", "⚠️ ENT urgently if lodged"],
    "foreign body in the nose":  ["Blow nose gently (one nostril at a time)", "⚠️ ENT doctor if not removed", "Do not probe with instruments"],
    "foreign body in the ear":  ["⚠️ ENT doctor — do not probe", "Tilt head, gravity may help", "Do not use cotton buds to remove"],
    "drug reaction":            ["Stop the suspected medication immediately", "Tab. Cetrizine 10mg (for rash/itch)", "⚠️ Doctor evaluation needed"],
    "drug poisoning due to medication": ["⚠️ Poison control / hospital immediately", "Activated charcoal (hospital)", "Do not induce vomiting unless told to"],
    "alcohol intoxication":     ["Ensure airway is clear (recovery position)", "Plenty of water/ORS", "⚠️ Hospital if unconscious or seizure"],
    "alcohol withdrawal":       ["⚠️ Hospital evaluation needed", "Tab. Diazepam (hospital supervised)", "Thiamine 100mg (IV, hospital)"],
    "dehydration":              ["ORS sachets every 15-20 min", "Drink 200ml water/15 min", "Avoid sports drinks for children"],
    "hypothyroidism":           ["Tab. Levothyroxine (as prescribed)", "Take on empty stomach in morning", "Regular TSH blood monitoring"],
    "pain after an operation":  ["Tab. Paracetamol 1g (4×/day)", "Tab. Ibuprofen 400mg with food (if not contraindicated)", "Follow surgeon's specific instructions"],
    "lupus":                    ["Tab. Hydroxychloroquine 200mg (as prescribed)", "Tab. Prednisolone (for flares)", "⚠️ Rheumatologist needed"],
    "systemic lupus erythematosis (sle)": ["Tab. Hydroxychloroquine 200mg (daily)", "Tab. Prednisolone (as prescribed)", "⚠️ Rheumatologist needed"],
    "scleroderma":              ["Moisturisers for skin", "Tab. Nifedipine (for Raynaud's)", "⚠️ Rheumatologist evaluation needed"],
    "sjogren syndrome":         ["Artificial tears (eye drops)", "Saline nasal spray", "⚠️ Rheumatologist evaluation needed"],
    "amyloidosis":              ["⚠️ Specialist evaluation needed", "Treat underlying cause", "Supportive care as directed"],
    "reactive arthritis":       ["Tab. Ibuprofen 400mg (3×/day)", "Tab. Sulfasalazine (as prescribed)", "⚠️ Rheumatologist if persistent"],
    "lyme disease":             ["Tab. Doxycycline 100mg (21 days)", "⚠️ Doctor confirmation needed", "Rest & hydration"],
    "septic arthritis":         ["⚠️ Hospital admission needed", "IV antibiotics (hospital)", "Joint aspiration (hospital)"],
    "gum disease":              ["Chlorhexidine mouthwash (twice daily)", "Improved brushing & flossing", "⚠️ Dentist evaluation needed"],
    "dental caries":            ["Topical fluoride application (dentist)", "Fluoride toothpaste (twice daily)", "⚠️ Dentist evaluation needed"],
    "tooth abscess":            ["Tab. Amoxicillin 500mg (5 days)", "Tab. Ibuprofen 400mg (for pain)", "⚠️ Dentist urgently — drainage needed"],
    "aphthous ulcer":           ["Bonjela gel (topical)", "Vitamin B12 + Iron + Folate (if deficient)", "Avoid acidic/spicy foods"],
    "oral thrush (yeast infection)": ["Nystatin oral drops/gel (4×/day for 7 days)", "Tab. Fluconazole 150mg (if severe)", "Rinse mouth after steroid inhaler use"],
    "lymphadenitis":            ["Tab. Amoxicillin 500mg (if bacterial)", "Warm compress on lymph node", "⚠️ Doctor evaluation if persisting >2 weeks"],
    "thyroid cancer":           ["⚠️ Oncologist/endocrinologist urgently", "Surgery (thyroidectomy — specialist)", "Radioiodine therapy (specialist)"],
    "pancreatic cancer":        ["⚠️ Oncologist evaluation urgently", "Surgery/chemotherapy (specialist)", "Nutritional support"],
    "bladder cancer":           ["⚠️ Urologist/oncologist urgently", "Cystoscopy + biopsy (specialist)", "Surgery/BCG therapy (specialist)"],
    "kidney cancer":            ["⚠️ Oncologist evaluation urgently", "Surgery (nephrectomy)", "Regular follow-up needed"],
    "brain cancer":             ["⚠️ Neurosurgeon/oncologist urgently", "Surgery + radiotherapy (specialist)", "Supportive & palliative care"],
    "bone cancer":              ["⚠️ Oncologist evaluation urgently", "Surgery + chemotherapy (specialist)", "Pain management"],
    "testicular cancer":        ["⚠️ Urologist/oncologist urgently", "Surgery (orchiectomy)", "Chemotherapy if needed (specialist)"],
    "diabetes insipidus":       ["Tab. Desmopressin (as prescribed)", "Adequate water intake", "⚠️ Endocrinologist needed"],
    "hyperthyroidism":          ["Tab. Carbimazole (as prescribed)", "Tab. Propranolol 20mg (for symptoms)", "⚠️ Endocrinologist needed"],
    "addison disease":          ["Tab. Hydrocortisone (as prescribed)", "Tab. Fludrocortisone (as prescribed)", "⚠️ Endocrinologist needed"],
    "acromegaly":               ["⚠️ Endocrinologist evaluation needed", "Surgery/Octreotide (specialist)", "Regular GH monitoring"],
    "pituitary adenoma":        ["⚠️ Neurosurgeon/endocrinologist needed", "Cabergoline/Bromocriptine (if prolactinoma)", "MRI monitoring"],
    "diabetes mellitus type 2": ["Metformin 500mg (as prescribed)", "Monitor blood sugar daily", "Low-sugar, high-fibre diet & exercise"],
    "hyperosmotic hyperketotic state": ["⚠️ Emergency — hospital immediately", "IV fluids + insulin (hospital)", "Monitor electrolytes (hospital)"],
    "hyperopia":                ["Corrective glasses or contact lenses", "⚠️ Optometrist evaluation needed", "Regular eye check-ups"],
    "myopia":                   ["Corrective glasses or contact lenses", "⚠️ Optometrist evaluation", "Limit screen time; outdoor activities"],
    "astigmatism":              ["Corrective glasses or toric contact lenses", "⚠️ Optometrist evaluation needed", "LASIK surgery option"],
    "presbyopia":               ["Reading glasses (+1 to +3 lenses)", "⚠️ Optometrist evaluation needed", "Good lighting for reading"],
    "flat feet":                ["Arch-support insoles/orthotics", "Physiotherapy & foot strengthening", "Supportive footwear"],
    "bunion":                   ["Wide-fitting footwear", "Bunion pad/splint", "⚠️ Podiatrist/orthopaedic if severe"],
    "plantar fasciitis":        ["Tab. Ibuprofen 400mg (with food)", "Heel cushion insoles", "Stretching exercises (morning routine)"],
    "hammer toe":               ["Comfortable wide shoes", "Toe pad/splint", "⚠️ Podiatrist evaluation if severe"],
    "ingrown toe nail":         ["Soak foot in warm water (20 min, 3×/day)", "Gently lift nail edge", "⚠️ Doctor if infected (Tab. Cefalexin 500mg)"],
    "callus":                   ["Salicylic acid plaster (daily)", "Pumice stone after soaking", "Proper fitting footwear"],
    "cyst":                     ["Warm compress (may help drainage)", "⚠️ Doctor for aspiration if large", "Antibiotics if infected"],
    "lipoma":                   ["Usually harmless — no treatment needed", "⚠️ Doctor evaluation if growing rapidly", "Surgical excision if bothersome"],
    "ganglion cyst":            ["Usually self-resolving", "Wrist splint to reduce irritation", "⚠️ Doctor aspiration or surgery if persisting"],
    "scar":                     ["Silicone gel sheet (daily use for 3 months)", "SPF 50 sunscreen on scar", "Vitamin E oil (topical massage)"],
    "keloid":                   ["Silicone gel sheet (daily)", "Corticosteroid injection (doctor)", "⚠️ Dermatologist evaluation needed"],
}


def _medicine_suggestions(disease: str) -> list[str]:
    key = disease.lower().strip()

    # 1 — exact match
    if key in _MED_DB:
        return _MED_DB[key]

    # 2 — key is contained in a DB entry or DB entry is contained in key
    for db_key, meds in _MED_DB.items():
        if db_key in key or key in db_key:
            return meds

    # 3 — word overlap (at least 2 significant words match)
    key_words = set(key.split()) - {"the", "of", "a", "an", "and", "or", "due", "to"}
    best_match = None
    best_count = 1  # require at least 2 matching words
    for db_key, meds in _MED_DB.items():
        db_words = set(db_key.split()) - {"the", "of", "a", "an", "and", "or", "due", "to"}
        overlap = len(key_words & db_words)
        if overlap > best_count:
            best_count = overlap
            best_match = meds

    if best_match:
        return best_match

    # 4 — category-based fallback (smarter than generic message)
    if any(w in key for w in ["cancer", "carcinoma", "malignant", "malignancy", "tumor", "tumour", "sarcoma", "lymphoma", "leukemia"]):
        return ["⚠️ Oncologist evaluation urgently needed", "Specialist treatment required", "Supportive and palliative care"]
    if any(w in key for w in ["fracture", "dislocation", "injury", "trauma", "wound", "broken"]):
        return ["⚠️ Emergency/orthopaedic evaluation needed", "RICE protocol (Rest, Ice, Compress, Elevate)", "Tab. Ibuprofen 400mg for pain (if no contraindication)"]
    if any(w in key for w in ["poisoning", "overdose", "intoxication", "toxic"]):
        return ["⚠️ Poison control / hospital immediately", "Do not induce vomiting unless advised", "Activated charcoal (hospital, if early)"]
    if any(w in key for w in ["hemorrhage", "bleeding", "haemorrhage"]):
        return ["⚠️ Hospital evaluation urgently", "IV fluids (hospital)", "Treat underlying cause urgently"]
    if any(w in key for w in ["infection", "abscess", "cellulitis"]):
        return ["Tab. Amoxicillin-Clavulanate 625mg (as a starting antibiotic)", "⚠️ Doctor evaluation needed for correct antibiotic", "Rest & adequate hydration"]
    if any(w in key for w in ["allergy", "allergic", "hypersensitivity"]):
        return ["Tab. Cetrizine 10mg (daily)", "Identify and avoid the allergen", "⚠️ Doctor evaluation if severe or frequent"]
    if any(w in key for w in ["disorder", "syndrome", "disease", "condition"]):
        return ["⚠️ Specialist evaluation recommended", "Treat symptoms as advised by doctor", "Regular monitoring & follow-up"]

    return ["⚠️ Consult a qualified doctor for personalised treatment", "Bring a list of all symptoms to your appointment", "Do not self-medicate for undiagnosed conditions"]
