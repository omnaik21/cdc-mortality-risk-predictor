# 🫀 CDC Mortality Risk Predictor

> **Mortality Risk Prediction Using CDC BRFSS 2023 — A Binary Classification Approach for Early Clinical Intervention**

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8256-green?style=flat-square)]()
[![Recall](https://img.shields.io/badge/Recall%20%400.40-85%25-brightgreen?style=flat-square)]()

**PGPDSE Capstone | Great Learning | 2025**
**Author:** Om M. Naik &nbsp;|&nbsp; **Mentor:** Dr. Pranita Mahajan

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [App Features](#-app-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Key EDA Findings](#-key-eda-findings)
- [Feature Engineering](#-feature-engineering)
- [Data Leakage Lesson](#-data-leakage-lesson)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Team](#-team)

---

## 🎯 Project Overview

A machine learning system trained on **433,000+ Americans** from the CDC BRFSS 2023 survey that identifies individuals at **HIGH mortality risk** using only self-reported behavioral and demographic data — no blood tests, no lab values, no clinical visit required.

Deployed as a **professional Streamlit web application** with:
- 🔐 Local authentication system (register + login with hashed passwords)
- 🤖 Groq-powered AI clinical assistant (LLaMA 3.3 70B)
- 📊 Interactive prediction dashboard with charts and analytics
- 📈 Full model performance analytics tab

---

## ❌ Problem Statement

The CDC BRFSS surveys **433,000+ living Americans** annually but records **no mortality outcome** — the survey only reaches people who are alive to answer. This creates a fundamental challenge:

> *How do we define and predict mortality risk when our dataset has no death column?*

**Our three-layer approach:**

| Layer | Approach |
|-------|----------|
| Target Engineering | Define `high_risk` from 8 validated chronic conditions as a mortality proxy |
| Feature Prediction | Train classifier using ONLY behavioral & demographic features — NOT the target conditions |
| Imbalance Handling | Manage the 35%/65% class split without distorting the real-world distribution |

**Business Impact:**
- **Healthcare providers** — automatic flagging of HIGH RISK patients before advanced illness
- **Insurance companies** — survey-based risk stratification without expensive clinical testing
- **Public health agencies** — demographic and geographic insight into mortality risk drivers
- Every **1% improvement in Recall** = ~4,330 more HIGH RISK individuals correctly identified per year

---

## ✨ App Features

| Feature | Description |
|---------|-------------|
| 🔐 Auth System | Local register + login with SHA-256 hashed passwords, stored in `users.json` |
| 👤 User Roles | Physician, Nurse, Clinical Researcher, Public Health Analyst, Medical Student |
| 🎯 Risk Prediction | XGBoost model with 30 clinical features, threshold 0.40 |
| 📊 Risk Gauge | Needle dial showing probability vs 40% threshold |
| 📊 Factor Chart | Top contributing risk factors — horizontal bar chart |
| 🕸️ Radar Chart | Patient vs average HIGH/LOW RISK population profile |
| 🤖 AI Assistant | Groq LLaMA 3.3 70B explains predictions in plain clinical language |
| 📈 ROC Curves | All 6 models compared on same held-out test set |
| 📉 Threshold Analysis | Precision vs Recall tradeoff visualisation |
| 🔢 Confusion Matrix | Interactive heatmap at threshold 0.40 |

---

## 🛠 Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.9+ |
| Web Framework | Streamlit |
| ML Model | XGBoost (tuned via RandomizedSearchCV) |
| AI Assistant | Groq API — LLaMA 3.3 70B Versatile |
| Visualisation | Plotly |
| Auth | SHA-256 hashing + local JSON storage |
| Model Serialisation | joblib |
| Environment | python-dotenv |

---

## 📊 Dataset

| Attribute | Detail |
|-----------|--------|
| Source | CDC BRFSS 2023 (LLCP2023.XPT) — SAS transport format |
| URL | https://www.cdc.gov/brfss |
| Total rows | 433,074 (after removing 249 duplicates from 433,323) |
| Original columns | 350 |
| Selected columns | 35 (1,330 MB → 124 MB — 10× memory reduction) |
| Target variable | `high_risk` — binary: 1 = HIGH RISK, 0 = LOW RISK |
| Class distribution | HIGH RISK 35.1% (151,803) \| LOW RISK 64.9% (281,271) \| Ratio 1:1.9 |
| Train / Test split | 80% train (346,459) \| 20% test (86,615) \| stratified on target |

### Column Selection Logic — Why 35 from 350?

| Reason for Dropping | Columns (~count) | Example |
|---------------------|-----------------|---------|
| Admin & Interview | ~80 cols | Interview dates, interviewer IDs, state FIPS |
| Redundant Calculated | ~95 cols | BRFSS auto-generates weighted versions of raw columns |
| Irrelevant Behaviors | ~65 cols | Seatbelt use, fruit intake, dental visits |
| Extreme Missing Values | ~75 cols | State-specific modules, 50–99% missing |

### Data Dictionary (35 Selected Columns)

**Target Components — used to BUILD `high_risk` label**

| Column | Original Code | Description |
|--------|--------------|-------------|
| `heart_attack` | CVDINFR4 | Ever told you had a heart attack |
| `heart_disease` | CVDCRHD4 | Ever told you had coronary heart disease |
| `stroke` | CVDSTRK3 | Ever told you had a stroke |
| `diabetes` | DIABETE4 | Diabetes — incl. pregnancy diagnosis |
| `kidney_disease` | CHCKDNY2 | Kidney disease |
| `copd` | CHCCOPD3 | COPD or emphysema |
| `cancer` | CHCOCNC1 | Non-skin cancer |
| `general_health` | GENHLTH | Self-rated health status (5=Poor → HIGH RISK) |

**Demographic Features (8 columns)**

| Column | Description | Values |
|--------|-------------|--------|
| `age_group` | Age group (6 tiers) | 1=18-24 … 6=65+ |
| `age` | Continuous age, top-coded at 80 | 18–80 |
| `sex` | Sex of respondent | 1=Male, 2=Female |
| `race` | Race and ethnicity | 1=White … 8=Hispanic |
| `education` | Highest education level | 1=No HS … 4=College |
| `income` | Household income bracket | 1=<$15K … 7=$200K+ |
| `marital_status` | Marital status | 1=Married … 6=Unmarried |
| `employment` | Employment status | 1=Employed … 7=Unable |

**Chronic Conditions — predictor features, NOT target components (5 columns)**

| Column | Description |
|--------|-------------|
| `high_bp` | High blood pressure (incl. pregnancy) |
| `high_cholesterol` | High cholesterol |
| `asthma` | Asthma diagnosis |
| `arthritis` | Arthritis diagnosis |
| `depression` | Depression — kept as feature, not target |

**Health Behaviors, Status, Access & Geography (14 columns)**

`bmi`, `bmi_category`, `smoking_status`, `physical_activity`, `binge_drinking`, `physical_health_days`, `mental_health_days`, `cost_barrier`, `last_checkup`, `has_doctor`, `has_insurance`, `metro_status`, `urban_status`

---

## 🔬 Methodology

This project follows the **CRISP-DM** framework across 12 steps:

```
Step 1  ✅  Problem definition — target variable from 8 clinical conditions
Step 2  ✅  Column selection — 35 from 350 with documented reasoning
Step 3  ✅  Data cleaning — 7-strategy logical imputation
Step 4  ✅  EDA — 8 charts across demographics, behavior, conditions
Step 4+ ✅  Outlier detection — IQR method on 4 continuous columns
Step 5  ✅  Feature engineering — 4 features, leakage fixed V4→V6
Step 6  ✅  Preprocessing — split → NaN safety → StandardScaler
Step 7  ✅  Logistic Regression baseline — AUC 0.8173, Accuracy 75.49%
Step 8  ✅  Random Forest, XGBoost, LightGBM, Neural Network
Step 9  ✅  Hyperparameter tuning — RandomizedSearchCV (50 iterations × 3 folds)
Step 10 ✅  SHAP model explainability — TreeExplainer
Step 11 ✅  Threshold tuning — Precision-Recall curve
Step 12 ✅  Streamlit deployment — app.py + Groq AI assistant
```

### Data Cleaning — 7-Strategy Logical Imputation

| Column | Missing % | Severity | Strategy Applied |
|--------|-----------|----------|-----------------|
| `poor_health_days` | 43.02% | HIGH — DROPPED | 43% missing + redundant with `health_burden` |
| `income` | 19.99% | MEDIUM | Group median by `education` + `employment` |
| `smoking_status` | 15.82% | MEDIUM | Group mode by `age_group` + `education` |
| `physical_activity` | 14.40% | MEDIUM | Group mode by `age_group` + `bmi_category` |
| `high_cholesterol` | 12.71% | MEDIUM | Rule-based: if `heart_disease=1` OR obese → 1, else → 0 |
| `bmi` | 9.27% | LOW | Group median by `age_group` + `sex` |
| `mental_health_days` | 5.82% | LOW | Group median by `depression` + `age_group` |
| `race` | 4.54% | LOW | Mode — KNN for race is ethically problematic |
| All diagnosis cols | < 2% | SAFE | Fill 0 — missing = never diagnosed |
| Demographics/access | < 2% | SAFE | Most frequent (mode) |

**Additional fixes applied:**
- BMI ÷ 100 (BRFSS stores as implied decimal — 2573 = 25.73)
- BMI < 10 or > 100 set to NaN
- 7/77/9/99 special codes → NaN
- 88 (zero days) → 0
- 2=No recoded to 0=No for 13 binary columns
- 249 duplicate rows removed

---

## 📈 Model Performance

All 6 models evaluated on the same held-out 20% test set (86,615 records):

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|-------|----------|---------|-----------|--------|----|
| Logistic Regression | 75.46% | 0.8170 | 0.6811 | 0.5860 | 0.6300 |
| Random Forest | 75.49% | 0.8195 | 0.6802 | 0.5898 | 0.6318 |
| XGBoost | 74.03% | 0.8244 | 0.6076 | 0.7668 | 0.6780 |
| **XGBoost Tuned ★** | **73.92%** | **0.8256** | **0.6057** | **0.7691** | **0.6777** |
| Neural Network | 73.25% | 0.8144 | 0.5983 | 0.7602 | 0.6696 |
| LightGBM | 73.96% | 0.8253 | 0.6064 | 0.7683 | 0.6778 |

**Best model: XGBoost Tuned** — highest ROC-AUC (0.8256) and Recall (76.91% at default threshold)

### Threshold Tuning Results

| Threshold | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|----|
| 0.30 | 66.16% | 51.44% | 90.81% | 65.68% |
| 0.35 | 68.76% | 53.78% | 88.13% | 66.80% |
| **0.40 ← chosen** | **70.94%** | **56.10%** | **85.12%** | **67.63%** |
| 0.45 | 72.64% | 58.34% | 81.34% | 67.94% |
| 0.50 | 73.92% | 60.57% | 76.91% | 67.77% |

**Threshold 0.40 chosen** — best balance of Recall (85.1%) and clinical utility. Missing a HIGH RISK patient is worse than a false alarm.

### Confusion Matrix at Threshold 0.40 (86,615 test records)

| | Predicted LOW | Predicted HIGH |
|--|--------------|----------------|
| **Actual LOW** | ✅ TN: 54,700 | ⚠️ FP: 1,400 |
| **Actual HIGH** | ❌ FN: 12,700 | ✅ TP: 17,800 |

### XGBoost Tuned — Hyperparameters

```python
XGBClassifier(
    n_estimators     = 500,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.6,
    colsample_bytree = 0.6,
    scale_pos_weight = ratio,   # handles class imbalance
    eval_metric      = 'logloss',
    random_state     = 42,
)
```

---

## 🔍 Key EDA Findings

### 1. Age vs Mortality Risk
Risk accelerates sharply from age 45 — nearly **tripling** between the 35–44 and 45–54 cohorts.

| Age Group | HIGH RISK % |
|-----------|------------|
| 18–24 | 6.2% |
| 25–34 | 10.5% |
| 35–44 | 16.8% |
| 45–54 | 33.6% |
| 55–64 | 48.1% |
| 65+ | **55.2%** |

### 2. Income vs Mortality Risk — 2.6× Social Gradient
Validates Marmot et al. (2008) — *The Lancet* — social determinants of health confirmed at scale.

| Income | HIGH RISK % |
|--------|------------|
| < $15K | **51.4%** |
| $15–25K | 46.8% |
| $25–35K | 42.1% |
| $35–50K | 37.5% |
| $50–100K | 31.2% |
| $100–200K | 24.8% |
| $200K+ | **19.6%** |

### 3. Sick-Quitter Bias Confirmed
Former smokers (45.7% HIGH RISK) ≈ daily smokers (46.6%) — only 0.9% difference. People quit **because** they already developed serious illness, not before.

### 4. BMI Paradox — U-Shaped Relationship
| BMI Category | HIGH RISK % |
|-------------|------------|
| Underweight (<18.5) | 36.8% |
| Normal (18.5–24.9) | **29.4%** (lowest) |
| Overweight (25–29.9) | 34.1% |
| Obese (≥30) | **42.5%** |

Underweight signals existing serious illness — justifies tree-based models over Logistic Regression.

### 5. Race & Ethnicity Findings
- **Hispanic Paradox confirmed** — 25.4% HIGH RISK despite lower avg income (social cohesion + diet)
- **Asian Non-Hispanic** — lowest at 19.1%
- **AI/AN** — highest at 39.7% (documented healthcare access barriers)

### 6. Condition Prevalence — HIGH vs LOW RISK
| Condition | HIGH RISK | LOW RISK |
|-----------|-----------|---------|
| High Blood Pressure | 63% | 28% |
| Arthritis | 51% | 18% |
| Diabetes | **54%** | **7%** (largest gap) |
| High Cholesterol | 48% | 32% |
| Depression | 31% | 12% |

### 7. Statistical Significance (Effect Size)
| Feature | Test | Effect Size | Interpretation |
|---------|------|------------|----------------|
| `comorbidity_count` | Point-biserial r | \|r\| ≈ 0.44 — Large | Strongest numeric predictor |
| `age` / `age_risk_tier` | Point-biserial r | \|r\| ≈ 0.38 — Large | Most powerful demographic |
| `high_bp` | Chi-square | Cramér's V ≈ 0.32 — Large | Most prevalent co-condition |
| `ses_score` | Point-biserial r | \|r\| ≈ 0.28 — Medium | Strong SES signal |
| `sex` | Chi-square | Cramér's V ≈ 0.003 — Small | Minimal standalone effect |

---

## ⚙️ Feature Engineering

Four new features created from domain knowledge and EDA findings.
**Critical rule: all engineered features use ONLY columns NOT in the `high_risk` target definition.**

| Feature | Formula | Why Created | Signal |
|---------|---------|-------------|--------|
| `comorbidity_count` | `high_bp + high_cholesterol + asthma + arthritis + depression` | Co-existing disease burden without leakage | 7× higher in HIGH RISK |
| `age_risk_tier` | 0=18–44 \| 1=45–64 \| 2=65+ | Encodes EDA turning points directly | 5× risk diff. tier 0→2 |
| `health_burden` | `physical_health_days + mental_health_days` (0–60) | Combined daily health impact | 2× higher in HIGH RISK |
| `ses_score` | `income + education` (range 2–11) | Captures social gradient | Inverse relationship |

---

## ⚠️ Data Leakage Lesson

One of the most important lessons in this project — documented for academic value:

```
V4: condition_count = heart_attack + heart_disease + stroke + diabetes...
    (ALL TARGET COLUMNS!)
    → 98.88% PHANTOM accuracy — model was cheating

V6: comorbidity_count = high_bp + high_cholesterol + asthma + arthritis + depression
    (5 NON-TARGET columns only)
    → 75.49% HONEST accuracy — real-world deployable
```

> A model that cheats in training achieves 99% in testing and 0% in deployment.

---

## 📁 Project Structure

```
CDC_Mortality_Risk/
├── cdc_mortality_app/              ← Streamlit app (this repo)
│   ├── app.py                      ← Main application
│   ├── auth.py                     ← Authentication module
│   ├── model.pkl                   ← Trained XGBoost model (joblib)
│   ├── requirements.txt            ← Python dependencies
│   ├── .gitignore                  ← Git exclusions
│   ├── .env                        ← API keys (NOT committed)
│   └── venv/                       ← Virtual environment (NOT committed)
│
├── Notebook/
│   └── Group_6_CDC_Mortality_Risk.ipynb  ← Full analysis (138 cells)
│
└── data/
    └── LLCP2023.XPT                ← CDC BRFSS 2023 raw data (NOT committed — 1.3 GB)
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/cdc-mortality-risk-predictor.git
cd cdc-mortality-risk-predictor
```

### 2. Create & activate virtual environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your Groq API key
Create a `.env` file in the project folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at: https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 6. Register & Login
- Click **Create Account**
- Enter your name, email, role and password
- Sign in and start predicting

---

## 📚 Literature Review

| Paper | Authors | Relevance |
|-------|---------|-----------|
| ML for Mortality and Disease Risk Prediction | Weng et al. (2017), PLOS ONE | Supports ensemble models (RF, XGBoost) over Framingham scores |
| CRISP-DM Data Mining Methodology | Wirth & Hipp (2000) | Project follows CRISP-DM throughout all 12 steps |
| Socioeconomic Determinants of Health | Marmot et al. (2008), The Lancet | Validates 2.6× income gap; motivates `ses_score` feature |
| CDC BRFSS as a Research Dataset | Pierannunzi et al. (2012), CDC | Confirms BRFSS validity — used in 7,000+ published studies |

---

## 👤 Author

**Om M. Naik**
Data Science & Machine Learning Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/omnaik21)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/omnaik21)

**Mentor:** Dr. Pranita Mahajan
**Programme:** Post Graduate Programme in Data Science & Engineering
**Institution:** Great Learning | 2025

---

## ⚠️ Disclaimer

This tool is a **screening aid**, not a clinical diagnosis. It is built from self-reported survey data and is intended to help healthcare professionals prioritise follow-up — not to replace clinical evaluation. All predictions must be reviewed by a qualified healthcare professional before any action is taken.

---

## 📄 Data Source

CDC Behavioral Risk Factor Surveillance System (BRFSS) 2023
- URL: https://www.cdc.gov/brfss
- Format: SAS transport (.XPT)
- Records: 433,323 respondents across all 50 US states
- The BRFSS dataset is publicly available and free to use for research purposes.
