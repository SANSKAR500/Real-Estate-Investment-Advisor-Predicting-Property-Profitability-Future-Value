<div align="center">

<img src="assets/banner.svg" alt="Real Estate Investment Advisor" width="100%">

<a href="https://advisorproperty.streamlit.app/">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1200&color=2EE6C5&center=true&vCenter=true&width=700&lines=Is+this+property+a+good+investment%3F;What+will+it+be+worth+in+5+years%3F;250%2C000+listings.+2+tasks.+6+models.;Try+the+live+app+below+%F0%9F%91%87" alt="Typing animation">
</a>

<br>

[![Live Demo](https://img.shields.io/badge/%F0%9F%9A%80_LIVE_DEMO-Open_the_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://advisorproperty.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Champion-189AB4?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Records](https://img.shields.io/badge/Records-250%2C000-2EE6C5?style=flat-square)

**An end-to-end machine learning system that classifies Indian residential properties as good or poor investments and forecasts their price five years ahead.**

[**Live App**](https://advisorproperty.streamlit.app/) &nbsp;·&nbsp; [Results](#-results) &nbsp;·&nbsp; [EDA](#-exploratory-data-analysis) &nbsp;·&nbsp; [Run Locally](#-run-it-locally) &nbsp;·&nbsp; [Limitations](#-honest-limitations)

</div>

---

## 📑 Table of Contents

<details open>
<summary><b>Click to collapse</b></summary>

1. [Overview](#-overview)
2. [Live Demo](#-live-demo)
3. [Key Results](#-results)
4. [Dataset](#-dataset)
5. [Exploratory Data Analysis](#-exploratory-data-analysis)
6. [How the Targets Were Built](#-how-the-targets-were-built)
7. [ML Pipeline](#-ml-pipeline)
8. [Honest Limitations](#-honest-limitations)
9. [Run It Locally](#-run-it-locally)
10. [Project Structure](#-project-structure)
11. [Future Work](#-future-work)

</details>

---

## 🎯 Overview

Buying property involves trading off price, size, locality, transit, nearby schools and hospitals, building age, and expected appreciation. This project turns those factors into two clear answers:

| Task | Question | Output |
|---|---|---|
| 🟢 **Classification** | Is this a good investment? | Recommended / Not recommended, with a confidence score |
| 📈 **Regression** | What will it be worth in 5 years? | Future price in ₹ lakhs, plus gain, ROI, and implied annual growth |

Everything is tracked with **MLflow**, packaged as reusable pipelines, and deployed as a **Streamlit** web app.

---

## 🚀 Live Demo

<div align="center">

### 👉 [**advisorproperty.streamlit.app**](https://advisorproperty.streamlit.app/) 👈

*(Free-tier apps may sleep when idle. If you see a "wake up" button, click it and wait a few seconds.)*

<img src="assets/app-advisor.jpg" alt="Property Investment Advisor input form" width="90%">

</div>

The app has four sections:

- 🔮 **Property Investment Advisor**: enter a property's details and get a verdict plus a 5-year forecast
- 📊 **Model Performance & MLflow**: benchmark tables, confusion matrix, ROC curve
- 📈 **20-Question EDA Dashboard**: every analysis plot in one place
- 📖 **Project Documentation**: how both targets were engineered

<details>
<summary><b>📸 See more screenshots</b></summary>
<br>

**Model performance page**

<img src="assets/app-performance.jpg" alt="Model performance page" width="90%">

**Methodology page**

<img src="assets/app-methodology.jpg" alt="Methodology page" width="90%">

</details>

---

## 🏆 Results

Evaluated on **50,000 held-out records**. XGBoost won both tasks.

### Classification: `Good_Investment`

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression (baseline) | 89.66% | 0.9012 | 0.8978 | 0.8995 | 0.9630 |
| Random Forest | 93.71% | 0.9312 | 0.9480 | 0.9396 | 0.9890 |
| **XGBoost (champion)** 🥇 | **97.36%** | **0.9706** | **0.9785** | **0.9745** | **0.9972** |

### Regression: `Future_Price_5Y_Dynamic`

| Model | RMSE (₹L) | MAE (₹L) | R² | MAPE |
|---|:---:|:---:|:---:|:---:|
| Ridge Regression (baseline) | 11.61 | 8.52 | 0.9971 | 5.59% |
| Random Forest | 12.69 | 9.33 | 0.9966 | 2.52% |
| **XGBoost (champion)** 🥇 | **3.26** | **2.49** | **0.9998** | **1.00%** |

<div align="center">
<img src="plots/confusion_matrix_xgboost_classifier.png" alt="Confusion matrix" width="32%">
<img src="plots/roc_curve_xgboost_classifier.png" alt="ROC curve" width="32%">
<img src="plots/regression_actual_vs_pred_xgboost_regressor.png" alt="Actual vs predicted" width="32%">
</div>

> ⚠️ **Read these numbers with context.** The targets were built by formula from the dataset's own columns, so near-perfect scores confirm the pipeline works, not that it predicts real market prices. See [Honest Limitations](#-honest-limitations).

---

## 📋 Dataset

**250,000 residential listings** across **20 states**, **42 cities**, and **500 localities**, with 22 columns.

| Group | Columns |
|---|---|
| 🏠 Structure | `Property_Type`, `BHK`, `Size_in_SqFt`, `Year_Built`, `Floor_No`, `Total_Floors`, `Age_of_Property` |
| 📍 Location | `State`, `City`, `Locality` |
| 🚇 Infrastructure | `Nearby_Schools`, `Nearby_Hospitals`, `Public_Transport_Accessibility` |
| 🏷️ Listing | `Price_in_Lakhs`, `Price_per_SqFt`, `Furnished_Status`, `Parking_Space`, `Security`, `Amenities`, `Facing`, `Owner_Type`, `Availability_Status` |

Prices range from ₹10L to ₹500L and sizes from 500 to 5,000 sq.ft.

---

## 🔍 Exploratory Data Analysis

Twenty questions across price, location, relationships, and investment factors. Every plot lives in [`plots/`](plots/) and the [notebook](notebooks/eda_analysis.ipynb).

| Finding | Detail |
|---|---|
| 💰 **Balanced prices** | Mean ₹254.6L, median ₹253.9L, no unrealistic outliers |
| 📐 **Size is a weak signal** | Size-price correlation ≈ 0.003, so location and context dominate |
| 🗺️ **Top states by ₹/sq.ft** | Karnataka (₹13,252) and Andhra Pradesh (₹13,202) |
| 🛋️ **Furnishing barely matters** | Averages differ by under ₹1 lakh |
| 🚇 **Transit drives investment quality** | Good-investment rate: **65.3%** high access, 49.3% medium, 39.9% low |

<div align="center">
<img src="plots/q20_transport_vs_investment.png" alt="Transport access vs investment rate" width="48%">
<img src="plots/q11_correlation_heatmap.png" alt="Correlation heatmap" width="48%">
</div>

---

## 🧮 How the Targets Were Built

The dataset has no "future price" or "good investment" column, so both were engineered from real estate logic.

### 📈 5-year future price

Each property gets its own annual growth rate between 5% and 13%:

```
P5 = Price_in_Lakhs × (1 + r)^5
r  = state_tier + property_type + transit + infrastructure + age + amenities
```

| Factor | Adjustment |
|---|---|
| State tier | Tier 1 hubs 8.5%, Tier 2 7.5%, others 6.8% (base) |
| Property type | Villa +1.2%, independent house +0.8% |
| Transit | High +0.8%, low −0.4% |
| Social infrastructure | Schools + hospitals ≥ 12: +0.5% |
| Building age | ≤ 7 yrs +0.5%, ≥ 25 yrs −0.6% |
| Amenities | 4 or more: +0.4% |

### ✅ Good investment (score ≥ 60 of 100)

| Pillar | Max points |
|---|:---:|
| Valuation vs locality median ₹/sq.ft | 35 |
| Appreciation potential (CAGR) | 20 |
| Connectivity & social infrastructure | 25 |
| Desirability & liquidity (BHK, parking, security, ready-to-move) | 20 |

Result: **51.55% good** (128,863) vs **48.45% not** (121,137), a balanced split with no resampling.

---

## ⚙️ ML Pipeline

```mermaid
flowchart LR
    A[📦 Raw Data<br/>250K rows] --> B[🧹 Cleaning &<br/>Imputation]
    B --> C[🛠️ Feature Engineering<br/>& Targets]
    C --> D[✂️ 80/20 Split]
    D --> E[🔧 ColumnTransformer<br/>Scale + One-Hot]
    E --> F[🟢 Classifiers<br/>LogReg · RF · XGBoost]
    E --> G[📈 Regressors<br/>Ridge · RF · XGBoost]
    F --> H[📊 MLflow Tracking]
    G --> H
    H --> I[💾 Saved Pipelines]
    I --> J[🚀 Streamlit App]
```

**Features:** 23 numeric and 11 categorical, including engineered ones such as amenity count, school and hospital density, space per BHK, floor ratio, and top-floor flags.

---

## ⚠️ Honest Limitations

Good projects state their limits. This one does:

- **Targets come from formulas.** Both labels are computed from the same input columns, and listing price is itself a feature. The models largely relearn those formulas, which is why R² is near 1.0.
- **The data looks synthetic.** Category shares are almost perfectly balanced and size barely correlates with price.
- **Assumptions, not observations.** Growth rates and score thresholds are modelling choices, not measured market data.

**What this proves:** the pipeline is sound and the models recover complex rules. **What it does not prove:** accuracy on real future prices.

---

## 💻 Run It Locally

```bash
# 1. Clone
git clone https://github.com/<your-username>/Real-Estate-Investment-Advisor-Predicting-Property-Profitability-Future-Value.git
cd Real-Estate-Investment-Advisor-Predicting-Property-Profitability-Future-Value

# 2. Install
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the full pipeline (clean, engineer, EDA, train, track)
python run_pipeline.py

# 4. Launch the app
streamlit run app.py

# 5. (Optional) Browse experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

<details>
<summary><b>Run individual stages</b></summary>

```bash
python -m src.preprocessing           # cleaning
python -m src.feature_engineering     # features + targets
python -m src.eda                     # 20-question EDA
python -m src.train_classification    # classifiers
python -m src.train_regression        # regressors
```

</details>

---

## 📁 Project Structure

```
├── app.py                      # Streamlit web app
├── run_pipeline.py             # Master pipeline orchestrator
├── requirements.txt
├── assets/                     # README banner and screenshots
├── models/                     # Saved pipelines and metadata (.joblib, .json)
├── notebooks/
│   └── eda_analysis.ipynb      # 20-question EDA notebook
├── plots/                      # EDA and evaluation charts
└── src/
    ├── preprocessing.py        # Loading, cleaning, ColumnTransformer
    ├── feature_engineering.py  # Dynamic CAGR, investment score, features
    ├── eda.py                  # EDA plots
    ├── train_classification.py # LogReg, RF, XGBoost + MLflow
    ├── train_regression.py     # Ridge, RF, XGBoost + MLflow
    └── utils.py
```

---

## 🔭 Future Work

- [ ] Train on real historical transaction data with observed price changes
- [ ] Remove leaky inputs and use time-based validation
- [ ] Add hyperparameter tuning and SHAP explanations
- [ ] Add locality comparison and rental-yield estimates

---

<div align="center">

**Built by Dazai**

If this project helped you, consider giving it a ⭐

[![Live Demo](https://img.shields.io/badge/Try_it_now-advisorproperty.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://advisorproperty.streamlit.app/)

</div>
