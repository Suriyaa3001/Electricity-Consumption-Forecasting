# ⚡ Electricity Consumption Forecasting — ML Web App

A full-stack machine learning application that forecasts hourly electricity
consumption using **Linear Regression** and **Random Forest**, served through
a modern **Streamlit** web interface.

---

## 🚀 Quick Start (3 steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run app.py
```

### Step 3 — Open in browser
Streamlit will print a local URL, typically:
```
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

> **Note:** On first launch the app will automatically:
> 1. Generate `electricity_data.csv` (17,520 rows of synthetic hourly data)
> 2. Train both models and save them to `models/`
> This takes ~30–60 seconds. Subsequent launches are instant.

---

## 📁 Project Structure

```
electricity_forecasting/
│
├── app.py                  # Main Streamlit application (UI + logic)
├── model_training.py       # ML pipeline: preprocess → train → evaluate → predict
├── generate_dataset.py     # Synthetic dataset generator
│
├── electricity_data.csv    # Auto-generated on first run
├── models/                 # Auto-created model artefacts
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   ├── best_model.pkl
│   └── metrics.pkl
│
├── requirements.txt        # Python dependencies
├── PROJECT_REPORT.md       # Full academic report + viva Q&A
└── README.md               # This file
```

---

## 🖥️ Expected UI

| Tab | What You'll See |
|-----|----------------|
| **⚡ Prediction** | Input form → amber KPI result card, smart alerts, download button |
| **📊 Model Comparison** | MAE / MSE / R² for both models, scatter plot, time-series overlay |
| **📈 Visualisations** | Daily area chart, monthly bar chart, hourly pattern, heatmap |
| **📅 Forecasts** | 24-hour, 7-day and 2-month rolling forecasts with CSV export |
| **🔍 Data Explorer** | Filter by month/hour/consumption/temperature, distribution charts, export CSV |

### Sidebar Controls
- 📅 Date picker
- 🕐 Time picker
- 🔋 Previous consumption slider (10–250 kWh)
- 🌡️ Temperature toggle + slider
- ⚠️ Alert threshold input
- ⚡ **Predict Consumption** button (amber, full-width)
- 🔄 Retrain Models button

---

## 🧠 Models Used

| Model | Library | Best For |
|-------|---------|----------|
| Linear Regression | `sklearn.linear_model` | Baseline; fast & interpretable |
| Random Forest | `sklearn.ensemble` | Non-linear patterns; typically wins |

Auto-selection: highest **R² score** on the 20% held-out test set.

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
matplotlib>=3.7.0
joblib>=1.3.0
```
*Built with Python · Scikit-learn · Streamlit · Plotly*
