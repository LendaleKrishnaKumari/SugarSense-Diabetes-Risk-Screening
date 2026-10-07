# SugarSense: Early Diabetes Risk Screening

## 📌 Project Overview

SugarSense is a machine learning project for **early diabetes risk screening** using clinical measurements.

The project builds and evaluates multiple supervised learning models to predict whether a patient belongs to the diabetic class (`Outcome = 1`).

The primary focus is not accuracy alone. Because missing a diabetic patient can be clinically important, the project emphasizes:

- **ROC-AUC**
- **Recall for the diabetic class**
- Precision
- F1-score
- Confusion matrix analysis
- Threshold optimization

> **Important:** SugarSense is a screening-support tool and is **not a medical diagnosis system**. It should not replace evaluation by a qualified healthcare professional.

---

## 🎯 Problem Statement

The objective is to develop a binary classification model that estimates diabetes risk from eight clinical measurements.

The dataset contains:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target:

- `Outcome = 0` → Non-diabetic
- `Outcome = 1` → Diabetic

The project uses the **Pima Indians Diabetes Database**, containing 768 patient records.

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## 🔬 Project Workflow

The project follows nine mandatory phases.

### Phase 1 — Data Audit

Performed:

- Dataset shape and data-type inspection
- Descriptive statistics
- Duplicate detection
- Class-balance analysis
- Identification of clinically impossible zero values
- Visualization of problematic values

### Phase 2 — Exploratory Data Analysis

Performed:

- Feature distributions for diabetic and non-diabetic groups
- Correlation heatmap
- Feature correlation with the target
- Identification of potentially useful and weak predictors

### Phase 3 — Feature Engineering

Created domain-informed features using row-wise operations.

The engineered features were designed to provide additional clinically meaningful information while avoiding data leakage.

### Phase 4 — Train/Test Split and Pipelines

The dataset was divided using a **stratified 80/20 train-test split** with:

```text
random_state = 42
