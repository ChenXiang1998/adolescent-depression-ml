**"Protective and Risk Factors in Adolescent Depression Using Machine Learning and External Validation in Two Large Population-Based Cohorts"**

---

# adolescent-depression-ml

This repository contains a generalized pipeline for implementing and interpreting machine learning models for binary classification tasks related to mental health, specifically focusing on XGBoost and SHAP. The script is designed to be flexible for use in population-based psychological research and public health studies.

> **Note**: This repository provides a template based on general methods used in a study submitted for peer review. Specific data and model parameters have been removed to preserve anonymity and comply with journal policies.

---

## 🧠 Overview

This code demonstrates a standard supervised machine learning workflow using:

* Gradient boosting trees via XGBoost
* Evaluation metrics including Accuracy and AUC
* Feature importance via model-based and SHAP interpretability

It is intended to serve as a reference for constructing reproducible, interpretable models in the context of health and behavioral research.

---

## 📂 Repository Contents

* `core_code.py`: The main script for model training, testing, evaluation, and SHAP-based interpretation.
* `README.md`: Documentation and usage instructions.

---

## 🔧 Requirements

Install dependencies using:

```bash
pip install pandas scikit-learn xgboost shap matplotlib
```

---

## 🚀 Usage Instructions

1. Place your dataset in the same directory as the script (expected filename: `DATA.csv`).
2. Define your predictors and target in the script:

   ```python
   predictors = [...]  # Fill in with your feature column names
   target = '...'      # Replace with your target column
   ```
3. Run the script:

   ```bash
   python core_code.py
   ```

---

## 📊 Model Output

The pipeline includes:

* Accuracy, AUC, and classification report on the test set.
* Bar plot of feature importances.
* SHAP summary plots for model interpretability.

---

## 📌 Citation

If you use or adapt this template, please cite the following when the paper becomes publicly available:

> Zou, H., Chen, X., et al. (2025). *Protective and Risk Factors in Adolescent Depression Using Machine Learning and External Validation in Two Large Population-Based Cohorts*. (Submitted).

---
