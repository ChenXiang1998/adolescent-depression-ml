import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import shap

# 1. Load data
data = pd.read_csv('DATA.csv')  # Ensure the file is present in the working directory

# 2. Define predictors and target
predictors = [...]  # Replace with selected feature column names
target = '...'      # Replace with the name of your target column

X = data[predictors]
y = data[target]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Build XGBoost classifier
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05
)

# 5. Train model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 7. Evaluation
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print("Classification Report:")
print(report)

# 8. Feature importance plot
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
sorted_idx = importances.argsort()
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [X.columns[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importances')
plt.tight_layout()
plt.show()

# 9. SHAP interpretation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global SHAP summary plots
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

# Optional: SHAP force plot for individual observation
# shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])
