import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

# =========================
# 1. LOAD DATA
# =========================
print("Loading dataset...")
df = pd.read_csv("../data/KDDTrain.txt", header=None)

# =========================
# 2. CREATE BINARY LABEL
# =========================
df['label'] = df[41]
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
df = df.drop(columns=[41, 42])  # drop attack & difficulty columns

# =========================
# 3. SEPARATE FEATURES & LABEL
# =========================
X = df.drop(columns=['label'])
y = df['label']

# One-hot encoding for categorical columns
X = pd.get_dummies(X)
X.columns = X.columns.astype(str)

print("Data shape after encoding:", X.shape)

# =========================
# 4. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. APPLY SMOTE
# =========================
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# =========================
# 6. TRAIN INITIAL XGBOOST MODEL
# =========================
print("Training XGBoost model...")
model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train_smote, y_train_smote)

# =========================
# 7. FEATURE SELECTION
# =========================
print("Selecting important features...")
selector = SelectFromModel(model, prefit=True)
X_train_selected = selector.transform(X_train_smote)
X_test_selected = selector.transform(X_test)

# Keep column names for SHAP
selected_features = X.columns[selector.get_support()]
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)

# Retrain final model on selected features
final_model = XGBClassifier(eval_metric='logloss', random_state=42)
final_model.fit(X_train_selected_df, y_train_smote)

# =========================
# 8. EVALUATE MODEL
# =========================
y_pred = final_model.predict(X_test_selected_df)
print("\nFinal Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 9. ROC CURVE
# =========================
y_prob = final_model.predict_proba(X_test_selected_df)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()
print("✅ ROC curve saved as roc_curve.png")

# =========================
# 10. SHAP EXPLAINABILITY
# =========================
print("Generating SHAP explanations...")

# Use small sample for speed
X_sample = X_train_selected_df.sample(500, random_state=42)
explainer = shap.Explainer(final_model)
shap_values = explainer(X_sample)

# Feature importance bar plot
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.title("SHAP Feature Importance")
plt.savefig("feature_importance.png")
plt.close()
print("✅ Top feature importance saved as feature_importance.png")

# SHAP summary plot
shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig("shap_summary.png")
plt.close()
print("✅ SHAP summary plot saved as shap_summary.png")

# =========================
# 11. SAVE MODEL, SELECTOR & FEATURE NAMES
# =========================
joblib.dump(final_model, "best_model.pkl")
joblib.dump(selector, "selector.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
print("✅ Model, selector, and feature columns saved successfully!")