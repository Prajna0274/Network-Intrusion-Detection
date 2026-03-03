import matplotlib
matplotlib.use('Agg')   # 🔥 VERY IMPORTANT (prevents GUI freeze)

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("../data/KDDTrain.txt", header=None)

# =========================
# 2. CREATE BINARY LABEL
# =========================
df['label'] = df[41]
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Drop original attack + difficulty column
df = df.drop(columns=[41, 42])

# =========================
# 3. SEPARATE FEATURES & LABEL
# =========================
X = df.drop(columns=['label'])
y = df['label']

# One-hot encoding
X = pd.get_dummies(X)
X.columns = X.columns.astype(str)

# =========================
# 4. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Before SMOTE:")
print(y_train.value_counts())

# =========================
# 5. APPLY SMOTE (ONLY TRAINING DATA)
# =========================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# =========================
# 6. TRAIN MODELS
# =========================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']

    results[name] = [acc, f1]

# =========================
# 7. SHOW MODEL COMPARISON
# =========================
results_df = pd.DataFrame(results, index=["Accuracy", "F1 Score"]).T
print("\n📊 Model Comparison After SMOTE:\n")
print(results_df)

# =========================
# 8. TRAIN BEST MODEL (XGBoost)
# =========================
best_model = XGBClassifier(eval_metric='logloss', random_state=42)
best_model.fit(X_train_smote, y_train_smote)

y_pred_best = best_model.predict(X_test)

print("\nClassification Report (XGBoost with SMOTE):\n")
print(classification_report(y_test, y_pred_best))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_best))

# =========================
# 9. SHAP EXPLAINABILITY
# =========================
print("\nGenerating SHAP explanations...")

# Use TreeExplainer for XGBoost (FASTER + CORRECT)
explainer = shap.TreeExplainer(best_model)

# Small sample for speed
X_sample = X_train_smote.sample(200, random_state=42)

# Get SHAP values
shap_values = explainer.shap_values(X_sample)

# =========================
# SAVE FEATURE IMPORTANCE
# =========================
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()

print("✅ SHAP summary plot saved as shap_summary.png")

# =========================
# SAVE BAR IMPORTANCE
# =========================
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig("shap_bar.png")
plt.close()

print("✅ SHAP bar plot saved as shap_bar.png")