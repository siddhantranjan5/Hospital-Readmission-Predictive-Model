# ===========================================================
# HOSPITAL PATIENT READMISSION PREDICTION MODEL
# ===========================================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# 2️⃣ Load Dataset
df = pd.read_csv("E:\\PGDM\\Data analytics\\Datasets\\Hospital Patient.csv")
print("Shape:", df.shape)
print(df.head())

# 3️⃣ Clean & Normalize Column Names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
print("\nCleaned Columns:", df.columns.tolist())

# 4️⃣ Drop Irrelevant Columns
drop_cols = ['patient_id', 'discharge_date']  # adjust if needed
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# 5️⃣ Handle Missing Values
df = df.dropna()

# 6️⃣ Encode Categorical Variables
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 7️⃣ Define Features (X) and Target (y)
target_col = 'readmission'  # <-- change to 'outcome' if you want to predict outcome instead
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")

X = df.drop(target_col, axis=1)
y = df[target_col]

# 8️⃣ Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 9️⃣ Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔟 Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

# 1️⃣1️⃣ Cross-Validation Comparison
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n🔍 Evaluating Models...\n")
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    results[name] = np.mean(scores)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 1️⃣2️⃣ Select & Train Best Model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n✅ Best Model: {best_model_name}")

best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

# 1️⃣3️⃣ Evaluation
print("\n📈 Model Evaluation:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ROC-AUC (if binary)
try:
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"{best_model_name} (AUC={roc_auc_score(y_test, y_prob):.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
except:
    print("ROC curve skipped (multi-class classification).")

# 1️⃣4️⃣ Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()

# 1️⃣5️⃣ Feature Importance
if hasattr(best_model, "feature_importances_"):
    feat_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp.head(10))
    plt.title(f"Top 10 Important Features - {best_model_name}")
    plt.show()
    print("\n📊 Top Features:\n", feat_imp.head(10))

# ===========================================================
# 1️⃣6️⃣ Predict Individual Patient Readmission
# ===========================================================

# Predict outcomes and probabilities
predictions = best_model.predict(X_test_scaled)
probabilities = best_model.predict_proba(X_test_scaled)[:, 1]

# Combine into one DataFrame
predicted_df = X_test.copy()
predicted_df['actual_readmission'] = y_test
predicted_df['predicted_readmission'] = predictions
predicted_df['readmission_probability'] = probabilities

# Decode back to labels if encoded
try:
    predicted_df['actual_readmission'] = le.inverse_transform(predicted_df['actual_readmission'])
    predicted_df['predicted_readmission'] = le.inverse_transform(predicted_df['predicted_readmission'])
except:
    pass

# Save results
predicted_df.to_csv("Predicted_Patient_Readmission.csv", index=False)
print("\n✅ Predictions saved to 'Predicted_Patient_Readmission.csv'")
print(predicted_df.head())

# ===========================================================
# 1️⃣7️⃣ Summary
# ===========================================================
print("\n==============================")
print("🏥 FINAL MODEL SUMMARY")
print("==============================")
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
if 'y_prob' in locals():
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
print("==============================")
