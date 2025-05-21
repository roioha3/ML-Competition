# smoker_status_cv.py

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

# === Load data ===
train = pd.read_csv("smoker_status/train.csv")
test = pd.read_csv("smoker_status/test.csv")

# === Prepare features and labels ===
X = train.drop(columns=["smoking"])
y = train["smoking"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Classifier ===
model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# === Cross-validation using AUC ===
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
print(f"Cross-validation AUC scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.4f}")

# === Fit & Evaluate ===
model.fit(X_train, y_train)
y_proba_val = model.predict_proba(X_val)[:, 1]
auc_val = roc_auc_score(y_val, y_proba_val)
print(f"Validation AUC: {auc_val:.4f}")

# === Predict test set probabilities ===
y_test_proba = model.predict_proba(test)[:, 1]

# === Create submission file ===
submission = pd.DataFrame({
    "id": range(len(test)),         # generate sequential ids
    "smoking": y_test_proba
})
submission.to_csv("smoker_status/submission.csv", index=False)
print("✅ submission.csv written.")
