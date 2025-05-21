# dataset4_mohs.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import median_absolute_error

# === Load data ===
train = pd.read_csv("mohs/train.csv")
test = pd.read_csv("mohs/test.csv")

# === Features & target ===
X = train.drop(columns=["Hardness"])
y = train["Hardness"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# === Model ===
model = Ridge()

# === Cross-validation (MedAE) ===
cv_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring="neg_median_absolute_error")
print(f"Cross-validation MedAE scores: {cv_scores}")
print(f"Mean CV MedAE: {cv_scores.mean():.4f}")

# === Train & Validate ===
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
medae_val = np.median(np.abs(y_val - y_pred))
print(f"Validation MedAE: {medae_val:.4f}")

# === Predict on test set ===
y_test_pred = model.predict(test)

# === Create submission file (generate IDs) ===
submission = pd.DataFrame({
    "id": range(len(test)),
    "Hardness": y_test_pred
})
submission.to_csv("mohs/submission.csv", index=False)
print("✅ mohs/submission.csv written.")
