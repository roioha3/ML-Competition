# blueberry_mae_regression_cv.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

# === Load Data ===
train = pd.read_csv("blueberry/train.csv")
test = pd.read_csv("blueberry/test.csv")

# === Prepare Features ===
X = train.drop(columns=["yield"])
y = train["yield"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)

# === Cross-validation ===
cv_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
print(f"Cross-validation MAE scores: {cv_scores}")
print(f"Mean CV MAE: {cv_scores.mean():.4f}")

# === Fit & Evaluate ===
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
mae_val = mean_absolute_error(y_val, y_pred)
print(f"Validation MAE: {mae_val:.4f}")

# === Predict Test Set ===
X_test = test.drop(columns=["Id"]) if "Id" in test.columns else test
predictions = model.predict(X_test)

# === Create Submission ===
submission = pd.DataFrame({
    "Id": test["Id"] if "Id" in test.columns else range(len(test)),
    "Prediction": predictions
})
submission.to_csv("blueberry/submission.csv", index=False)
print("✅ submission.csv written.")
