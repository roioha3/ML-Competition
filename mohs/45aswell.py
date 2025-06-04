import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os

# Load all datasets
train1 = pd.read_csv("mohs/train.csv")
train2 = pd.read_csv("mohs/train2.csv")
test = pd.read_csv("mohs/test.csv")

# Combine train datasets
train = pd.concat([train1, train2], ignore_index=True)
train = train.reset_index(drop=True)

# Separate target and drop id
X = train.drop(columns=["id", "Hardness"])
y = train["Hardness"]
X_test = test.drop(columns=["id"])

# Split for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

# Define bins manually (based on prior performance)
bins = np.array([1.25, 2.25, 3.05, 4.05, 4.85, 5.75, 6.55, 7.75, 9.25])

# Binning function: assign each value to the closest bin
def assign_categories(data, bins):
    return np.argmin(np.abs(np.expand_dims(data, axis=1) - bins), axis=1)

# Assign bin-based categories to target values
y_train_cats = assign_categories(y_train.values, bins)
y_valid_cats = assign_categories(y_valid.values, bins)

# Train XGBClassifier
xgb_cls = XGBClassifier(random_state=42, n_estimators=300, max_depth=5)
xgb_cls.fit(X_train, y_train_cats, eval_set=[(X_valid, y_valid_cats)], verbose=0)

# Predict categories and map them back to bin values
valid_preds = bins[xgb_cls.predict(X_valid)]
medae = np.median(np.abs(y_valid.values - valid_preds))
print(f"Validation Median Absolute Error: {medae:.4f}")

# Predict on test set
test_preds = bins[xgb_cls.predict(X_test)]

# Prepare submission
submission = pd.DataFrame({
    "id": test["id"],
    "Hardness": test_preds
})
submission.to_csv("mohs/submission2.csv", index=False)
print(submission.describe())
