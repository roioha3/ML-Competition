# smoker_status_lgbm_final.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

# === Load Data ===
train = pd.read_csv("smoker_status/train.csv")
test = pd.read_csv("smoker_status/test.csv")

X = train.drop(columns=["smoking"])
y = train["smoking"]
test_ids = test["id"] if "id" in test.columns else range(len(test))
test = test.drop(columns=["id"], errors="ignore")

# === One-hot encode and align ===
X = pd.get_dummies(X).fillna(0)
test = pd.get_dummies(test).fillna(0)
X, test = X.align(test, join="left", axis=1, fill_value=0)

# === Train-validation split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === LightGBM with best config ===
model = LGBMClassifier(
    n_estimators=1535,
    num_leaves=7,
    min_child_samples=15,
    learning_rate=0.03510502541109822,
    max_bin=255,
    colsample_bytree=0.5171306509107687,
    reg_alpha=0.3452226770934567,
    reg_lambda=144.53202098425584,
    objective="binary",
    random_state=42,
    device="gpu"  # change to "cpu" if needed
)

# === Train & evaluate ===
model.fit(X_train, y_train)
y_val_proba = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_proba)
print(f"\n✅ LGBM Validation AUC: {val_auc:.4f}")

# === Predict on test set ===
y_test_proba = model.predict_proba(test)[:, 1]
submission = pd.DataFrame({
    "id": test_ids,
    "smoking": y_test_proba
})
submission.to_csv("smoker_status/submission.csv", index=False)
print("📁 Final LGBM-only submission written: smoker_status/submission.csv")ע
