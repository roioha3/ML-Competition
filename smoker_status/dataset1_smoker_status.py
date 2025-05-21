# smoker_status_auto_benchmark.py

import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# === Load data ===
train = pd.read_csv("smoker_status/train.csv")
test = pd.read_csv("smoker_status/test.csv")

X = train.drop(columns=["smoking"])
y = train["smoking"]

# === Preserve test IDs if available ===
test_ids = test["id"] if "id" in test.columns else range(len(test))
test = test.drop(columns=["id"], errors='ignore')

# === Encode and align ===
X = pd.get_dummies(X).fillna(0)
test = pd.get_dummies(test).fillna(0)
X, test = X.align(test, join="left", axis=1, fill_value=0)

# === Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === AutoML ===
automl = AutoML()
automl_settings = {
    "time_budget": 3600,  # ⏱️ 1 hour for best results
    "metric": "roc_auc",
    "task": "classification",
    "estimator_list": ["xgboost", "lgbm", "catboost", "rf", "extra_tree", "lrl1", "lrl2", "kneighbor", "svc", "histgb", "sgd"],
    "log_file_name": "flaml_smoker_status_deep.log",
    "n_jobs": -1,
    "seed": 42,
    "verbose": 3,
}

automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

# === Evaluate best model ===
y_val_proba = automl.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_proba)

print("\n🎯 TOP MODEL REPORT")
print(f"✅ Validation AUC: {val_auc:.4f}")
print(f"✅ Best Estimator: {automl.best_estimator}")
print("✅ Best Config:")
for k, v in automl.best_config.items():
    print(f"    {k}: {v}")

# === Save test predictions ===
y_test_proba = automl.predict_proba(test)[:, 1]
submission = pd.DataFrame({
    "id": test_ids,
    "smoking": y_test_proba
})
submission.to_csv("smoker_status/submission.csv", index=False)
print("\n📁 smoker_status/submission.csv written.")
