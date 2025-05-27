import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# === Load Data ===
train1 = pd.read_csv("smoker_status/train.csv")
train2 = pd.read_csv("smoker_status/train2.csv")
train3 = pd.read_csv("smoker_status/train_dataset.csv")
test = pd.read_csv("smoker_status/test.csv")

# === Standardize column names (if needed) ===
for df in [train1, train2, train3, test]:
    df.columns = df.columns.str.strip().str.replace(" ", "_")

# === Concatenate all training sets ===
train_all = pd.concat([train1, train2, train3], ignore_index=True).drop_duplicates()

# === Prepare Data ===
X = train_all.drop(columns=["id", "smoking"])
y = train_all["smoking"]
X_test = test.drop(columns=["id"])
test_ids = test["id"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Define model
model = GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=4,
    max_features=0.9,
    min_samples_leaf=15,
    min_samples_split=5,
    n_estimators=100,
    subsample=0.65,
    random_state=42
)

# === Cross-validated AUC ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
for train_idx, valid_idx in skf.split(X_scaled, y):
    model.fit(X_scaled[train_idx], y.iloc[train_idx])
    preds = model.predict_proba(X_scaled[valid_idx])[:, 1]
    auc = roc_auc_score(y.iloc[valid_idx], preds)
    auc_scores.append(auc)
print(f"Cross-validated AUC: {np.mean(auc_scores):.5f}")

# === Train on full data, predict on test ===
model.fit(X_scaled, y)
preds = model.predict_proba(X_test_scaled)[:, 1]

submission = pd.DataFrame({
    "id": test_ids,
    "smoking": preds
})
submission.to_csv("smoker_status/gbc_submission.csv", index=False)
print("Submission saved to smoker_status/gbc_submission.csv")
