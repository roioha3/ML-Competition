import pandas as pd
from tpot import TPOTClassifier
from sklearn.preprocessing import StandardScaler

# === Load Data ===
train = pd.read_csv("smoker_status/train.csv")
test = pd.read_csv("smoker_status/test.csv")

X = train.drop(columns=["id", "smoking"])
y = train["smoking"]
X_test = test.drop(columns=["id"])
test_ids = test["id"]

# (Optional, but improves some models: scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# === Run TPOT ===
tpot = TPOTClassifier(
    generations=5,               # These control pipeline evolution, not run time
    population_size=50,
    verbosity=2,
    max_time_mins=45,            # <-- 45 minutes runtime
    random_state=42,
    scoring='roc_auc',
    n_jobs=-1
)
tpot.fit(X_scaled, y)

# === Validation AUC on training folds (optional, for reference) ===
print("Best pipeline score (training data):", tpot.score(X_scaled, y))

# === Predict on Test and Save ===
preds = tpot.predict_proba(X_test_scaled)[:, 1]
submission = pd.DataFrame({
    "id": test_ids,
    "smoking": preds
})
submission.to_csv("smoker_status/tpot_submission.csv", index=False)
print("Submission saved to smoker_status/tpot_submission.csv")

# === (Optional) Export the pipeline code TPOT found ===
tpot.export("smoker_status/tpot_best_pipeline.py")
