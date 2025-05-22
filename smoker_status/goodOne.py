import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv("smoker_status/train.csv")
test = pd.read_csv("smoker_status/test.csv")

# Prepare features and target
X = train.drop(columns=["id", "smoking"])
y = train["smoking"]
X_test = test.drop(columns=["id"])
test_ids = test["id"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Setup
n_folds = 5
n_models = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store predictions
oof_predictions = []
test_predictions = []

print("🚀 Training models...")
for seed in range(n_models):
    print(f"\n🔢 Model {seed}")
    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for train_idx, valid_idx in skf.split(X_scaled, y):
        X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric='auc',
            random_state=seed
        )
        model.fit(X_train, y_train)

        oof_pred[valid_idx] = model.predict_proba(X_valid)[:, 1]
        test_pred += model.predict_proba(X_test_scaled)[:, 1] / n_folds

    auc = roc_auc_score(y, oof_pred)
    print(f"✅ OOF AUC: {auc:.5f}")

    oof_predictions.append(oof_pred)
    test_predictions.append(test_pred)

    # Save to file
    pd.DataFrame({"id": train["id"], "prediction": oof_pred}).to_csv(f"oof_model_{seed}.csv", index=False)
    pd.DataFrame({"id": test_ids, "prediction": test_pred}).to_csv(f"test_model_{seed}.csv", index=False)

print("\n✅ Finished training and saving OOF/Test predictions.")

# ==============================
# 🧠 Hill Climbing Ensemble Step
# ==============================

print("\n🔁 Starting Hill Climbing Ensemble...")

def hill_climb_ensemble(oof_preds_list, y_true, test_preds_list, max_models=20):
    ensemble = np.zeros_like(oof_preds_list[0])
    test_ensemble = np.zeros_like(test_preds_list[0])
    used = []

    for _ in range(max_models):
        best_score = -np.inf
        best_idx = None
        for i, preds in enumerate(oof_preds_list):
            if i in used:
                continue
            candidate = (ensemble * len(used) + preds) / (len(used) + 1)
            score = roc_auc_score(y_true, candidate)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            break
        used.append(best_idx)
        ensemble = (ensemble * (len(used) - 1) + oof_preds_list[best_idx]) / len(used)
        test_ensemble = (test_ensemble * (len(used) - 1) + test_preds_list[best_idx]) / len(used)
        print(f"🔹 Added model {best_idx}, ensemble AUC: {best_score:.5f}")

    return test_ensemble

# Load saved OOF/Test preds
oof_preds_list = [pd.read_csv(f"oof_model_{i}.csv")["prediction"].values for i in range(n_models)]
test_preds_list = [pd.read_csv(f"test_model_{i}.csv")["prediction"].values for i in range(n_models)]

# Run ensemble
final_preds = hill_climb_ensemble(oof_preds_list, y, test_preds_list)

# Save final submission
submission = pd.DataFrame({
    "id": test_ids,
    "smoking": final_preds
})
submission.to_csv("smoker_status/hill_climb_submission.csv", index=False)

print("\n📁 Submission file saved as hill_climb_submission.csv")
