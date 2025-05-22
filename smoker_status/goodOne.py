import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

# ==========================
# 1. Load and Preprocess Data
# ==========================
train = pd.read_csv("smoker_status/train.csv")
test = pd.read_csv("smoker_status/test.csv")

X = train.drop(columns=["id", "smoking"])
y = train["smoking"]
X_test = test.drop(columns=["id"])
test_ids = test["id"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ==========================
# 2. Train Diverse Models
# ==========================
n_folds = 5
n_models = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_predictions = []
test_predictions = []

print("🚀 Training XGBoost models with varied hyperparameters...")
for seed in range(n_models):
    print(f"\n🔢 Model {seed}")
    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    max_depth = 3 + seed % 3
    lr = 0.03 + 0.01 * (seed % 4)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBClassifier(
            n_estimators=400,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.2,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        oof_pred[valid_idx] = model.predict_proba(X_valid)[:, 1]
        test_pred += model.predict_proba(X_test_scaled)[:, 1] / n_folds

    auc = roc_auc_score(y, oof_pred)
    print(f"✅ Model {seed} OOF AUC: {auc:.5f}")

    oof_predictions.append(oof_pred)
    test_predictions.append(test_pred)

    pd.DataFrame({"id": train["id"], "prediction": oof_pred}).to_csv(f"smoker_status/oof_model_{seed}.csv", index=False)
    pd.DataFrame({"id": test_ids, "prediction": test_pred}).to_csv(f"smoker_status/test_model_{seed}.csv", index=False)

print("\n✅ Saved all OOF and test predictions.")

# ==========================
# 3. Hill Climbing Ensemble
# ==========================

def hill_climb_ensemble(oof_preds_list, y_true, test_preds_list, max_models=20):
    ensemble = np.zeros_like(oof_preds_list[0])
    test_ensemble = np.zeros_like(test_preds_list[0])
    used = []
    last_score = 0

    print("\n🔁 Starting Hill Climbing Ensembling...")

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

        if best_score <= last_score + 1e-5:
            print("🔚 No further improvement. Stopping early.")
            break

        used.append(best_idx)
        ensemble = (ensemble * (len(used) - 1) + oof_preds_list[best_idx]) / len(used)
        test_ensemble = (test_ensemble * (len(used) - 1) + test_preds_list[best_idx]) / len(used)
        last_score = best_score
        print(f"✅ Added model {best_idx}, AUC = {best_score:.5f}")

    print(f"\n🎯 Final ensemble AUC: {last_score:.5f} using {len(used)} models.")
    return test_ensemble

# Load saved predictions
oof_preds_list = [pd.read_csv(f"smoker_status/oof_model_{i}.csv")["prediction"].values for i in range(n_models)]
test_preds_list = [pd.read_csv(f"smoker_status/test_model_{i}.csv")["prediction"].values for i in range(n_models)]

# Run ensemble
final_preds = hill_climb_ensemble(oof_preds_list, y, test_preds_list)

# Save submission
submission = pd.DataFrame({
    "id": test_ids,
    "smoking": final_preds
})
submission.to_csv("smoker_status/hill_climb_submission.csv", index=False)

print("\n📁 Submission saved to smoker_status/hill_climb_submission.csv")
