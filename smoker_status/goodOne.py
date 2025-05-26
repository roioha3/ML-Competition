import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

train = pd.read_csv("smoker_status/train.csv")
test = pd.read_csv("smoker_status/test.csv")

X = train.drop(columns=["id", "smoking"])
y = train["smoking"]
X_test = test.drop(columns=["id"])
test_ids = test["id"]

# Add ONLY BMI
X["BMI"] = X["weight(kg)"] / ((X["height(cm)"]/100) ** 2)
X_test["BMI"] = X_test["weight(kg)"] / ((X_test["height(cm)"]/100) ** 2)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

n_folds = 5
n_models = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_predictions = []
test_predictions = []

for seed in range(n_models):
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

    oof_predictions.append(oof_pred)
    test_predictions.append(test_pred)

# Simple averaging (try both this and your hill climb, compare!)
final_preds = np.mean(test_predictions, axis=0)

submission = pd.DataFrame({
    "id": test_ids,
    "smoking": final_preds
})
submission.to_csv("smoker_status/simple_avg_submission.csv", index=False)
print("Saved to smoker_status/simple_avg_submission.csv")
