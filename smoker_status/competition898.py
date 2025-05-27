# ================== Imports ==================
import os
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import wandb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ================== Initialize ==================
wandb.login(key="dfb04f00f29836b6832f472b7e052fb4861f3877")  # Replace or use environment var
wandb.init(project="smoking-prediction-xgboost", name="optuna-tuning", reinit=True)

# ================== Load Data ==================
train1 = pd.read_csv("smoker_status/train.csv")
test1 = pd.read_csv("smoker_status/test.csv")
train2 = pd.read_csv("smoker_status/train_dataset.csv")

# ================== Preprocessing ==================
def preprocess(train1, test1, train2):
    for df in [train1, test1, train2]:
        df.columns = df.columns.str.replace(' ', '_')

    missing_cols = set(train1.columns) - set(train2.columns) - {'id', 'smoking', 'is_train'}
    for col in missing_cols:
        train2[col] = np.nan

    train2 = train2[train1.drop(['id', 'smoking'], axis=1).columns.tolist() + ['smoking']]
    train1["is_train"] = train2["is_train"] = 1
    test1["is_train"] = 0
    test1["smoking"] = -1

    full_df = pd.concat([train1, train2, test1], axis=0, ignore_index=True)

    full_df['dental_caries_sq'] = full_df['dental_caries'] ** 2
    full_df['weight(kg)_sq'] = full_df['weight(kg)'] ** 2
    full_df['weightxheight'] = full_df['weight(kg)'] * full_df['height(cm)']
    full_df['ALT_sq'] = full_df['ALT'] ** 2
    full_df['hg_height'] = full_df['hemoglobin'] * full_df['height(cm)']
    full_df['Gtp_sq'] = full_df['Gtp'] ** 2
    full_df['waist_height_ratio'] = full_df['waist(cm)'] / full_df['height(cm)']

    full_df.drop(['hearing(left)', 'hearing(right)', 'Urine_protein'], axis=1, inplace=True)

    train_df = full_df[full_df["is_train"] == 1].drop(["id", "is_train"], axis=1)
    test_df = full_df[full_df["is_train"] == 0].drop(["id", "is_train", "smoking"], axis=1)

    return train_df, test_df, test1["id"]

train_df, test_df, test_ids = preprocess(train1, test1, train2)
X = train_df.drop("smoking", axis=1)
y = train_df["smoking"]

# ================== Optuna Objective ==================
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }

    aucs = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = xgb.XGBClassifier(**params, use_label_encoder=False)
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  eval_metric="auc",
                  early_stopping_rounds=50,
                  verbose=0)
        preds = model.predict_proba(X_valid)[:, 1]
        aucs.append(roc_auc_score(y_valid, preds))

    wandb.log({"mean_cv_auc": np.mean(aucs), **params})
    return np.mean(aucs)

# ================== Run Optuna ==================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_params
wandb.finish()
print("Best parameters:", best_params)

# ================== Train Final Model ==================
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_df))
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**best_params, random_state=fold, use_label_encoder=False, tree_method='hist', n_jobs=-1)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric="auc",
              early_stopping_rounds=100,
              verbose=0)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(test_df)[:, 1] / kf.n_splits

    auc = roc_auc_score(y_val, oof_preds[val_idx])
    auc_scores.append(auc)
    print(f"Fold {fold + 1} AUC: {auc:.5f}")

print(f"\nMean AUC: {np.mean(auc_scores):.5f}")

# ================== Save Submission ==================
submission = pd.DataFrame({
    "id": test_ids,
    "smoking": test_preds
})
submission.to_csv("smoking_prediction_submission_xgb_optuna.csv", index=False)
print("✅ Submission saved as smoking_prediction_submission_xgb_optuna.csv")
