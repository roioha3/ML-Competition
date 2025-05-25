import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt

# === 0) Optional: Suppress Optuna INFO logs and name the study ===
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_name = "blueberry_xgb_tuning"

# === 1) Detect input folder ===
INPUT_ROOT = Path('/kaggle/input')
if INPUT_ROOT.exists():
    subs = [d for d in INPUT_ROOT.iterdir() if d.is_dir()]
    DATA_DIR = subs[0] if len(subs)==1 else next(d for d in subs if (d/'train.csv').exists())
else:
    DATA_DIR = Path('.')

# === 2) Load data & feature engineering ===
def add_features(df):
    df = df.copy()
    df['bee_total']       = df[['honeybee','bumbles','andrena','osmia']].sum(axis=1)
    df['upper_temp_diff'] = df['MaxOfUpperTRange'] - df['MinOfUpperTRange']
    df['lower_temp_diff'] = df['MaxOfLowerTRange'] - df['MinOfLowerTRange']
    df['upper_temp_mean'] = (df['MaxOfUpperTRange'] + df['MinOfUpperTRange']) / 2
    df['lower_temp_mean'] = (df['MaxOfLowerTRange'] + df['MinOfLowerTRange']) / 2
    for col in ['honeybee','bumbles','andrena','osmia']:
        df[f'{col}_ratio'] = df[col] / (df['bee_total'] + 1e-9)
    df['bee×upper_mean'] = df['bee_total'] * df['upper_temp_mean']
    df['diff_ratio']     = df['lower_temp_diff'] / (df['upper_temp_diff'] + 1e-9)
    return df

train = pd.read_csv(DATA_DIR/'train.csv')
test  = pd.read_csv(DATA_DIR/'test.csv')
train = add_features(train)
test  = add_features(test)

X_full = train.drop(columns=['id','yield'])
y_full = np.log1p(train['yield'])
X_test = test.drop(columns=['id'])

# === 3) Feature selection via XGB importance ===
dmat = xgb.DMatrix(X_full, label=y_full)
fs_model = xgb.train(
    {'objective':'reg:absoluteerror', 'eval_metric':'mae', 'tree_method':'hist', 'seed':42},
    dmat, num_boost_round=500, verbose_eval=False
)
imp = fs_model.get_score(importance_type='gain')
n_keep = max(5, int(len(imp) * 0.5))
top_feats = sorted(imp, key=imp.get, reverse=True)[:n_keep]
X_full = X_full[top_feats]
X_test = X_test[top_feats]

# === 4) Hyperparameter tuning with Optuna ===
def objective(trial):
    param = {
        'tree_method': 'hist',
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'n_estimators': 1000,
        'seed': 42,
        'verbosity': 0
    }
    reg = xgb.XGBRegressor(**param)
    score = cross_val_score(
        reg, X_full, y_full,
        cv=3, scoring='neg_mean_absolute_error'
    )
    return -score.mean()

study = optuna.create_study(
    study_name=study_name,
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=20)
best_params = study.best_params
best_xgb = xgb.XGBRegressor(
    **best_params,
    objective='reg:absoluteerror',
    eval_metric='mae',
    verbosity=0,
    tree_method='hist'
)

# === 5) Stacking ensemble ===
lgb_base = lgb.LGBMRegressor(
    objective='regression_l1', metric='mae',
    n_estimators=1500, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42
)
stack = StackingRegressor(
    estimators=[('xgb', best_xgb), ('lgb', lgb_base)],
    final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
    cv=5, passthrough=True, n_jobs=-1
)

# === 6) 5-fold OOF training & predict ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X_full))
preds = np.zeros(len(X_test))
for train_idx, val_idx in kf.split(X_full):
    X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
    y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
    stack.fit(X_tr, y_tr)
    oof[val_idx] = stack.predict(X_val)
    preds += stack.predict(X_test) / kf.n_splits

# === 7) Evaluation & output ===
oof_mae = np.mean(np.abs(np.expm1(oof) - np.expm1(y_full)))
print(f"OOF MAE: {oof_mae:.4f}")
sub = pd.DataFrame({'id': test['id'], 'yield': np.expm1(preds)})
sub.to_csv('submission.csv', index=False)
print("Saved submission.csv")

# === 8) Plot ensemble feature importances ===
try:
    plt.figure(figsize=(8, 6))
    xgb.plot_importance(
        stack.named_estimators_['xgb'],
        max_num_features=10
    )
    plt.title('XGB Top Features')
    plt.tight_layout()
    plt.savefig('xgb_importance.png')
    plt.show()
except Exception:
    pass
