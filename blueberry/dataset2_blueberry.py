import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# 1) Auto‐detect input dir (Kaggle or local)
INPUT_ROOT = Path('/kaggle/input')
if INPUT_ROOT.exists():
    subs = [d for d in INPUT_ROOT.iterdir() if d.is_dir()]
    DATA_DIR = subs[0] if len(subs)==1 else next(d for d in subs if (d/'train.csv').exists())
else:
    DATA_DIR = Path('.')

# 2) Load
train = pd.read_csv(DATA_DIR/'train.csv')
test  = pd.read_csv(DATA_DIR/'test.csv')

# 3) Richer feature engineering + interactions
def add_features(df):
    df = df.copy()
    df['bee_total']       = df[['honeybee','bumbles','andrena','osmia']].sum(axis=1)
    df['upper_temp_diff'] = df['MaxOfUpperTRange'] - df['MinOfUpperTRange']
    df['lower_temp_diff'] = df['MaxOfLowerTRange'] - df['MinOfLowerTRange']
    df['upper_temp_mean'] = (df['MaxOfUpperTRange'] + df['MinOfUpperTRange']) / 2
    df['lower_temp_mean'] = (df['MaxOfLowerTRange'] + df['MinOfLowerTRange']) / 2
    for col in ['honeybee','bumbles','andrena','osmia']:
        df[f'{col}_ratio'] = df[col] / (df['bee_total'] + 1e-9)
    # interaction features
    df['bee×upper_mean'] = df['bee_total'] * df['upper_temp_mean']
    df['diff_ratio']     = df['lower_temp_diff'] / (df['upper_temp_diff'] + 1e-9)
    return df

train = add_features(train)
test  = add_features(test)

X       = train.drop(columns=['id','yield'])
y       = train['yield']
X_test  = test.drop(columns=['id'])

# 4) Log‐transform target
y_log = np.log1p(y)

# 5) Stacked 5‐fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds  = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# 6) XGBoost params (CPU histogram; switch to GPU by adding device='cuda')
params = {
    'objective':        'reg:absoluteerror',
    'eval_metric':      'mae',
    'tree_method':      'hist',     # use 'hist' here
    'learning_rate':    0.01,
    'max_depth':        6,
    'subsample':        0.9,
    'colsample_bytree': 0.8,
    'gamma':            0.1,
    'reg_alpha':        1.0,
    'reg_lambda':       1.0,
    'min_child_weight': 1,
    'seed':             42,
    'verbosity':        0,
    # if you *do* spin up a GPU runtime, uncomment:
    # 'device': 'cuda',
}

# 7) Learning‐rate schedule callback: halve LR every 1k rounds
def lr_schedule(iteration):
    base_lr = params['learning_rate']
    return base_lr * (0.5 ** (iteration // 1000))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n=== Fold {fold} ===")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(X_test)

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=20_000,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=200,
        callbacks=[xgb.callback.LearningRateScheduler(lr_schedule)],
        verbose_eval=500
    )

    oof_preds[val_idx] = bst.predict(
        dvalid, iteration_range=(0, bst.best_iteration+1)
    )
    test_preds += bst.predict(
        dtest, iteration_range=(0, bst.best_iteration+1)
    ) / kf.n_splits

# 8) Final OOF MAE & submission
oof = np.expm1(oof_preds)
print("\n>>> OOF MAE:", mean_absolute_error(y, oof))

submission = pd.DataFrame({
    'id':    test['id'],
    'yield': np.expm1(test_preds)
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv written")
