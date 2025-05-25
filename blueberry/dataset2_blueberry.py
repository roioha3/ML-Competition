import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb

# 1) Auto-detect input dir
INPUT_ROOT = Path('/kaggle/input')
if INPUT_ROOT.exists():
    subs = [d for d in INPUT_ROOT.iterdir() if d.is_dir()]
    DATA_DIR = subs[0] if len(subs)==1 else next(d for d in subs if (d/'train.csv').exists())
else:
    DATA_DIR = Path('.')

# 2) Load data
train = pd.read_csv(DATA_DIR/'train.csv')
test  = pd.read_csv(DATA_DIR/'test.csv')

# 3) Enhanced feature engineering
def add_features(df):
    df = df.copy()
    # baseline features
    df['bee_total']       = df[['honeybee','bumbles','andrena','osmia']].sum(axis=1)
    df['upper_temp_diff'] = df['MaxOfUpperTRange'] - df['MinOfUpperTRange']
    df['lower_temp_diff'] = df['MaxOfLowerTRange'] - df['MinOfLowerTRange']
    # new: mean temperatures
    df['upper_temp_mean'] = (df['MaxOfUpperTRange'] + df['MinOfUpperTRange']) / 2
    df['lower_temp_mean'] = (df['MaxOfLowerTRange'] + df['MinOfLowerTRange']) / 2
    # new: pollinator ratios
    for col in ['honeybee','bumbles','andrena','osmia']:
        df[f'{col}_ratio'] = df[col] / (df['bee_total'] + 1e-9)
    return df

train = add_features(train)
test  = add_features(test)

# 4) Prepare matrices
X       = train.drop(columns=['id','yield'])
y       = train['yield']
X_test  = test.drop(columns=['id'])

# 5) Log-transform target
y_log = np.log1p(y)

# 6) Train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

dtrain = xgb.DMatrix(X_tr,    label=y_tr)
dvalid = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(X_test)

# 7) XGB params tuned for MAE
params = {
    'objective':        'reg:absoluteerror',
    'eval_metric':      'mae',
    'tree_method':      'hist',
    'learning_rate':    0.02,
    'max_depth':        5,
    'subsample':        0.85,
    'colsample_bytree': 0.7,
    'gamma':            0.2,
    'reg_alpha':        0.8,
    'reg_lambda':       1.2,
    'min_child_weight': 1,
    'seed':             42,
}

# 8) Train with early stopping
evals = [(dtrain, 'train'), (dvalid, 'valid')]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=20_000,
    evals=evals,
    early_stopping_rounds=100,
    verbose_eval=200
)

# 9) Predict & back-transform
preds_log = bst.predict(dtest, iteration_range=(0, bst.best_iteration+1))
preds     = np.expm1(preds_log)

# 10) Save submission
submission = pd.DataFrame({
    'id':    test['id'],
    'yield': preds
})
submission.to_csv('submission.csv', index=False)
print("✅ OOF MAE on validation:", 
      np.expm1(bst.best_score) if hasattr(bst, 'best_score') else '—')
print("✅ Saved submission.csv")
