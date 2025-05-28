#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
import random
import warnings
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping

warnings.filterwarnings('ignore')
np.random.seed(2023)
random.seed(2023)

# Load and merge training data
train1 = pd.read_csv('mohs/train.csv')
train2 = pd.read_csv('mohs/train2.csv')
train2 = train2[train1.columns]  # reorder to match train1
train_df = pd.concat([train1, train2], ignore_index=True)

# Load test and auxiliary data
test_df = pd.read_csv('mohs/test.csv')
origin_data = pd.read_csv('mohs/Mineral_Dataset_Supplementary_Info.csv', index_col=0)

# Append additional data to training
train_df = pd.concat([train_df, origin_data], ignore_index=True)

# Assign class labels for classification task
y = train_df['Hardness'].values
unique_target = np.array([1.25, 2.25, 3.15, 4, 5.2, 5.75, 6.25, 7, 8.1, 9.2])
train_df['Hardness_label'] = [np.argmin([abs(val - u) for u in unique_target]) for val in y]

# Feature engineering
def feature_engineer(df):
    df['allelectrons_Average_grade'] = df['allelectrons_Average'] // 10
    df['val_e_Average_grade'] = df['val_e_Average'] // 2
    df['atomicweight_Average_grade'] = df['atomicweight_Average'] // 20
    df['ionenergy_Average_grade'] = df['ionenergy_Average'] // 5
    df['el_neg_chi_Average_grade'] = df['el_neg_chi_Average'] // 1
    return df

def skew(data):
    data = np.asarray(data)
    return np.mean((data - np.mean(data)) ** 3)

# Merge train + test for consistent preprocessing
total_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
total_df.drop(['id'], axis=1, inplace=True)

for col in total_df.drop(['Hardness', 'Hardness_label'], axis=1).columns:
    if skew(total_df[col].values) > 0.5:
        total_df[col] = np.log1p(total_df[col])
    elif skew(total_df[col].values) < -0.5:
        total_df[col] = total_df[col] ** 2

total_df = feature_engineer(total_df)
total_df.drop(['atomicweight_Average'], axis=1, inplace=True)

# Target-based encodings
train_end = len(train_df)
train_df = total_df[:train_end]
test_df = total_df[train_end:]

target_col = 'Hardness'
categoricals = [col for col in train_df.columns if target_col not in col and 2 < train_df[col].nunique() <= 30]

for col in categoricals:
    for agg in ['mean', 'std', 'skew']:
        total_df[f"{col}_target_{agg}"] = total_df.groupby(col)[target_col].transform(agg)
    count_df = train_df[target_col].groupby(train_df[col]).count().reset_index()
    count_df.columns = [col, f"{col}_{target_col}_count"]
    total_df = pd.merge(total_df, count_df, on=col, how="left")

train_df = total_df[:train_end]
test_df = total_df[train_end:]

# Modeling
def MEDAE(y_true, y_pred): return np.median(np.abs(y_true - y_pred))
def accuracy(y_true, y_pred): return np.mean(y_true == y_pred)

X = train_df.drop(['Hardness', 'Hardness_label'], axis=1)
y = train_df[['Hardness', 'Hardness_label']]
models = []

lgbm_params = {
    'random_state': 1819,
    'n_estimators': 309,
    'reg_alpha': 0.009,
    'reg_lambda': 6.93,
    'colsample_bytree': 0.618,
    'subsample': 0.659,
    'learning_rate': 0.0168,
    'num_leaves': 50,
    'min_child_samples': 27
}

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2023)

for train_idx, val_idx in kf.split(X, y['Hardness_label']):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(**lgbm_params)
    model.fit(
        X_train.values,
        y_train['Hardness_label'],
        eval_set=[(X_val.values, y_val['Hardness_label'])],
        callbacks=[early_stopping(stopping_rounds=100)]
    )

    y_pred_train = model.predict(X_train).astype(int)  # ensure integer indices
    ae = np.abs(y_train['Hardness'].values - unique_target[y_pred_train])

    q10, q75 = np.percentile(ae, [10, 75])
    weights = 0.3 + ((ae >= q10) & (ae <= q75))

    model = LGBMClassifier(**lgbm_params)
    model.fit(
        X_train.values,
        y_train['Hardness_label'],
        eval_set=[(X_val.values, y_val['Hardness_label'])],
        callbacks=[early_stopping(stopping_rounds=100)],
        sample_weight=weights
    )
    models.append(model)

# Inference
original_test = pd.read_csv('mohs/test.csv')
test_features = test_df.iloc[-len(original_test):].reset_index(drop=True)
X_test = test_features.drop(['Hardness', 'Hardness_label'], axis=1).values

preds = [unique_target[model.predict(X_test).astype(int)] for model in models]
preds = np.array(preds)
final_preds = mode(preds, axis=0, keepdims=True)[0][0]

# Output
submission = pd.DataFrame({'id': original_test['id'], 'Hardness': final_preds})
submission.to_csv('mohs/classifier.csv', index=False)
