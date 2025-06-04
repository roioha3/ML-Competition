from lightgbm import early_stopping

# coding: utf-8

import pandas as pd
import numpy as np
import random
import warnings
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
np.random.seed(2023)
random.seed(2023)

# Load Data
origin_data = pd.read_csv('./mohs/Mineral_Dataset_Supplementary_Info.csv', index_col=0)
train_df = pd.read_csv('./mohs/train.csv')
train_df = pd.concat((train_df, origin_data), axis=0)
test_df = pd.read_csv('./mohs/test.csv')

# Create hardness labels
y = train_df['Hardness'].values
unique_target = np.array([1.25, 2.25, 3.15, 4, 5.2, 5.75, 6.25, 7, 8.1, 9.2])
y_label = []
for i in range(len(y)):
    min_dis = 1
    best_label = 0
    for j in range(len(unique_target)):
        dis = abs(y[i] - unique_target[j])
        if dis < min_dis:
            min_dis = dis
            best_label = j
    y_label.append(best_label)
train_df['Hardness_label'] = y_label

# Feature Engineering
def feature_engineer(df):
    df['allelectrons_Average_grade'] = (df['allelectrons_Average'] // 10)
    df['val_e_Average_grade'] = (df['val_e_Average'] // 2)
    df['atomicweight_Average_grade'] = (df['atomicweight_Average'] // 20)
    df['ionenergy_Average_grade'] = (df['ionenergy_Average'] // 5)
    df['el_neg_chi_Average_grade'] = (df['el_neg_chi_Average'] // 1)
    return df

def skew(data):
    data = np.asarray(data)
    return np.mean((data - np.mean(data)) ** 3)

total_df = pd.concat((train_df, test_df), axis=0)
total_df.drop(['id'], axis=1, inplace=True)
keys = total_df.drop(['Hardness', 'Hardness_label'], axis=1).keys().values
for key in keys:
    key_skew = skew(train_df[key].values)
    if key_skew > 0.5:
        total_df[key] = np.log1p(total_df[key])
    elif key_skew < -0.5:
        total_df[key] = total_df[key] ** 2

total_df = feature_engineer(total_df)
total_df.drop(['atomicweight_Average'], axis=1, inplace=True)

train_df = total_df[:len(train_df)]
keys = train_df.keys().values
TARGET_NAME = 'Hardness'
cat_keys = [key for key in keys if ((TARGET_NAME not in key) and (train_df[key].nunique() > 2) and (train_df[key].nunique() <= 30))]
for key in cat_keys:
    total_df[key + '_target_mean'] = total_df.groupby([key])[TARGET_NAME].transform('mean')
    total_df[key + '_target_std'] = total_df.groupby([key])[TARGET_NAME].transform('std')
    total_df[key + '_target_skew'] = total_df.groupby([key])[TARGET_NAME].transform('skew')
    key_target = train_df[TARGET_NAME].groupby([train_df[key]]).count()
    keys_ = key_target.keys().values
    target = key_target.values
    key_target = pd.DataFrame({key: keys_, key + f"_{TARGET_NAME}_count": target})
    total_df = pd.merge(total_df, key_target, on=key, how="left")

train_df = total_df[:len(train_df)]
test_df = total_df[len(train_df):]

# Model Training
def MEDAE(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

lgbm_params = {
    'random_state': 1819,
    'n_estimators': 309,
    'reg_alpha': 0.009043959900513852,
    'reg_lambda': 6.932606602460183,
    'colsample_bytree': 0.6183243994985523,
    'subsample': 0.6595851034943229,
    'learning_rate': 0.016870023802940223,
    'num_leaves': 50,
    'min_child_samples': 27
}

folds = 10
y = train_df[['Hardness', 'Hardness_label']]
X = train_df.drop(['Hardness', 'Hardness_label'], axis=1)
models = []
base_weight = 0.3
kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2023)

for train_index, valid_index in kf.split(X, y.iloc[:, 1]):
    x_train_cv = X.iloc[train_index]
    y_train_cv = y.iloc[train_index]
    x_valid_cv = X.iloc[valid_index]
    y_valid_cv = y.iloc[valid_index]

    model = LGBMClassifier(**lgbm_params)
    model.fit(
        x_train_cv.values,
        y_train_cv.values[:, 1],
        eval_set=[(x_train_cv.values, y_train_cv.values[:, 1]), (x_valid_cv.values, y_valid_cv.values[:, 1])],
        callbacks=[early_stopping(stopping_rounds=100)],
    )

    y_pred_train = model.predict(x_train_cv).astype(np.int64)
    ae = np.abs(y_train_cv.values[:, 0] - unique_target[y_pred_train])
    q10 = np.percentile(ae, 10)
    q75 = np.percentile(ae, 75)
    class_weight = base_weight + ((ae >= q10) & (ae <= q75))

    model = LGBMClassifier(**lgbm_params)
    model.fit(
        x_train_cv.values,
        y_train_cv.values[:, 1],
        eval_set=[(x_train_cv.values, y_train_cv.values[:, 1]), (x_valid_cv.values, y_valid_cv.values[:, 1])],
        callbacks=[early_stopping(stopping_rounds=100)],
        sample_weight=class_weight
    )
    models.append(model)

# Load original test data (10000 rows)
original_test = pd.read_csv('mohs/test.csv')

# Slice the last 10000 rows of processed test_df
test_features = test_df.iloc[-len(original_test):].reset_index(drop=True)
test_X = test_features.drop(['Hardness', 'Hardness_label'], axis=1).values

# Predict
preds_test = []
for model in models:
    pred = model.predict(test_X).astype(np.int64)
    preds_test.append(unique_target[pred])

# Voting
preds_test_np = np.array(preds_test)
test_pred = mode(pd.DataFrame(preds_test_np), axis=0, keepdims=True)[0][0]

# Write submission
submission = pd.read_csv('mohs/sample_submission.csv')
submission['Hardness'] = test_pred
submission.to_csv('mohs/classifier.csv', index=False)
