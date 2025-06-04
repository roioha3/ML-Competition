import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import median_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
import optuna

# === Load data ===
train1 = pd.read_csv('mohs/train.csv')
train2 = pd.read_csv('mohs/train2.csv')
test_dataset = pd.read_csv('mohs/test.csv')
sample_submission = pd.read_csv('mohs/sample_submission.csv')
original = pd.read_csv('mohs/Artificial_Crystals_Dataset.csv')

# === Prepare combined dataset ===
train_dataset = pd.concat([train1, train2], ignore_index=True)
original.drop(columns=['Unnamed: 0', 'Formula', 'Crystal structure'], inplace=True)
original.rename(columns={'Hardness (Mohs)': 'Hardness'}, inplace=True)
train_dataset = pd.concat([train_dataset, original], ignore_index=True)
train_dataset.reset_index(drop=True, inplace=True)

# === Feature engineering ===
def new_features(df):
    df['ionenergy_val_e'] = df['ionenergy_Average'] / (df['val_e_Average'] + 1e-7)
    df['el_neg_chi_R_cov'] = df['el_neg_chi_Average'] / (df['R_cov_element_Average'] + 1e-7)
    df['atomicweight_ionenergy_Ratio'] = df['atomicweight_Average'] / (df['ionenergy_Average'] + 1e-7)
    df['n_elements'] = df['allelectrons_Total'] / (df['allelectrons_Average'] + 1e-5)
    df['total_weight'] = df['n_elements'] * df['atomicweight_Average']

new_features(train_dataset)
new_features(test_dataset)

# === Drop highly correlated features ===
def high_corr_drop(df, threshold=0.91):
    corr = df.select_dtypes(include=[np.number]).corr().abs()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    to_drop = corr.where(mask).gt(threshold).sum().gt(0)
    return df.drop(columns=corr.columns[to_drop])

train_dataset = high_corr_drop(train_dataset)
kept_columns = train_dataset.columns  # Save consistent feature list
test_dataset = test_dataset[kept_columns.intersection(test_dataset.columns)]

# === Filter out rows with 0s in numerical columns ===
num_cols = [col for col in train_dataset.columns if train_dataset[col].dtype in [np.float64, np.int64] and train_dataset[col].nunique() > 50]
train_dataset = train_dataset[~(train_dataset[num_cols] == 0).any(axis=1)].reset_index(drop=True)

# === Ensure consistent columns between train and test ===
X = train_dataset.drop(columns=['id', 'Hardness'], errors='ignore')
y = train_dataset['Hardness']
X_test = test_dataset.drop(columns=['id'], errors='ignore')

# Align columns in test to match training features exactly
X_test = X_test[X.columns]


# === Optuna tuning ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1147)

def objective(trial):
    params = {
        'booster': 'gbtree',
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.3, 0.9, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        'objective': 'reg:absoluteerror',
        'random_state': 42,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return median_absolute_error(y_val, y_pred)

study = optuna.create_study(direction='minimize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=50)

# === Classification trick: round to categories ===
cats = np.array([1.75, 2.55, 3.75, 4.75, 5.75, 6.55, 7.75, 8.75, 9.75])
encoder = LabelEncoder()

def round_to_nearest(y, known_values=cats):
    y_array = np.tile(y.to_numpy(), (len(known_values), 1)).T
    return known_values[np.abs(y_array - known_values).argmin(axis=1)]

# === Stratified classification with XGB ===
y_cats = encoder.fit_transform(round_to_nearest(y))
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
xgb_cls = XGBClassifier(max_depth=3, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_cats), start=1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr = encoder.transform(round_to_nearest(y.iloc[train_idx]))
    xgb_cls.fit(X_tr, y_tr)
    y_pred = xgb_cls.predict(X.iloc[val_idx])
    medae = median_absolute_error(y.iloc[val_idx], encoder.inverse_transform(y_pred))
    print(f"Fold {fold:2d} - MedAE: {medae:.3f}")

# === Final training and prediction ===
xgb_cls_final = XGBClassifier(max_depth=3, random_state=42)
y_cats_full = encoder.transform(round_to_nearest(y))
xgb_cls_final.fit(X, y_cats_full)
test_preds = xgb_cls_final.predict(X_test)
predicted_hardness = encoder.inverse_transform(test_preds)

# === Submission ===
submission = pd.DataFrame({
    'id': test_dataset['id'],
    'Hardness': predicted_hardness
})
submission.to_csv('mohs/submission.csv', index=False)
