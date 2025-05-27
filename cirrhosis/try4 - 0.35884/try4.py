#!/usr/bin/env python3
"""
cirrhosis_pipeline.py

– Loads train.csv & test.csv
– Applies feature engineering
– Encodes, scales, optionally tunes with Optuna, trains an XGBClassifier
– Writes submission.csv with probabilities for C / CL / D
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from category_encoders import OrdinalEncoder
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import optuna

# --- Configuration ---
RUN_OPTUNA_STUDY = True   # False to skip tuning and use defaults
OPTUNA_N_TRIALS  = 30

# Flag for warning once
optuna_warning_printed = False

# 1) Load data
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")
ids      = test_df["id"]

# 2) Feature engineering
def convert_days_to_years(days):
    return days / 365.25

for df in (train_df, test_df):
    df["Age_in_year"]       = df["Age"].apply(convert_days_to_years).astype(int)
    df["thrombocytopenia"]  = (df["Platelets"] < 150).astype(int)
    df["elevated_alk_phos"] = ((df["Alk_Phos"] > 147) | (df["Alk_Phos"] < 44)).astype(int)
    df["normal_copper"]     = df["Copper"].between(62,140).astype(int)
    df["normal_sgot"]       = df["SGOT"].between(8,45).astype(int)
    df["normal_p_time"]     = df["Prothrombin"].between(9.4,12.5).astype(int)
    df["normal_albumin"]    = df["Albumin"].between(3.4,5.4).astype(int)
    df["normal_bilirubin"]  = df["Bilirubin"].between(0.2,1.2).astype(int)
    df["DiagnosisDays"]     = df["Age"] - df["N_Days"]

    # Age group (0,1,2,3)
    age_cat = pd.cut(
        df["Age_in_year"], bins=[19,29,49,64,np.inf], labels=[0,1,2,3], right=False
    )
    df["Age_Group"] = age_cat.cat.codes
    df["Age_Group"] = df["Age_Group"].where(df["Age_Group"] >= 0, 0).astype(int)

    # Composite features
    df["Bilirubin_Albumin_Product"] = df["Bilirubin"] * df["Albumin"]
    df["Symptom_Score"] = df[["Ascites","Hepatomegaly","Spiders"]].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).sum(axis=1)
    df["Liver_Function_Index"] = df[["Bilirubin","Albumin","Alk_Phos","SGOT"]].mean(axis=1)
    df["Risk_Score"] = df["Bilirubin"] + df["Albumin"] - df["Alk_Phos"]
    df["Diag_Year"]  = (df["N_Days"]/365).astype(int)
    df["Diag_Month"] = ((df["N_Days"]%365)/30).astype(int)

# 3) Prepare feature matrices
X      = train_df.drop(columns=["id","Status"])
y      = train_df["Status"].map({"C":0,"CL":1,"D":2}).astype(int)
X_test = test_df.drop(columns=["id"])

# Align columns
X_test = X_test.reindex(columns=X.columns, fill_value=np.nan)

# 4) Encode categorical and scale numerical
cat_cols   = X.select_dtypes(include="object").columns.tolist()
encoder    = OrdinalEncoder(cols=cat_cols, handle_unknown='value', handle_missing='value')
X_enc      = encoder.fit_transform(X, y)
X_test_enc = encoder.transform(X_test)

# Fill any remaining NaNs
for col in X_enc.columns:
    if X_enc[col].isnull().any():
        med = X_enc[col].median()
        X_enc[col] = X_enc[col].fillna(med)
        X_test_enc[col] = X_test_enc[col].fillna(med)

scaler      = StandardScaler()
X_scaled    = scaler.fit_transform(X_enc)
X_test_scaled = scaler.transform(X_test_enc)

# 5) Hyperparameter tuning with Optuna
if RUN_OPTUNA_STUDY:
    print("Running Optuna for hyperparameter optimization...")
    def objective(trial):
        params = {
            'objective':'multi:softprob','num_class':3,'eval_metric':'mlogloss','tree_method':'hist',
            'n_estimators': trial.suggest_int('n_estimators',200,2000,step=100),
            'learning_rate': trial.suggest_float('learning_rate',0.005,0.1,log=True),
            'max_depth': trial.suggest_int('max_depth',3,12),
            'subsample': trial.suggest_float('subsample',0.6,1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree',0.4,1.0),
            'min_child_weight': trial.suggest_int('min_child_weight',1,10),
            'reg_alpha': trial.suggest_float('reg_alpha',1e-7,1.0,log=True),
            'reg_lambda': trial.suggest_float('reg_lambda',1e-7,1.0,log=True),
            'gamma': trial.suggest_float('gamma',1e-7,0.5,log=True),
        }
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=trial.number)
        loglosses = []
        X_np = X_scaled
        y_np = y.to_numpy()
        for fold, (ti, vi) in enumerate(kf.split(X_np, y_np)):
            X_tr, X_val = X_np[ti], X_np[vi]
            y_tr, y_val = y_np[ti], y_np[vi]
            model = xgb.XGBClassifier(**params, random_state=42+fold)
            fit_xgboost_params = {'eval_set':[(X_val,y_val)], 'verbose':False}
            try:
                fit_xgboost_params['early_stopping_rounds'] = 50
                model.fit(X_tr, y_tr, **fit_xgboost_params)
            except TypeError as e:
                if "unexpected keyword argument 'early_stopping_rounds'" in str(e):
                    global optuna_warning_printed
                    if not optuna_warning_printed:
                        print("[Warning] XGBoost version may not support early_stopping_rounds; skipping for tuning.")
                        optuna_warning_printed = True
                    fit_xgboost_params.pop('early_stopping_rounds', None)
                    model.fit(X_tr, y_tr, **fit_xgboost_params)
                else:
                    raise
            preds = model.predict_proba(X_val)
            loglosses.append(log_loss(y_val, preds))
        return np.mean(loglosses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
    print(f"Best trial logloss: {study.best_value}")
    final_params = study.best_params
    final_params.update({'objective':'multi:softprob','num_class':3,'eval_metric':'mlogloss','tree_method':'hist'})
else:
    print("Using default XGBoost parameters.")
    final_params = {
        'objective':'multi:softprob','num_class':3,'eval_metric':'mlogloss','tree_method':'hist',
        'n_estimators':1000,'learning_rate':0.02,'max_depth':6,
        'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':3,
        'reg_alpha':0.1,'reg_lambda':0.1,'gamma':0.01
    }

# 6) Train final model
print("Training final model...")
model = xgb.XGBClassifier(**final_params, random_state=42)
model.fit(X_scaled, y)

# 7) Predict and save submission
probs = model.predict_proba(X_test_scaled)
submission = pd.DataFrame(probs, columns=["Status_C","Status_CL","Status_D"])
submission.insert(0, "id", ids)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")