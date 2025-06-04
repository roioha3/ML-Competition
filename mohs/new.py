import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import median_absolute_error
from xgboost import XGBClassifier

# === Configuration ===
SEED = 666
N_SPLITS, N_REPEATS = 5, 3
TARGET = "Hardness"
PATH = "./mohs"
USE_GPU = True

# === Fixed shared feature list ===
FEATURES = [
    "allelectrons_Total", "density_Total", "allelectrons_Average", "val_e_Average",
    "atomicweight_Average", "ionenergy_Average", "el_neg_chi_Average",
    "R_vdw_element_Average", "R_cov_element_Average", "zaratio_Average", "density_Average"
]

# === Target encoding bins ===
target_values = [2.0, 2.5, 3.0, 3.5, 4.0, 4.8, 5.75, 6.4, 6.8]
target_dict = {v: i for i, v in enumerate(target_values)}
dec_target_dict = {i: v for v, i in target_dict.items()}
N_CLASSES = len(target_values)

def encode_target(df):
    diffs = np.abs(df[[TARGET]].values - np.array(target_values).reshape(1, -1))
    df[f"{TARGET}_enc"] = np.argmin(diffs, axis=1)
    df[f"{TARGET}_new"] = df[f"{TARGET}_enc"].map(dec_target_dict)
    return df

# === Load datasets ===
train1 = pd.read_csv(f"{PATH}/train.csv")
train2 = pd.read_csv(f"{PATH}/train2.csv")
origin = pd.read_csv(f"{PATH}/Mineral_Dataset_Supplementary_Info.csv")
test = pd.read_csv(f"{PATH}/test.csv")

# Rename column in origin to match target
origin.rename(columns={"Hardness (Mohs)": "Hardness"}, inplace=True)

# Keep only required features + target
origin = origin[[TARGET] + FEATURES]
train1 = train1[[TARGET] + FEATURES]
train2 = train2[[TARGET] + FEATURES]
test_X = test[FEATURES]

# Combine train1 and train2
train = pd.concat([train1, train2], ignore_index=True)

# Encode target to bins
train = encode_target(train)
origin = encode_target(origin)

X = train[FEATURES]
y = train["Hardness_enc"]

# === Cross-validation setup ===
rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

oof_preds = np.zeros((len(X), N_CLASSES))
test_preds = np.zeros((len(test_X), N_CLASSES))

# === Training loop ===
for fold, (train_idx, valid_idx) in enumerate(rkf.split(X, y), 1):
    print(f"Training Fold {fold}")
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    
    # Augment training data with origin
    X_train_aug = pd.concat([X_train, origin[FEATURES]], axis=0)
    y_train_aug = pd.concat([y_train, origin["Hardness_enc"]], axis=0)
    
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=N_CLASSES,
        tree_method='gpu_hist' if USE_GPU else 'hist',
        learning_rate=0.1,
        max_depth=5,
        subsample=0.6,
        colsample_bytree=0.6,
        n_estimators=300,
        eval_metric="mlogloss",
        random_state=SEED + fold
    )

    model.fit(
        X_train_aug, y_train_aug,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=30,
        verbose=False
    )
    
    oof_preds[valid_idx] = model.predict_proba(X_valid)
    test_preds += model.predict_proba(test_X[FEATURES]) / (N_SPLITS * N_REPEATS)

# === Evaluate OOF Median Absolute Error ===
oof_labels = np.argmax(oof_preds, axis=1)
decoded_oof = np.vectorize(dec_target_dict.get)(oof_labels)
true_values = train[TARGET].values
medae = median_absolute_error(true_values, decoded_oof)
print(f"✅ OOF Median Absolute Error: {medae:.4f}")

# === Final test prediction ===
final_labels = np.argmax(test_preds, axis=1)
decoded_test = np.vectorize(dec_target_dict.get)(final_labels)

submission = pd.DataFrame({
    "id": test["id"],
    "Hardness": decoded_test
})
submission.to_csv("mohs/submission.csv", index=False)
print("✅ submission.csv saved.")
