import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from category_encoders import OrdinalEncoder
import xgboost as xgb

# 1) Load data
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

# 2) Constants
IDCOL    = "id"
TARGET   = "Status"
FEATURES = [
    "N_Days", "Drug", "Age", "Sex", "Ascites", "Hepatomegaly",
    "Spiders", "Edema", "Bilirubin", "Cholesterol", "Albumin",
    "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets",
    "Prothrombin", "Stage",
]

# 3) Impute numeric columns
num_cols = train[FEATURES].select_dtypes(include="number").columns
for c in num_cols:
    m = train[c].median()
    train[c].fillna(m, inplace=True)
    test[c].fillna(m, inplace=True)

# 4) Encode categoricals
cat_cols = train[FEATURES].select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder(cols=cat_cols)
X_full  = encoder.fit_transform(train[FEATURES])
X_test  = encoder.transform(test[FEATURES])

# 5) Prepare target
y_full = train[TARGET].map({"C":0, "CL":1, "D":2})

# 6) Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full,
    test_size=0.2,
    stratify=y_full,
    random_state=42
)

# 7) Build DMatrix objects
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(X_test)

# 8) Set up parameters
params = {
    "objective":     "multi:softprob",
    "num_class":     3,
    "tree_method":   "hist",
    "eval_metric":   "mlogloss",
    "seed":          42,
}

# 9) Train with early stopping
evallist = [(dtrain, "train"), (dval, "eval")]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evallist,
    early_stopping_rounds=10,
    verbose_eval=True,
)

# 10) Validate
val_preds = bst.predict(dval)
print("Validation log loss:", log_loss(y_val, val_preds))

# 11) Predict test set & write submission
test_preds = bst.predict(dtest)
submission = pd.DataFrame(
    test_preds,
    columns=["Status_C", "Status_CL", "Status_D"]
)
submission.insert(0, IDCOL, test[IDCOL])
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
