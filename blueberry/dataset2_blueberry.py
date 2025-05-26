import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

# === 1. Load ===
train = pd.read_csv("blueberry/train.csv")
test  = pd.read_csv("blueberry/test.csv")

# === 2. Prepare X/y ===
X = train.drop(columns=["id", "yield"])
y = train["yield"]

# === 3. Simple Feature Engineering ===
def add_features(df):
    df = df.copy()
    # total pollinators
    df["bee_total"] = df[["honeybee", "bumbles", "andrena", "osmia"]].sum(axis=1)
    # temperature ranges
    df["upper_temp_diff"] = df["MaxOfUpperTRange"] - df["MinOfUpperTRange"]
    df["lower_temp_diff"] = df["MaxOfLowerTRange"] - df["MinOfLowerTRange"]
    return df

X = add_features(X)
X_test = add_features(test.drop(columns=["id"], errors="ignore"))

# === 4. Drop any now-constant cols & align features ===
# (this will also drop any columns that are constant after adding features)
valid_feats = X.columns[X.nunique() > 1]
X = X[valid_feats]
X_test = X_test.reindex(columns=valid_feats, fill_value=0)

# === 5. Train/Validation Split ===
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 6. Hyperparameter Tuning ===
param_dist = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [None, 10, 20, 30],
    "max_features":      ["sqrt", "log2", 0.5],  # fixed: one key only
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}


base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=False  # we'll refit manually on the full set
)

start = time.time()
search.fit(X_tr, y_tr)
print("🔍 Best params:", search.best_params_)
print("🏆 Best CV MAE: {:.4f}".format(-search.best_score_))
print("⏱  Tuning time: {:.1f} min".format((time.time() - start)/60))

# === 7. Refit on full training data ===
best_model = RandomForestRegressor(**search.best_params_, random_state=42, n_jobs=-1)
best_model.fit(X, y)

# === 8. Validation score ===
y_val_pred = best_model.predict(X_val)
print("✅ Validation MAE: {:.4f}".format(mean_absolute_error(y_val, y_val_pred)))

# === 9. Retrain on all data & predict test set ===
# (already refit on full X,y)
predictions = best_model.predict(X_test)

# === 10. Write submission ===
submission = pd.DataFrame({
    "id": test["id"],
    "Prediction": predictions
})
submission.to_csv("blueberry/submission.csv", index=False)
print("✅ submission.csv written.")