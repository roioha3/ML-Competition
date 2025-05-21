# dataset3_cirrhosis.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, log_loss

# === Load data ===
train = pd.read_csv("cirrhosis/train.csv")
test = pd.read_csv("cirrhosis/test.csv")

# === Encode categorical features ===
X = pd.get_dummies(train.drop(columns=["Status"]), drop_first=False)
y = train["Status"]

test_encoded = pd.get_dummies(test, drop_first=False)
X, test_encoded = X.align(test_encoded, join='left', axis=1, fill_value=0)

# === Split train/val ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# === Model ===
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_val)
y_proba_val = model.predict_proba(X_val)
print(classification_report(y_val, y_pred))
print(f"Validation Log Loss: {log_loss(y_val, y_proba_val, labels=model.classes_):.4f}")

# === Predict test probabilities ===
y_test_proba = model.predict_proba(test_encoded)

# === Create submission ===
submission = pd.DataFrame(y_test_proba, columns=[f"Status_{c}" for c in model.classes_])
submission.insert(0, "id", range(len(test)))
submission.to_csv("cirrhosis/submission.csv", index=False)
print("✅ cirrhosis/submission.csv written.")
