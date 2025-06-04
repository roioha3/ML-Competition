import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
# import optuna # Optuna is commented out as per your request
import seaborn as sns # Not directly used in the final prediction logic, but often useful for EDA
import matplotlib.pyplot as plt # Not directly used in the final prediction logic, but often useful for EDA

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestClassifier # Not used in your final ensemble
from sklearn.model_selection import train_test_split # Used in your original snippet for Optuna, removed for final training
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split # Duplicate import, removed
from sklearn.metrics import confusion_matrix # Not directly used in the final prediction logic
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, make_scorer
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings("ignore")

# --- 1. Data Loading ---
print("--- 1. Data Loading ---")

# Load the datasets. Assuming 'train.csv' and 'test.csv' are in the same directory.
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    # Removed main = pd.read_csv('/kaggle/input/cirrhosis-patient-survival-prediction/cirrhosis.csv')
    print("Datasets loaded successfully: train.csv, test.csv")
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure 'train.csv' and 'test.csv' are in the same directory.")
    exit() # Exit if files are not found

# Store test IDs for submission before dropping the 'id' column
test_ids = test_df['id']

# No concatenation with 'main' dataset as per your instruction
# train = pd.concat([train, main], axis = 0) # This line is removed

# --- 2. Data Preprocessing ---
print("\n--- 2. Data Preprocessing ---")

# One-hot encode the target variable
# This will map 'C', 'CL', 'D' to 0, 1, 2 respectively (alphabetical order by default)
y = train_df['Status']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y) # Renamed to y_encoded to avoid confusion with X, y split

# Drop 'id' and 'Status' from the training features
X = train_df.drop(['id', 'Status'], axis=1)
# Drop 'id' from the test features
test_features = test_df.drop(['id'], axis=1) # Renamed to test_features to avoid overwriting 'test_df'

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

print(f"Numerical features: {list(num_cols)}")
print(f"Categorical features: {list(cat_cols)}")

# Preprocessing for numerical data: imputation and scaling
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: imputation and one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# --- 3. Model Definitions with Preprocessor Pipelines ---
print("\n--- 3. Model Definitions ---")

# XGBoost Model Pipeline
# Best hyperparameters provided in your snippet
xgb_model = XGBClassifier(**{
    'objective': 'multi:softprob',
    'max_depth': 29,
    'learning_rate': 0.044754600706634465,
    'n_estimators': 850,
    'subsample': 0.6529370752777335,
    'colsample_bytree': 0.17930720266844047,
    'gamma': 0.8811229412407853,
    'booster': 'gbtree',
    'random_state': 42 # Added for reproducibility
})
XBG_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb_model)])
print("XGBoost pipeline created.")

# LightGBM Model Pipeline
# Best hyperparameters provided in your snippet
lgbm_model = LGBMClassifier(**{
    'objective': 'multiclass',
    'metric': 'softmax',
    'boosting_type': 'gbdt',
    'num_leaves': 13,
    'learning_rate': 0.040717487378551125,
    'n_estimators': 340,
    'subsample': 0.7621117946415148,
    'colsample_bytree': 0.5254951985706161,
    'reg_alpha': 0.9395639914591739,
    'reg_lambda': 0.12423847695048462,
    'random_state': 42, # Added for reproducibility
    'n_jobs': -1, # Use all available cores
    'verbose': -1 # Suppress verbose output
})
LGBM_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', lgbm_model)])
print("LightGBM pipeline created.")

# CatBoost Model Pipeline
# Best hyperparameters provided in your snippet
cat_model = CatBoostClassifier(**{
    'iterations': 860,
    'learning_rate': 0.07693815312911455,
    'depth': 3,
    'subsample': 0.5592260503739381,
    'colsample_bylevel': 0.6756953610569407,
    'bootstrap_type': 'Bernoulli', # Changed from 'Bayesian' to 'Bernoulli' to allow 'subsample'
    'grow_policy': 'Depthwise',
    'min_child_samples': 17,
    'reg_lambda': 0.39267373457724253,
    'random_seed': 42, # Added for reproducibility
    'verbose': 0 # Suppress verbose output
})
Cat_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', cat_model)])
print("CatBoost pipeline created.")

# Gradient Boosting Model Pipeline (though not included in your final VotingClassifier)
# Best hyperparameters provided in your snippet
# Kept for completeness if you decide to include it later
gradient_model = GradientBoostingClassifier(**{
    'loss': 'deviance',
    'learning_rate': 0.05885877940951943,
    'n_estimators': 300,
    'subsample': 0.5470272507158589,
    'max_depth': 3,
    'random_state': 42 # Added for reproducibility
})
Gradient_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', gradient_model)])
print("Gradient Boosting pipeline created (not included in final ensemble).")


# --- 4. Ensemble Training ---
print("\n--- 4. Ensemble Training (VotingClassifier) ---")

# Create a VotingClassifier with the three specified pipelines
# 'voting='soft'' means it will average the predicted probabilities
voting_classifier = VotingClassifier(estimators=[
    ('XGBoost', XBG_pipeline),
    ('LightGBM', LGBM_pipeline),
    ('CatBoost', Cat_pipeline)
], voting='soft', n_jobs=-1) # Use all available cores for fitting

# Train the VotingClassifier on the full training data
# Removed X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# as the final model is trained on the full dataset.
print("Fitting VotingClassifier on the entire training dataset...")
voting_classifier.fit(X, y_encoded)
print("VotingClassifier training complete.")

# --- 5. Generating the Submission File ---
print("\n--- 5. Generating the Submission File ---")

# Make predictions on the preprocessed test data
# Use the test_features DataFrame which has 'id' column dropped
predictions = voting_classifier.predict_proba(test_features)

# Create submission DataFrame
submission_df = pd.DataFrame({'id': test_ids})

# Map numerical predictions back to original class labels
# label_encoder.classes_ will give the order: ['C', 'CL', 'D'] if they are sorted alphabetically
# Ensure the column names match the required format: Status_C, Status_CL, Status_D
for i, class_name in enumerate(label_encoder.classes_):
    submission_df[f'Status_{class_name}'] = predictions[:, i]

# Reorder columns to ensure 'id', 'Status_C', 'Status_CL', 'Status_D'
# Assuming label_encoder.classes_ naturally sorts to C, CL, D
required_columns = ['id', 'Status_C', 'Status_CL', 'Status_D']
submission_df = submission_df[required_columns]

# Save the submission file
submission_file_name = 'submission.csv'
submission_df.to_csv(submission_file_name, index=False)

print(f"\nSubmission file '{submission_file_name}' created successfully.")
print("First 5 rows of the submission file:")
print(submission_df.head())
