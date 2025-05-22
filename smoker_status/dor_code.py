import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import h2o
from h2o.automl import H2OAutoML
import lightgbm as lgb
import xgboost as xgb
import optuna
from category_encoders import TargetEncoder
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    # Feature engineering logic...
    # [omitted for brevity - use the full version you've already written]
    return df

def handle_outliers(df, columns):
    for col in columns:
        q1 = df[col].quantile(0.01)
        q3 = df[col].quantile(0.99)
        df[col] = df[col].clip(q1, q3)
    return df

def preprocess_features(df, is_train=True, imputer=None, scaler=None, power_transformer=None, target_encoder=None):
    if is_train:
        X = df.drop(['id', 'smoking'], axis=1)
        y = df['smoking'].astype(float)
    else:
        X = df.drop(['id'], axis=1)
        y = None

    if imputer is None:
        imputer = KNNImputer(n_neighbors=5)
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    else:
        X = pd.DataFrame(imputer.transform(X), columns=X.columns)

    X = create_features(X)
    X = handle_outliers(X, X.select_dtypes(include=['float64', 'int64']).columns)

    if scaler is None:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)

    if power_transformer is None:
        power_transformer = PowerTransformer(method='yeo-johnson')
        X = pd.DataFrame(power_transformer.fit_transform(X), columns=X.columns)
    else:
        X = pd.DataFrame(power_transformer.transform(X), columns=X.columns)

    if target_encoder is None and is_train:
        target_encoder = TargetEncoder()
        X = pd.DataFrame(target_encoder.fit_transform(X, y), columns=X.columns)
    elif target_encoder is not None:
        X = pd.DataFrame(target_encoder.transform(X), columns=X.columns)

    return X, y, imputer, scaler, power_transformer, target_encoder

def optimize_lgbm(trial, X, y):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 16, 64),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_uniform('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, pred))

    return np.mean(cv_scores)

def train_models(X_train, y_train, X_test):
    models = {}

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optimize_lgbm(trial, X_train, y_train), n_trials=50)
    best_params = study.best_params

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_models = {
        'lgb': lgb.LGBMClassifier(**best_params),
        'xgb': xgb.XGBClassifier(
            n_estimators=2000, learning_rate=0.01, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1, reg_alpha=0.1, reg_lambda=0.1, random_state=42
        )
    }

    for model_name, model in base_models.items():
        cv_scores = []
        cv_predictions = np.zeros(len(X_test))

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_fold_train, y_fold_train)
            val_pred = model.predict_proba(X_fold_val)[:, 1]
            cv_scores.append(roc_auc_score(y_fold_val, val_pred))
            cv_predictions += model.predict_proba(X_test)[:, 1] / 5

        models[model_name] = {
            'model': model,
            'cv_score': np.mean(cv_scores),
            'predictions': cv_predictions
        }
        print(f"{model_name} CV Score: {np.mean(cv_scores):.4f}")

    return models

def main():
    try:
        print("Loading data...")
        train = pd.read_csv('smoker_status/train.csv')
        test = pd.read_csv('smoker_status/test.csv')

        print("Preprocessing data...")
        X_train, y_train, imputer, scaler, power_transformer, target_encoder = preprocess_features(train)
        X_test, _, _, _, _, _ = preprocess_features(test, is_train=False,
                                                    imputer=imputer, scaler=scaler,
                                                    power_transformer=power_transformer,
                                                    target_encoder=target_encoder)

        print("Initializing H2O...")
        h2o.init(nthreads=-1, max_mem_size="8G")

        train_h2o = h2o.H2OFrame(pd.concat([X_train, pd.Series(y_train, name='smoking')], axis=1))
        test_h2o = h2o.H2OFrame(X_test)
        train_h2o['smoking'] = train_h2o['smoking'].asfactor()

        print("Training H2O AutoML...")
        h2o_automl = H2OAutoML(
            max_models=20,
            seed=42,
            nfolds=5,
            balance_classes=True,
            max_runtime_secs=1800,
            stopping_metric="AUC",
            sort_metric="AUC",
            stopping_rounds=10,
            stopping_tolerance=0.01,
            keep_cross_validation_predictions=True,
            exclude_algos=['DeepLearning']
        )
        h2o_automl.train(y='smoking', training_frame=train_h2o)

        print("Training additional models...")
        additional_models = train_models(X_train, y_train, X_test)

        print("Generating final predictions...")
        h2o_preds = h2o_automl.leader.predict(test_h2o).as_data_frame()['p1']

        model_scores = {
            'h2o': h2o_automl.leader.auc(),
            **{name: info['cv_score'] for name, info in additional_models.items()}
        }
        total_score = sum(model_scores.values())
        weights = {name: score / total_score for name, score in model_scores.items()}

        final_preds = (
            weights['h2o'] * h2o_preds.values +
            weights['lgb'] * additional_models['lgb']['predictions'] +
            weights['xgb'] * additional_models['xgb']['predictions']
        )

        submission = pd.DataFrame({
            'id': test['id'],
            'smoking': final_preds
        })
        submission.to_csv('smoker_status/submission.csv', index=False)
        print("✅ submission.csv written.")

        print("\nModel Weights:")
        for model, weight in weights.items():
            print(f"{model}: {weight:.4f}")

        print("\n📊 Top 3 H2O AutoML Models:")
        leaderboard_df = h2o_automl.leaderboard.as_data_frame()
        print(leaderboard_df[['model_id', 'auc']].head(3))

        print("\n💾 Saving top 3 H2O models locally...")
        model_dir = Path("smoker_status/saved_models")
        model_dir.mkdir(parents=True, exist_ok=True)
        top_models = leaderboard_df['model_id'].head(3).tolist()
        for model_id in top_models:
            model = h2o.get_model(model_id)
            path = h2o.save_model(model=model, path=str(model_dir), force=True)
            print(f"✅ Saved: {model_id} → {path}")

        h2o.cluster().shutdown()
        print("\n✅ Completed successfully!")

    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
        if 'h2o' in locals():
            h2o.cluster().shutdown()
        raise

if __name__ == "__main__":
    main()