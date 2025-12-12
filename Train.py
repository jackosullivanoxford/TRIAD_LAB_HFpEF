"""XGBoost ML Model for HFpEF prediction."""

import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import pandas as pd
import joblib

TARGET = "HFpEF"

PARAM_GRID = {
    "learning_rate": [0.05, 0.06, 0.07],
    "n_estimators": [100, 150, 200],
    "max_depth": [2, 3, 4],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "lambda": [0, 0.1, 1],
    "alpha": [0, 0.1, 1],
}


def train(X, y, n_iter=150, cv=3, random_state=42):
    """Train XGBoost with randomized hyperparameter search."""
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)
    
    search = RandomizedSearchCV(
        xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="auc"),
        PARAM_GRID, n_iter=n_iter, scoring="roc_auc", cv=cv,
        verbose=1, n_jobs=-1, random_state=random_state
    )
    search.fit(X_train, y_train)
    
    print(f"Best params: {search.best_params_}")
    print(f"CV AUC: {search.best_score_:.3f}")
    
    model = search.best_estimator_
    y_pred = model.predict_proba(X_test)[:, 1]
    print(f"Test AUC: {roc_auc_score(y_test, y_pred):.3f}")
    
    return model


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    print(f"Features: {X.shape[1]}, Samples: {len(y)}, Cases: {y.sum()}")
    
    model = train(X, y)
    joblib.dump(model, "model.joblib")
