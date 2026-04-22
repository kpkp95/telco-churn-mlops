import optuna
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1",
    n_trials: int = 40,
    random_state: int = 42
) -> dict:
    """
    Tune an XGBoost model using Optuna.

    Args:
        X: Feature matrix.
        y: Target values.
        scoring: Metric to optimize. Use "f1" or "recall".
        n_trials: Number of Optuna trials.
        random_state: Random seed.

    Returns:
        Best hyperparameters found by Optuna.
    """

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=random_state
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        model = XGBClassifier(**params)

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best score:", study.best_value)
    print("Best params:", study.best_params)

    best_params = study.best_params

    best_params.update({
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "n_jobs": -1,
        "eval_metric": "logloss",
    })

    return best_params

def tune_model1(X, y):
    """
    Tunes an XGBoost model using Optuna.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="recall")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    print("Best Params:", study.best_params)
    return study.best_params