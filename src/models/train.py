import pandas as pd
from typing import Tuple
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def split_data(
    df: pd.DataFrame,
    target_col: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the feature-engineered dataset into training and testing sets.

    Args:
        df: Feature-engineered dataframe.
        target_col: Name of the target column.
        test_size: Percentage of data used for testing.
        random_state: Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> XGBClassifier:
    """
    Train an XGBoost classifier for Telco churn prediction.

    Args:
        X_train: Training features.
        y_train: Training target.
        random_state: Random seed for reproducibility.

    Returns:
        Trained XGBoost model.
    """

    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    return model


def train_model(
    df: pd.DataFrame,
    target_col: str = "Churn"
):
    """
    Complete training function.

    Steps:
    - Split data into train/test sets
    - Train XGBoost model
    - Return model and data needed for evaluation

    Args:
        df: Feature-engineered dataframe.
        target_col: Target column name.

    Returns:
        model, X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = split_data(
        df=df,
        target_col=target_col
    )

    model = train_xgboost_model(
        X_train=X_train,
        y_train=y_train
    )

    print("XGBoost model training completed.")
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")

    return model, X_train, X_test, y_train, y_test




def train_model1(df: pd.DataFrame, target_col: str = "Churn"):
    """
    Trains an XGBoost model and logs results with MLflow.

    Args:
        df: Feature-engineered dataset.
        target_col: Name of the target column.

    Returns:
        model, X_test, y_test
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.3).astype(int)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, pos_label=1)
        recall = recall_score(y_test, preds, pos_label=1)
        f1 = f1_score(y_test, preds, pos_label=1)
        roc_auc = roc_auc_score(y_test, proba)

        mlflow.log_param("n_estimators", 500)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("subsample", 0.8)
        mlflow.log_param("colsample_bytree", 0.8)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("threshold", 0.3)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.xgboost.log_model(model, "model")

        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained.")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

    return model, X_test, y_test, proba, preds