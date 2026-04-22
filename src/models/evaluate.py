import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.3
) -> dict:
    """
    Evaluate a trained classification model.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True test labels.
        threshold: Probability threshold for predicting churn.

    Returns:
        Dictionary containing evaluation metrics.
    """

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, pos_label=1)
    recall = recall_score(y_test, preds, pos_label=1)
    f1 = f1_score(y_test, preds, pos_label=1)
    roc_auc = roc_auc_score(y_test, proba)

    report = classification_report(y_test, preds, digits=3)
    matrix = confusion_matrix(y_test, preds)

    print("Classification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)

    print("Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "threshold": threshold,
    }

    return metrics


def evaluate_model1(model, X_test, y_test):
    """
    Evaluates an XGBoost model on test data.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
    """
    preds = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))