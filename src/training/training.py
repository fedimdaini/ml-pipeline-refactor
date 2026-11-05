"""
Training utilities: split/scale, train, grid search, predict/evaluate.
Replaces the repeated train_test_split + StandardScaler + fit/predict blocks
from every Flask route.
"""
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from src.metrics.metrics import compute_classification_metrics, compute_regression_metrics


def split_and_scale(X, y, test_size=0.2, random_state=42, stratify=False):
    """
    Split data and scale features with StandardScaler.

    Replaces the ~8× copy-pasted train_test_split + scaler blocks.

    stratify=True keeps class proportions in both splits; use it for
    classification only (a continuous regression target cannot be stratified).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(model, X_train, y_train):
    """
    Simple wrapper around model.fit().

    Parameters
    ----------
    model : sklearn estimator
    X_train : array-like
    y_train : array-like

    Returns
    -------
    fitted_model
    """
    model.fit(X_train, y_train)
    return model


def train_with_gridsearch(model, param_grid, X_train, y_train, cv=5, n_jobs=-1):
    """
    Run GridSearchCV and return the best model + params.

    Replaces the ~4× copy-pasted GridSearchCV blocks.
    """
    grid = GridSearchCV(model, param_grid, cv=cv, n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def predict_and_evaluate(model, X_test, y_test, problem_type):
    """
    Generate predictions and compute metrics.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test : array-like
    y_test : array-like
    problem_type : str
        "classification" or "regression"

    Returns
    -------
    tuple
        (y_pred, metrics_dict)
    """
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        metrics = compute_classification_metrics(y_test, y_pred)
    elif problem_type == "regression":
        metrics = compute_regression_metrics(y_test, y_pred)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    return y_pred, metrics