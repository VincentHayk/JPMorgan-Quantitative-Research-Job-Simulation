"""Task 3 - Credit default modeling and expected-loss estimation.

The script benchmarks several classification models and converts estimated
default probabilities into expected credit losses.

Run from the repository root with:
    python -m src.task3_credit_risk
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Loan_Data.csv"
OUTPUT_DIR = ROOT / "outputs"

FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]

TARGET = "default"


def load_loan_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the credit-risk dataset."""
    data = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data


def build_models(random_state: int = 42):
    """Return the model benchmark used in the cleaned repository version."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting (XGBoost)": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Neural Network (MLP)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(20, 10),
                        max_iter=1_000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def specificity_score(y_true, y_pred) -> float:
    """True negative rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def evaluate_models(
    data: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Train and evaluate the model benchmark on a stratified holdout set."""
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    rows = []
    fitted_models = {}

    for name, model in build_models(random_state).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_default": precision_score(y_test, y_pred),
                "recall_default": recall_score(y_test, y_pred),
                "specificity": specificity_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
        )
        fitted_models[name] = model

    results = (
        pd.DataFrame(rows)
        .sort_values(["roc_auc", "accuracy"], ascending=False)
        .reset_index(drop=True)
    )

    return results, fitted_models, X_test, y_test


def default_probability(model, client: pd.DataFrame) -> float:
    """Estimated probability that a client defaults."""
    return float(model.predict_proba(client[FEATURES])[0, 1])


def expected_loss(
    model,
    client: pd.DataFrame,
    recovery_rate: float = 0.10,
) -> float:
    """Expected loss = PD x EAD x LGD."""
    pd_estimate = default_probability(model, client)
    ead = float(client["loan_amt_outstanding"].iloc[0])
    lgd = 1.0 - recovery_rate
    return pd_estimate * ead * lgd


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = load_loan_data()
    results, fitted_models, _, _ = evaluate_models(data)
    results.to_csv(OUTPUT_DIR / "credit_model_comparison.csv", index=False)

    print("Credit model comparison:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Client example retained from the original simulation work.
    client = pd.DataFrame(
        [
            {
                "credit_lines_outstanding": 3,
                "loan_amt_outstanding": 7153.76123,
                "total_debt_outstanding": 21065.06124,
                "income": 113071.52470,
                "years_employed": 4,
                "fico_score": 583,
            }
        ]
    )

    client_rows = []
    for name, model in fitted_models.items():
        client_rows.append(
            {
                "model": name,
                "default_probability": default_probability(model, client),
                "expected_loss_at_10pct_recovery": expected_loss(
                    model,
                    client,
                    recovery_rate=0.10,
                ),
            }
        )

    client_results = pd.DataFrame(client_rows)
    client_results.to_csv(
        OUTPUT_DIR / "example_client_credit_estimates.csv",
        index=False,
    )

    print("\nExample client estimates:")
    print(
        client_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()
