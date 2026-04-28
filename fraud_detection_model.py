from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except ImportError:
    RandomUnderSampler = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False


DATA_PATH = Path(__file__).with_name("creditcard.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    if "Class" not in data.columns:
        raise ValueError("Expected a 'Class' column in the dataset.")
    features = data.drop(columns=["Class"])
    target = data["Class"].astype(int)
    return features, target


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
    )


def evaluate_model(name: str, model, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    predictions = model.predict(x_test)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print(f"\n{name}")
    print("=" * len(name))
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Classification report:")
    print(classification_report(y_test, predictions, digits=4, zero_division=0))


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    x, y = load_data(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Class distribution")
    print("===================")
    print("Train:")
    print(y_train.value_counts(normalize=True).rename("proportion"))
    print("Test:")
    print(y_test.value_counts(normalize=True).rename("proportion"))

    feature_columns = list(x.columns)
    preprocessor = build_preprocessor(feature_columns)

    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    logistic_regression.fit(x_train, y_train)
    evaluate_model("Logistic Regression (class_weight='balanced')", logistic_regression, x_test, y_test)

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    random_forest.fit(x_train, y_train)
    evaluate_model("Random Forest (class_weight='balanced_subsample')", random_forest, x_test, y_test)

    if IMBLEARN_AVAILABLE:
        undersampled_logistic = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("undersampler", RandomUnderSampler(sampling_strategy=0.1, random_state=RANDOM_STATE)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        undersampled_logistic.fit(x_train, y_train)
        evaluate_model("Logistic Regression + RandomUnderSampler", undersampled_logistic, x_test, y_test)
    else:
        print("\nimblearn is not installed, so undersampling was skipped.")
        print("Install it with: pip install imbalanced-learn")


if __name__ == "__main__":
    main()


