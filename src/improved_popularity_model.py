
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "src" / "data" / "dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "reports" / "improved_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "popularity"

NUMERIC_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "explicit",
    "key",
    "mode",
    "time_signature",
]

CATEGORICAL_FEATURES = ["track_genre"]


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # IMPORTANT FIX:
    # sparse_output=False produces dense data.
    # HistGradientBoostingRegressor requires dense input.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("=" * 65)
    print("PLAYLISTPULSE - IMPROVED POPULARITY PREDICTION")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Check that dataset.csv is inside src/data."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    )
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required dataset columns: {missing_columns}"
        )

    # Keep only the columns required by this experiment.
    df = df[required_columns].copy()

    # Convert popularity and numeric features safely.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    for column in NUMERIC_FEATURES:
        if column == "explicit":
            # Supports Boolean values and strings such as True/False.
            df[column] = df[column].replace({
                True: 1,
                False: 0,
                "True": 1,
                "False": 0,
                "true": 1,
                "false": 0,
            })

        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["track_genre"] = (
        df["track_genre"].astype("string").fillna("unknown").astype(str)
    )

    # Popularity cannot be imputed as a target.
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid popularity values were found.")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Hold out test data. Do not use test results to tune models.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"\nDataset rows: {len(df):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Features: {', '.join(NUMERIC_FEATURES + CATEGORICAL_FEATURES)}")
    print("\nUsing dense one-hot encoded data to avoid the sparse-input error.")

    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    preprocessor = build_preprocessor()

    # Small, practical search to reduce training time.
    experiments = {
        "Ridge Regression": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("model", Ridge()),
            ]),
            "params": {
                "model__alpha": [1.0, 10.0, 50.0],
            },
        },
        "HistGradientBoosting": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        random_state=42,
                        early_stopping=True,
                    ),
                ),
            ]),
            "params": {
                "model__max_iter": [100, 150],
                "model__learning_rate": [0.08],
                "model__max_leaf_nodes": [15],
            },
        },
    }

    results = []
    prediction_frames = []
    fitted_models = {}

    for model_name, experiment in experiments.items():
        print("\n" + "-" * 65)
        print(f"Training: {model_name}")
        print("Please wait...")

        search = GridSearchCV(
            estimator=experiment["pipeline"],
            param_grid=experiment["params"],
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=1,
            verbose=1,
            refit=True,
            error_score="raise",
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)

        metrics = calculate_metrics(y_test, predictions)

        result = {
            "Model": model_name,
            "CV_MAE": -search.best_score_,
            "Test_MAE": metrics["MAE"],
            "Test_RMSE": metrics["RMSE"],
            "Test_R2": metrics["R2"],
            "Best_Parameters": str(search.best_params_),
        }
        results.append(result)
        fitted_models[model_name] = best_model

        prediction_frames.append(pd.DataFrame({
            "actual_popularity": y_test.to_numpy(),
            "predicted_popularity": predictions,
            "model": model_name,
        }))

        # Save after every completed model so progress is not lost.
        pd.DataFrame(results).to_csv(
            OUTPUT_DIR / "popularity_model_comparison.csv",
            index=False,
        )

        pd.concat(prediction_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "popularity_model_predictions.csv",
            index=False,
        )

        print(f"\nModel: {model_name}")
        print(f"Best CV MAE: {-search.best_score_:.4f}")
        print(f"Test MAE: {metrics['MAE']:.4f}")
        print(f"Test RMSE: {metrics['RMSE']:.4f}")
        print(f"Test R²: {metrics['R2']:.4f}")
        print(f"Best parameters: {search.best_params_}")
        print("Progress saved.")

    # Select using cross-validation MAE, not test MAE.
    best_result = min(results, key=lambda row: row["CV_MAE"])
    best_name = best_result["Model"]
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)

    pd.DataFrame({
        "actual_popularity": y_test.to_numpy(),
        "predicted_popularity": best_predictions,
        "absolute_error": np.abs(
            y_test.to_numpy() - best_predictions
        ),
    }).to_csv(
        OUTPUT_DIR / "best_popularity_predictions.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)
    print(
        pd.DataFrame(results)[
            ["Model", "CV_MAE", "Test_MAE", "Test_RMSE", "Test_R2"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    print(f"\nBest model by CV MAE: {best_name}")
    print(f"Results folder: {OUTPUT_DIR}")
    print("Files saved:")
    print("1. popularity_model_comparison.csv")
    print("2. popularity_model_predictions.csv")
    print("3. best_popularity_predictions.csv")
    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
 
 