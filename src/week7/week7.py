import os
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# SHAP
# ============================================================

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

TRACKS_PATH = os.path.join(
    BASE_DIR,
    "src",
    "data",
    "dataset.csv"
)

HISTORY_PATH = os.path.join(
    BASE_DIR,
    "src",
    "data",
    "spotify_history.csv"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "week7"
)

FIGURE_DIR = os.path.join(
    REPORT_DIR,
    "figures"
)

MODEL_DIR = os.path.join(
    REPORT_DIR,
    "models"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# MLFLOW
# ============================================================

MLFLOW_DB = os.path.join(
    BASE_DIR,
    "mlflow.db"
)

mlflow.set_tracking_uri(
    "sqlite:///" + MLFLOW_DB
)

EXPERIMENT_NAME = (
    "PlaylistPulse-Week7-Final-Evaluation"
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PLAYLISTPULSE - WEEK 7")
print("FINAL MODEL EVALUATION AND EXPLAINABILITY")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

tracks = pd.read_csv(
    TRACKS_PATH
)

history = pd.read_csv(
    HISTORY_PATH
)

print(
    f"Tracks shape: {tracks.shape}"
)

print(
    f"History shape: {history.shape}"
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

tracks.columns = (
    tracks.columns
    .str.strip()
)

history.columns = (
    history.columns
    .str.strip()
)


# ============================================================
# TRACK ID
# ============================================================

if "track_id" not in tracks.columns:

    raise ValueError(
        "dataset.csv does not contain track_id."
    )


if "spotify_track_uri" not in history.columns:

    raise ValueError(
        "spotify_history.csv does not contain "
        "spotify_track_uri."
    )


history["track_id"] = (
    history["spotify_track_uri"]
    .astype(str)
    .str.strip()
    .str.replace(
        "spotify:track:",
        "",
        regex=False
    )
)

tracks["track_id"] = (
    tracks["track_id"]
    .astype(str)
    .str.strip()
)

history["track_id"] = (
    history["track_id"]
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE INVALID IDS
# ============================================================

tracks = tracks[
    tracks["track_id"].notna()
    & (tracks["track_id"] != "")
    & (tracks["track_id"] != "nan")
].copy()

history = history[
    history["track_id"].notna()
    & (history["track_id"] != "")
    & (history["track_id"] != "nan")
].copy()


print(
    f"\nValid track rows: {len(tracks)}"
)

print(
    f"Valid history rows: {len(history)}"
)


# ============================================================
# STREAM COUNT
# ============================================================

stream_counts = (
    history
    .groupby("track_id")
    .size()
    .reset_index(
        name="stream_count"
    )
)

print(
    f"Unique history tracks: "
    f"{len(stream_counts)}"
)


# ============================================================
# DEDUPLICATE TRACKS
# ============================================================

tracks_unique = (
    tracks
    .drop_duplicates(
        subset=["track_id"]
    )
    .copy()
)

print(
    f"Unique track records: "
    f"{len(tracks_unique)}"
)


# ============================================================
# MERGE
# ============================================================

data = tracks_unique.merge(
    stream_counts,
    on="track_id",
    how="inner"
)

print(
    f"Matched dataset: {data.shape}"
)


if data.empty:

    raise ValueError(
        "No tracks matched between datasets."
    )


# ============================================================
# TARGET
# ============================================================

data["log_stream_count"] = (
    np.log1p(
        data["stream_count"]
    )
)

data = (
    data
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .dropna(
        subset=["log_stream_count"]
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE REGRESSION DATASET
# ============================================================

data.to_csv(
    os.path.join(
        REPORT_DIR,
        "final_regression_dataset.csv"
    ),
    index=False
)


# ============================================================
# FEATURES
# ============================================================

drop_columns = [
    "track_id",
    "stream_count",
    "log_stream_count"
]

X = data.drop(
    columns=drop_columns
)

y = data[
    "log_stream_count"
]


# ============================================================
# FEATURE TYPES
# ============================================================

numeric_features = (
    X
    .select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    )
    .columns
    .tolist()
)

categorical_features = (
    X
    .select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    )
    .columns
    .tolist()
)


print(
    f"\nNumeric features: "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features: "
    f"{len(categorical_features)}"
)


# ============================================================
# PREPROCESSOR
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
)


print(
    f"\nTraining samples: "
    f"{len(X_train)}"
)

print(
    f"Testing samples: "
    f"{len(X_test)}"
)


# ============================================================
# PREPROCESS
# ============================================================

X_train_processed = (
    preprocessor
    .fit_transform(X_train)
)

X_test_processed = (
    preprocessor
    .transform(X_test)
)


print(
    "\nProcessed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


pd.DataFrame(
    {
        "feature": feature_names
    }
).to_csv(
    os.path.join(
        REPORT_DIR,
        "processed_feature_names.csv"
    ),
    index=False
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Lasso_Week5": Lasso(
        alpha=0.0001,
        max_iter=20000
    ),

    "Decision_Tree_Week6":
        DecisionTreeRegressor(
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),

    "Random_Forest_Week6":
        RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient_Boosting_Week6":
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []

predictions = {}

trained_models = {}


print("\n")
print("=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)


for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    model.fit(
        X_train_processed,
        y_train
    )

    y_pred = model.predict(
        X_test_processed
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append(
        {
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    )

    predictions[
        model_name
    ] = y_pred

    trained_models[
        model_name
    ] = model

    print(
        f"{model_name}: "
        f"MAE={mae:.6f}, "
        f"RMSE={rmse:.6f}, "
        f"R2={r2:.6f}"
    )


# ============================================================
# SORT RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        by="RMSE",
        ascending=True
    )
    .reset_index(drop=True)
)


print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE METRICS
# ============================================================

results_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "final_metrics.csv"
    ),
    index=False
)


# ============================================================
# SELECT FINAL MODEL
# ============================================================

final_model_name = (
    results_df
    .iloc[0]["model"]
)

final_model = trained_models[
    final_model_name
]

final_predictions = predictions[
    final_model_name
]


print(
    f"\nFinal selected model: "
    f"{final_model_name}"
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame(
    {
        "actual_log_stream_count":
            y_test.reset_index(
                drop=True
            ),

        "predicted_log_stream_count":
            final_predictions
    }
)

prediction_results[
    "residual"
] = (
    prediction_results[
        "actual_log_stream_count"
    ]
    -
    prediction_results[
        "predicted_log_stream_count"
    ]
)

prediction_results[
    "absolute_error"
] = np.abs(
    prediction_results[
        "residual"
    ]
)

prediction_results[
    "squared_error"
] = (
    prediction_results[
        "residual"
    ] ** 2
)


prediction_results.to_csv(
    os.path.join(
        REPORT_DIR,
        "final_predictions.csv"
    ),
    index=False
)

prediction_results.to_csv(
    os.path.join(
        REPORT_DIR,
        "residual_analysis.csv"
    ),
    index=False
)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    prediction_results[
        "actual_log_stream_count"
    ],
    prediction_results[
        "predicted_log_stream_count"
    ],
    alpha=0.7
)

minimum = min(
    prediction_results[
        "actual_log_stream_count"
    ].min(),
    prediction_results[
        "predicted_log_stream_count"
    ].min()
)

maximum = max(
    prediction_results[
        "actual_log_stream_count"
    ].max(),
    prediction_results[
        "predicted_log_stream_count"
    ].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "Actual Log Stream Count"
)

plt.ylabel(
    "Predicted Log Stream Count"
)

plt.title(
    "Actual vs Predicted"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "actual_vs_predicted.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RESIDUAL DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    prediction_results[
        "residual"
    ],
    bins=30
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Residual"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Residual Distribution"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "residual_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PREDICTION ERRORS
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    prediction_results[
        "predicted_log_stream_count"
    ],
    prediction_results[
        "residual"
    ],
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Log Stream Count"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Prediction Error Analysis"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "prediction_errors.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if hasattr(
    final_model,
    "coef_"
):

    coefficients = (
        final_model.coef_
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient":
                np.abs(coefficients)
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "absolute_coefficient",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    top_features = (
        importance_df
        .head(20)
        .sort_values(
            "absolute_coefficient"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_features["feature"],
        top_features[
            "absolute_coefficient"
        ]
    )

    plt.xlabel(
        "Absolute Lasso Coefficient"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 20 Lasso Features"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "feature_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


elif hasattr(
    final_model,
    "feature_importances_"
):

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance":
                final_model.feature_importances_
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    top_features = (
        importance_df
        .head(20)
        .sort_values(
            "importance"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 20 Features"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "feature_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

print("\n")
print("=" * 70)
print("SHAP EXPLAINABILITY")
print("=" * 70)


if not SHAP_AVAILABLE:

    print(
        "\nSHAP is not installed."
    )

    print(
        "Run:"
    )

    print(
        "python -m pip install shap"
    )

else:

    print(
        "\nSHAP package detected."
    )

    try:

        shap_sample_size = min(
            342,
            X_test_processed.shape[0]
        )

        X_shap = X_test_processed[
            :shap_sample_size
        ]

        print(
            f"Explaining "
            f"{shap_sample_size} "
            f"test samples..."
        )


        # ====================================================
        # LASSO SHAP
        # ====================================================

        if isinstance(
            final_model,
            Lasso
        ):

            print(
                "Using SHAP LinearExplainer "
                "for Lasso..."
            )

            background_size = min(
                100,
                X_train_processed.shape[0]
            )

            background = (
                X_train_processed[
                    :background_size
                ]
            )

            explainer = (
                shap.LinearExplainer(
                    final_model,
                    background
                )
            )

            shap_values = (
                explainer.shap_values(
                    X_shap
                )
            )

            expected_value = (
                explainer.expected_value
            )

            if np.ndim(
                expected_value
            ) > 0:

                expected_value = (
                    expected_value[0]
                )


        # ====================================================
        # TREE SHAP
        # ====================================================

        else:

            print(
                "Using SHAP TreeExplainer..."
            )

            explainer = (
                shap.TreeExplainer(
                    final_model
                )
            )

            shap_values = (
                explainer.shap_values(
                    X_shap
                )
            )

            expected_value = (
                explainer.expected_value
            )


        # ====================================================
        # NORMALIZE SHAP ARRAY
        # ====================================================

        shap_values = np.asarray(
            shap_values
        )

        if shap_values.ndim == 3:

            shap_values = (
                shap_values[:, :, 0]
            )


        # ====================================================
        # SHAP IMPORTANCE
        # ====================================================

        shap_importance = np.mean(
            np.abs(
                shap_values
            ),
            axis=0
        )

        shap_importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap":
                    shap_importance
            }
        )

        shap_importance_df = (
            shap_importance_df
            .sort_values(
                "mean_abs_shap",
                ascending=False
            )
            .reset_index(drop=True)
        )

        shap_importance_df.to_csv(
            os.path.join(
                REPORT_DIR,
                "shap_feature_importance.csv"
            ),
            index=False
        )


        # ====================================================
        # SHAP BAR PLOT
        # ====================================================

        top_shap = (
            shap_importance_df
            .head(20)
            .sort_values(
                "mean_abs_shap"
            )
        )

        plt.figure(
            figsize=(10, 8)
        )

        plt.barh(
            top_shap["feature"],
            top_shap["mean_abs_shap"]
        )

        plt.xlabel(
            "Mean Absolute SHAP Value"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            "Top 20 SHAP Feature Importances"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURE_DIR,
                "shap_bar.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        # ====================================================
        # SHAP SUMMARY PLOT
        # ====================================================

        plt.figure(
            figsize=(10, 8)
        )

        shap.summary_plot(
            shap_values,
            X_shap,
            feature_names=feature_names,
            max_display=20,
            show=False
        )

        plt.title(
            "SHAP Summary Plot"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURE_DIR,
                "shap_summary.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        # ====================================================
        # SHAP WATERFALL
        # ====================================================

        try:

            if isinstance(
                final_model,
                Lasso
            ):

                explanation = (
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=expected_value,
                        data=X_shap[0],
                        feature_names=feature_names
                    )
                )

                plt.figure(
                    figsize=(10, 8)
                )

                shap.plots.waterfall(
                    explanation,
                    max_display=15,
                    show=False
                )

            else:

                explanation = (
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=expected_value,
                        data=X_shap[0],
                        feature_names=feature_names
                    )
                )

                plt.figure(
                    figsize=(10, 8)
                )

                shap.plots.waterfall(
                    explanation,
                    max_display=15,
                    show=False
                )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    FIGURE_DIR,
                    "shap_waterfall.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        except Exception as waterfall_error:

            print(
                "\nWaterfall plot skipped:"
            )

            print(
                waterfall_error
            )


        print(
            "\nSHAP analysis completed successfully."
        )

        print(
            "SHAP feature importance saved."
        )

        print(
            "SHAP bar plot saved."
        )

        print(
            "SHAP summary plot saved."
        )


    except Exception as shap_error:

        print(
            "\nSHAP analysis failed."
        )

        print(
            f"Reason: {shap_error}"
        )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "final_selected_model.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        final_model,
        file
    )


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "final_preprocessor.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        preprocessor,
        file
    )


# ============================================================
# SAVE FINAL MODEL INFO
# ============================================================

selected_model_df = pd.DataFrame(
    [
        {
            "final_model":
                final_model_name,
            "MAE":
                results_df.iloc[0]["MAE"],
            "RMSE":
                results_df.iloc[0]["RMSE"],
            "R2":
                results_df.iloc[0]["R2"]
        }
    ]
)

selected_model_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "selected_final_model.csv"
    ),
    index=False
)


# ============================================================
# MLFLOW
# ============================================================

with mlflow.start_run(
    run_name="Final_Model_Evaluation"
):

    mlflow.log_param(
        "final_model",
        final_model_name
    )

    mlflow.log_param(
        "train_samples",
        len(X_train)
    )

    mlflow.log_param(
        "test_samples",
        len(X_test)
    )

    mlflow.log_param(
        "processed_features",
        X_train_processed.shape[1]
    )

    mlflow.log_metric(
        "MAE",
        float(
            results_df.iloc[0]["MAE"]
        )
    )

    mlflow.log_metric(
        "RMSE",
        float(
            results_df.iloc[0]["RMSE"]
        )
    )

    mlflow.log_metric(
        "R2",
        float(
            results_df.iloc[0]["R2"]
        )
    )

    mlflow.sklearn.log_model(
        final_model,
        "final_model"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("WEEK 7 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nFinal selected model: "
    f"{final_model_name}"
)

print(
    f"MAE: "
    f"{results_df.iloc[0]['MAE']:.6f}"
)

print(
    f"RMSE: "
    f"{results_df.iloc[0]['RMSE']:.6f}"
)

print(
    f"R2: "
    f"{results_df.iloc[0]['R2']:.6f}"
)

print(
    "\nReports:"
)

print(
    REPORT_DIR
)

print(
    "\nFigures:"
)

print(
    FIGURE_DIR
)

print(
    "\nModels:"
)

print(
    MODEL_DIR
)

print(
    "\nMLflow experiment:"
)

print(
    EXPERIMENT_NAME
)

print(
    "\nWeek 7 final evaluation and explainability completed."
)
