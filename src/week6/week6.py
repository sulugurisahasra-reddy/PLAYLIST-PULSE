import os
import warnings
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
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
    "week6"
)

FIGURE_DIR = os.path.join(
    REPORT_DIR,
    "figures"
)

MODEL_DIR = os.path.join(
    REPORT_DIR,
    "models"
)

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


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
    "PlaylistPulse-Week6-Tree-Ensemble-Models"
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PLAYLISTPULSE - WEEK 6")
print("TREE AND ENSEMBLE MODELS")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

tracks = pd.read_csv(TRACKS_PATH)
history = pd.read_csv(HISTORY_PATH)

print(f"Tracks shape: {tracks.shape}")
print(f"History shape: {history.shape}")


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
# IDENTIFY TRACK COLUMN
# ============================================================

print("\nDataset columns:")
print("Tracks:", tracks.columns.tolist())
print("History:", history.columns.tolist())


track_id_candidates = [
    "track_id",
    "spotify_track_id",
    "spotify_track_uri",
    "id"
]

track_id_col = None

for column in track_id_candidates:
    if column in tracks.columns:
        track_id_col = column
        break

if track_id_col is None:
    raise ValueError(
        "Could not identify the track identifier "
        "in dataset.csv"
    )


if "spotify_track_uri" in history.columns:

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

    history_id_col = "track_id"

elif "track_id" in history.columns:

    history["track_id"] = (
        history["track_id"]
        .astype(str)
        .str.strip()
    )

    history_id_col = "track_id"

elif "spotify_track_id" in history.columns:

    history["track_id"] = (
        history["spotify_track_id"]
        .astype(str)
        .str.strip()
    )

    history_id_col = "track_id"

else:

    raise ValueError(
        "Could not identify the track identifier "
        "in spotify_history.csv"
    )


# ============================================================
# STANDARDIZE TRACK IDENTIFIERS
# ============================================================

tracks[track_id_col] = (
    tracks[track_id_col]
    .astype(str)
    .str.strip()
)

if track_id_col == "spotify_track_uri":

    tracks["track_id"] = (
        tracks[track_id_col]
        .astype(str)
        .str.strip()
        .str.replace(
            "spotify:track:",
            "",
            regex=False
        )
    )

    track_id_col = "track_id"

else:

    tracks["track_id"] = (
        tracks[track_id_col]
        .astype(str)
        .str.strip()
    )

    track_id_col = "track_id"


print(
    f"\nStandardized track identifier: "
    f"{track_id_col}"
)


# ============================================================
# REMOVE INVALID IDS
# ============================================================

tracks = tracks[
    tracks[track_id_col].notna()
    & (tracks[track_id_col] != "")
    & (tracks[track_id_col] != "nan")
].copy()

history = history[
    history[history_id_col].notna()
    & (history[history_id_col] != "")
    & (history[history_id_col] != "nan")
].copy()


print(
    f"Valid track rows: {len(tracks)}"
)

print(
    f"Valid history rows: {len(history)}"
)


# ============================================================
# STREAM COUNT
# ============================================================

print("\nCalculating stream counts...")

stream_counts = (
    history
    .groupby(history_id_col)
    .size()
    .reset_index(
        name="stream_count"
    )
)

stream_counts = stream_counts.rename(
    columns={
        history_id_col: track_id_col
    }
)

print(
    f"Unique history tracks: "
    f"{len(stream_counts)}"
)


# ============================================================
# DEDUPLICATE TRACK DATA
# ============================================================

tracks_unique = (
    tracks
    .drop_duplicates(
        subset=[track_id_col]
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

regression_data = tracks_unique.merge(
    stream_counts,
    on=track_id_col,
    how="inner"
)

print(
    f"\nMatched regression dataset: "
    f"{regression_data.shape}"
)


if regression_data.empty:

    raise ValueError(
        "No matching tracks were found between "
        "dataset.csv and spotify_history.csv."
    )


# ============================================================
# TARGET
# ============================================================

regression_data["log_stream_count"] = (
    np.log1p(
        regression_data["stream_count"]
    )
)

regression_data = (
    regression_data
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .dropna(
        subset=["log_stream_count"]
    )
    .reset_index(drop=True)
)


target = "log_stream_count"


# ============================================================
# SAVE REGRESSION DATASET
# ============================================================

dataset_path = os.path.join(
    REPORT_DIR,
    "regression_dataset.csv"
)

regression_data.to_csv(
    dataset_path,
    index=False
)


# ============================================================
# CREATE FEATURES
# ============================================================

drop_columns = [
    target,
    "stream_count",
    track_id_col
]

feature_data = regression_data.drop(
    columns=[
        column
        for column in drop_columns
        if column in regression_data.columns
    ]
)


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = (
    feature_data
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
    feature_data
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
# PREPROCESSING
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

X = feature_data
y = regression_data[target]


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


# ============================================================
# MODELS
# ============================================================

models = {

    "Decision Tree": DecisionTreeRegressor(
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []
predictions = {}


print("\n")
print("=" * 70)
print("TRAINING TREE AND ENSEMBLE MODELS")
print("=" * 70)


for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    with mlflow.start_run(
        run_name=model_name.replace(
            " ",
            "_"
        )
    ):

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

        mlflow.log_param(
            "model_type",
            model_name
        )

        for key, value in model.get_params().items():

            if (
                isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool
                    )
                )
                or value is None
            ):

                try:

                    mlflow.log_param(
                        key,
                        value
                    )

                except Exception:
                    pass

        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "RMSE",
            rmse
        )

        mlflow.log_metric(
            "R2",
            r2
        )

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        print(
            f"{model_name}: "
            f"MAE={mae:.6f}, "
            f"RMSE={rmse:.6f}, "
            f"R2={r2:.6f}"
        )


# ============================================================
# RESULTS
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
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    REPORT_DIR,
    "tree_ensemble_metrics.csv"
)

results_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "actual_log_stream_count":
            y_test.reset_index(
                drop=True
            )
    }
)


for model_name, prediction in predictions.items():

    column_name = (
        model_name
        .lower()
        .replace(
            " ",
            "_"
        )
        + "_prediction"
    )

    prediction_df[
        column_name
    ] = prediction


prediction_path = os.path.join(
    REPORT_DIR,
    "tree_ensemble_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

preprocessor_path = os.path.join(
    MODEL_DIR,
    "tree_ensemble_preprocessor.pkl"
)

with open(
    preprocessor_path,
    "wb"
) as file:

    pickle.dump(
        preprocessor,
        file
    )


# ============================================================
# SAVE MODELS
# ============================================================

for model_name, model in models.items():

    filename = (
        model_name
        .lower()
        .replace(
            " ",
            "_"
        )
        + "_model.pkl"
    )

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    with open(
        model_path,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )


# ============================================================
# BEST MODEL
# ============================================================

best_model_name = (
    results_df
    .iloc[0]["model"]
)

best_model = models[
    best_model_name
]

best_row = (
    results_df
    .iloc[0]
)


print(
    f"\nBest model based on RMSE: "
    f"{best_model_name}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if hasattr(
    best_model,
    "feature_importances_"
):

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance":
                best_model.feature_importances_
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_path = os.path.join(
        REPORT_DIR,
        "feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
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
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Top 20 Feature Importances - "
        f"{best_model_name}"
    )

    plt.tight_layout()

    importance_plot_path = os.path.join(
        FIGURE_DIR,
        "feature_importance.png"
    )

    plt.savefig(
        importance_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAE PLOT
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    results_df["model"],
    results_df["MAE"]
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "MAE"
)

plt.title(
    "Tree and Ensemble Models - MAE"
)

plt.tight_layout()

mae_plot_path = os.path.join(
    FIGURE_DIR,
    "model_comparison_mae.png"
)

plt.savefig(
    mae_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RMSE PLOT
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    results_df["model"],
    results_df["RMSE"]
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "RMSE"
)

plt.title(
    "Tree and Ensemble Models - RMSE"
)

plt.tight_layout()

rmse_plot_path = os.path.join(
    FIGURE_DIR,
    "model_comparison_rmse.png"
)

plt.savefig(
    rmse_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# R2 PLOT
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    results_df["model"],
    results_df["R2"]
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "R²"
)

plt.title(
    "Tree and Ensemble Models - R²"
)

plt.tight_layout()

r2_plot_path = os.path.join(
    FIGURE_DIR,
    "model_comparison_r2.png"
)

plt.savefig(
    r2_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# MODEL SELECTION
# ============================================================

selection_df = pd.DataFrame(
    [
        {
            "selected_model":
                best_model_name,
            "MAE":
                best_row["MAE"],
            "RMSE":
                best_row["RMSE"],
            "R2":
                best_row["R2"]
        }
    ]
)

selection_path = os.path.join(
    REPORT_DIR,
    "selected_model.csv"
)

selection_df.to_csv(
    selection_path,
    index=False
)


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names_path = os.path.join(
    REPORT_DIR,
    "processed_feature_names.csv"
)

pd.DataFrame(
    {
        "feature": feature_names
    }
).to_csv(
    feature_names_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("WEEK 6 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nBest model: "
    f"{best_model_name}"
)

print(
    f"MAE  : "
    f"{best_row['MAE']:.6f}"
)

print(
    f"RMSE : "
    f"{best_row['RMSE']:.6f}"
)

print(
    f"R2   : "
    f"{best_row['R2']:.6f}"
)

print("\nGenerated reports:")
print(
    f"  {REPORT_DIR}"
)

print("\nGenerated figures:")
print(
    f"  {FIGURE_DIR}"
)

print("\nGenerated models:")
print(
    f"  {MODEL_DIR}"
)

print("\nMLflow experiment:")
print(
    EXPERIMENT_NAME
)

print(
    "\nWeek 6 is ready for verification."
)