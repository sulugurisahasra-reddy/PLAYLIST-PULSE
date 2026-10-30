import os
import warnings
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


# ============================================================
# PLAYLIST PULSE - WEEK 3
# EXPECTED STREAM COUNT REGRESSION
# OLS + MANUAL GRADIENT DESCENT + MLFLOW
# ============================================================


# ============================================================
# 1. PROJECT PATHS
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
    "week3"
)

MODEL_DIR = os.path.join(
    REPORT_DIR,
    "models"
)

MLRUNS_DIR = os.path.join(
    BASE_DIR,
    "mlruns"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 70)
print("PLAYLIST PULSE - WEEK 3")
print("EXPECTED STREAM COUNT REGRESSION")
print("OLS + MANUAL GRADIENT DESCENT + MLFLOW")
print("=" * 70)


# ============================================================
# 3. CHECK FILES
# ============================================================

print("\nChecking required files...")

if not os.path.exists(TRACKS_PATH):
    raise FileNotFoundError(
        f"Track dataset not found:\n{TRACKS_PATH}"
    )

if not os.path.exists(HISTORY_PATH):
    raise FileNotFoundError(
        f"Streaming history not found:\n{HISTORY_PATH}"
    )

print("Track dataset found.")
print("Streaming history found.")


# ============================================================
# 4. LOAD TRACK DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING SPOTIFY TRACK DATASET")
print("=" * 70)

tracks_df = pd.read_csv(
    TRACKS_PATH
)

print(
    "Tracks Dataset Shape:",
    tracks_df.shape
)

print("\nTrack Dataset Columns:")

print(
    tracks_df.columns.tolist()
)


# ============================================================
# 5. LOAD STREAMING HISTORY
# ============================================================

print("\n" + "=" * 70)
print("LOADING STREAMING HISTORY")
print("=" * 70)

history_df = pd.read_csv(
    HISTORY_PATH
)

print(
    "Streaming History Shape:",
    history_df.shape
)

print("\nHistory Columns:")

print(
    history_df.columns.tolist()
)


# ============================================================
# 6. VALIDATE TRACK DATASET
# ============================================================

required_track_columns = [
    "track_id",
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "track_genre"
]

missing_track_columns = [
    column
    for column in required_track_columns
    if column not in tracks_df.columns
]

if missing_track_columns:

    raise ValueError(
        "The following required columns are missing "
        "from dataset.csv:\n"
        + str(missing_track_columns)
    )


# ============================================================
# 7. VALIDATE STREAMING HISTORY
# ============================================================

if "spotify_track_uri" not in history_df.columns:

    raise ValueError(
        "spotify_track_uri column is missing "
        "from spotify_history.csv"
    )


# ============================================================
# 8. CLEAN TRACK IDs
# ============================================================

print("\n" + "=" * 70)
print("CLEANING TRACK IDS")
print("=" * 70)

tracks_df["track_id"] = (
    tracks_df["track_id"]
    .astype("string")
    .str.strip()
)

history_df["spotify_track_uri"] = (
    history_df["spotify_track_uri"]
    .astype("string")
    .str.strip()
)


# ============================================================
# 9. REMOVE INVALID TRACK IDS
# ============================================================

tracks_df = tracks_df[
    tracks_df["track_id"].notna()
].copy()

history_df = history_df[
    history_df["spotify_track_uri"].notna()
].copy()

tracks_df = tracks_df[
    tracks_df["track_id"] != ""
].copy()

history_df = history_df[
    history_df["spotify_track_uri"] != ""
].copy()


print(
    "Valid track rows:",
    len(tracks_df)
)

print(
    "Valid history rows:",
    len(history_df)
)


# ============================================================
# 10. CREATE TRACK ID IN HISTORY
# ============================================================
#
# IMPORTANT:
#
# Your spotify_history.csv already stores the Spotify
# track ID directly.
#
# Example:
#
# 2J3n32GeLmMjwuAzyhcSNe
#
# There is no "spotify:track:" prefix.
#
# Therefore we simply copy it.
# ============================================================

print("\nUsing spotify_track_uri directly as track_id...")

history_df["track_id"] = (
    history_df["spotify_track_uri"]
    .astype("string")
    .str.strip()
)


# ============================================================
# 11. DISPLAY SAMPLE IDS
# ============================================================

print("\nSample track IDs from tracks dataset:")

print(
    tracks_df["track_id"]
    .head(5)
    .to_string(index=False)
)

print("\nSample track IDs from streaming history:")

print(
    history_df["track_id"]
    .head(5)
    .to_string(index=False)
)


# ============================================================
# 12. FIND OVERLAP BEFORE GROUPING
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DATASET OVERLAP")
print("=" * 70)

track_ids = set(
    tracks_df["track_id"]
)

history_track_ids = set(
    history_df["track_id"]
)

matched_track_ids = (
    track_ids.intersection(
        history_track_ids
    )
)

print(
    "Unique track IDs in dataset.csv:",
    len(track_ids)
)

print(
    "Unique track IDs in spotify_history.csv:",
    len(history_track_ids)
)

print(
    "Matched track IDs:",
    len(matched_track_ids)
)


if len(matched_track_ids) == 0:

    print(
        "\nERROR: No track IDs matched."
    )

    print(
        "\nFirst track dataset IDs:"
    )

    print(
        list(track_ids)[:10]
    )

    print(
        "\nFirst history IDs:"
    )

    print(
        list(history_track_ids)[:10]
    )

    raise ValueError(
        "No matching track IDs were found between "
        "dataset.csv and spotify_history.csv."
    )


# ============================================================
# 13. CREATE STREAM COUNT
# ============================================================

print("\n" + "=" * 70)
print("CREATING STREAM COUNT")
print("=" * 70)

print(
    "\nCounting listening events per track..."
)

stream_counts = (
    history_df
    .groupby("track_id")
    .size()
    .reset_index(
        name="stream_count"
    )
)


print(
    "Unique tracks with streaming history:",
    len(stream_counts)
)

print(
    "Total listening events:",
    stream_counts["stream_count"].sum()
)


print("\nStream count summary:")

print(
    stream_counts[
        "stream_count"
    ].describe()
)


# ============================================================
# 14. MERGE TRACK FEATURES WITH STREAM COUNT
# ============================================================

print("\n" + "=" * 70)
print("MERGING DATASETS")
print("=" * 70)

df = tracks_df.merge(
    stream_counts,
    on="track_id",
    how="inner"
)


print(
    "Merged Dataset Shape:",
    df.shape
)

print(
    "Matched Tracks:",
    len(df)
)


# ============================================================
# 15. VALIDATE MERGED DATASET
# ============================================================

if len(df) == 0:

    raise ValueError(
        "The merge produced zero rows. "
        "Track IDs are not matching correctly."
    )


print(
    "\nMinimum Stream Count:",
    df["stream_count"].min()
)

print(
    "Maximum Stream Count:",
    df["stream_count"].max()
)

print(
    "Average Stream Count:",
    df["stream_count"].mean()
)

print(
    "Median Stream Count:",
    df["stream_count"].median()
)


# ============================================================
# 16. CREATE LOG STREAM COUNT
# ============================================================

print("\n" + "=" * 70)
print("CREATING REGRESSION TARGET")
print("=" * 70)

print(
    "\nApplying log transformation:"
)

print(
    "log_stream_count = log(1 + stream_count)"
)

df["log_stream_count"] = np.log1p(
    df["stream_count"]
)


print(
    "Target created successfully."
)


# ============================================================
# 17. CLEAN NUMERIC VALUES
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df[
    df["log_stream_count"].notna()
].copy()


# ============================================================
# 18. DEFINE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE PREPARATION")
print("=" * 70)


numeric_features = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature"
]


categorical_features = [
    "track_genre"
]


# ============================================================
# 19. EXCLUDE LEAKAGE
# ============================================================

print(
    "\nExcluded columns:"
)

print(
    " - track_id"
)

print(
    " - stream_count"
)

print(
    " - log_stream_count"
)

print(
    " - popularity"
)


# Popularity is excluded because it can represent
# an existing popularity measurement and may encode
# information related to listening behavior.


X = df[
    numeric_features +
    categorical_features
].copy()

y = df[
    "log_stream_count"
].astype(float)


print(
    "\nNumber of samples:",
    len(X)
)

print(
    "Number of raw predictor columns:",
    X.shape[1]
)


if len(X) < 10:

    raise ValueError(
        "Too few matched samples for regression."
    )


# ============================================================
# 20. TRAIN TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 21. PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)


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
    ]
)


print(
    "Fitting preprocessing pipeline..."
)


X_train_processed = (
    preprocessor.fit_transform(
        X_train
    )
)

X_test_processed = (
    preprocessor.transform(
        X_test
    )
)


X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float64
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float64
)


print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# 22. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


# ============================================================
# 23. OLS REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1 - ORDINARY LEAST SQUARES")
print("=" * 70)

ols_model = LinearRegression()


print(
    "\nTraining OLS model..."
)


ols_model.fit(
    X_train_processed,
    y_train
)


ols_predictions = (
    ols_model.predict(
        X_test_processed
    )
)


ols_mae, ols_rmse, ols_r2 = (
    calculate_metrics(
        y_test,
        ols_predictions
    )
)


print(
    "\nOLS Results:"
)

print(
    f"MAE  : {ols_mae:.6f}"
)

print(
    f"RMSE : {ols_rmse:.6f}"
)

print(
    f"R²   : {ols_r2:.6f}"
)


# ============================================================
# 24. SAVE OLS MODEL
# ============================================================

joblib.dump(
    ols_model,
    os.path.join(
        MODEL_DIR,
        "ols_model.pkl"
    )
)


joblib.dump(
    preprocessor,
    os.path.join(
        MODEL_DIR,
        "preprocessor.pkl"
    )
)


# ============================================================
# 25. MANUAL GRADIENT DESCENT
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2 - MANUAL GRADIENT DESCENT")
print("=" * 70)


def gradient_descent(
    X,
    y,
    learning_rate=0.01,
    epochs=3000
):

    n_samples = X.shape[0]

    n_features = X.shape[1]

    weights = np.zeros(
        n_features,
        dtype=np.float64
    )

    loss_history = []


    for epoch in range(
        epochs
    ):

        predictions = (
            X @ weights
        )

        errors = (
            predictions - y
        )


        gradients = (
            2.0 / n_samples
        ) * (
            X.T @ errors
        )


        weights = (
            weights -
            learning_rate * gradients
        )


        mse = np.mean(
            errors ** 2
        )


        loss_history.append(
            mse
        )


        if (
            epoch == 0
            or (epoch + 1) % 500 == 0
        ):

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- MSE: {mse:.6f}"
            )


    return (
        weights,
        loss_history
    )


# ============================================================
# 26. ADD INTERCEPT
# ============================================================

X_train_bias = np.c_[
    np.ones(
        X_train_processed.shape[0]
    ),
    X_train_processed
]


X_test_bias = np.c_[
    np.ones(
        X_test_processed.shape[0]
    ),
    X_test_processed
]


print(
    "\nTraining manual Gradient Descent..."
)


# ============================================================
# 27. TRAIN GRADIENT DESCENT
# ============================================================

gd_weights, loss_history = (
    gradient_descent(
        X_train_bias,
        y_train.to_numpy(),
        learning_rate=0.01,
        epochs=3000
    )
)


# ============================================================
# 28. GRADIENT DESCENT PREDICTIONS
# ============================================================

gd_predictions = (
    X_test_bias @ gd_weights
)


gd_mae, gd_rmse, gd_r2 = (
    calculate_metrics(
        y_test,
        gd_predictions
    )
)


print(
    "\nGradient Descent Results:"
)

print(
    f"MAE  : {gd_mae:.6f}"
)

print(
    f"RMSE : {gd_rmse:.6f}"
)

print(
    f"R²   : {gd_r2:.6f}"
)


# ============================================================
# 29. SAVE GRADIENT DESCENT WEIGHTS
# ============================================================

np.save(
    os.path.join(
        MODEL_DIR,
        "gd_weights.npy"
    ),
    gd_weights
)


# ============================================================
# 30. SAVE GRADIENT DESCENT LOSS
# ============================================================

loss_df = pd.DataFrame(
    {
        "epoch": np.arange(
            1,
            len(loss_history) + 1
        ),
        "mse": loss_history
    }
)


loss_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "gd_loss.csv"
    ),
    index=False
)


# ============================================================
# 31. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "actual_log_stream_count":
            y_test.to_numpy(),

        "ols_prediction":
            ols_predictions,

        "gradient_descent_prediction":
            gd_predictions
    }
)


prediction_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "predictions.csv"
    ),
    index=False
)


# ============================================================
# 32. MODEL COMPARISON
# ============================================================

metrics_df = pd.DataFrame(
    [
        {
            "model": "OLS",
            "MAE": ols_mae,
            "RMSE": ols_rmse,
            "R2": ols_r2
        },
        {
            "model":
                "Manual Gradient Descent",
            "MAE":
                gd_mae,
            "RMSE":
                gd_rmse,
            "R2":
                gd_r2
        }
    ]
)


metrics_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "metrics.csv"
    ),
    index=False
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


print(
    metrics_df.to_string(
        index=False
    )
)


# ============================================================
# 33. MLFLOW SETUP
# ============================================================

print("\n" + "=" * 70)
print("MLFLOW EXPERIMENT TRACKING")
print("=" * 70)


mlflow.set_tracking_uri(
    "file:" + MLRUNS_DIR
)


MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")

mlflow.set_tracking_uri("sqlite:///" + MLFLOW_DB)
mlflow.set_experiment("PlaylistPulse-Week3-Stream-Regression")

# ============================================================
# 34. MLFLOW - OLS
# ============================================================

with mlflow.start_run(
    run_name="OLS-Stream-Regression"
):

    mlflow.log_param(
        "model",
        "Ordinary Least Squares"
    )

    mlflow.log_param(
        "target",
        "log_stream_count"
    )

    mlflow.log_param(
        "test_size",
        0.20
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "processed_features",
        X_train_processed.shape[1]
    )

    mlflow.log_param(
        "matched_tracks",
        len(df)
    )

    mlflow.log_metric(
        "MAE",
        float(ols_mae)
    )

    mlflow.log_metric(
        "RMSE",
        float(ols_rmse)
    )

    mlflow.log_metric(
        "R2",
        float(ols_r2)
    )

    mlflow.sklearn.log_model(
        ols_model,
        name="ols_model"
    )


# ============================================================
# 35. MLFLOW - GRADIENT DESCENT
# ============================================================

with mlflow.start_run(
    run_name="Gradient-Descent-Stream-Regression"
):

    mlflow.log_param(
        "model",
        "Manual Gradient Descent"
    )

    mlflow.log_param(
        "target",
        "log_stream_count"
    )

    mlflow.log_param(
        "learning_rate",
        0.01
    )

    mlflow.log_param(
        "epochs",
        3000
    )

    mlflow.log_param(
        "test_size",
        0.20
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "processed_features",
        X_train_processed.shape[1]
    )

    mlflow.log_param(
        "matched_tracks",
        len(df)
    )

    mlflow.log_metric(
        "MAE",
        float(gd_mae)
    )

    mlflow.log_metric(
        "RMSE",
        float(gd_rmse)
    )

    mlflow.log_metric(
        "R2",
        float(gd_r2)
    )

    mlflow.log_artifact(
        os.path.join(
            REPORT_DIR,
            "gd_loss.csv"
        )
    )


# ============================================================
# 36. SAVE FINAL MERGED DATASET
# ============================================================

merged_output_path = os.path.join(
    REPORT_DIR,
    "regression_dataset.csv"
)

df.to_csv(
    merged_output_path,
    index=False
)


# ============================================================
# 37. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WEEK 3 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nRegression target:"
)

print(
    "log_stream_count = log(1 + stream_count)"
)


print(
    "\nMatched tracks:",
    len(df)
)


print(
    "\nGenerated files:"
)

print(
    "1.",
    os.path.join(
        REPORT_DIR,
        "regression_dataset.csv"
    )
)

print(
    "2.",
    os.path.join(
        REPORT_DIR,
        "metrics.csv"
    )
)

print(
    "3.",
    os.path.join(
        REPORT_DIR,
        "predictions.csv"
    )
)

print(
    "4.",
    os.path.join(
        REPORT_DIR,
        "gd_loss.csv"
    )
)

print(
    "5.",
    os.path.join(
        MODEL_DIR,
        "ols_model.pkl"
    )
)

print(
    "6.",
    os.path.join(
        MODEL_DIR,
        "preprocessor.pkl"
    )

)

print(
    "7.",
    os.path.join(
        MODEL_DIR,
        "gd_weights.npy"
    )
)

print(
    "\nMLflow directory:"
)

print(
    MLRUNS_DIR
)

print(
    "\nAll Week 3 regression tasks completed."
)

print("=" * 70)