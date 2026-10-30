import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

warnings.filterwarnings("ignore")


# ==============================================================
# PLAYLIST PULSE - WEEK 4
# RIDGE + LASSO + LOGISTIC REGRESSION + MLFLOW
# ==============================================================

print("=" * 70)
print("PLAYLIST PULSE - WEEK 4")
print("REGULARIZATION + CLASSIFICATION")
print("RIDGE + LASSO + LOGISTIC REGRESSION + MLFLOW")
print("=" * 70)


# ==============================================================
# PROJECT PATHS
# ==============================================================

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
    "week4"
)

MODEL_DIR = os.path.join(
    REPORT_DIR,
    "models"
)

PLOT_DIR = os.path.join(
    REPORT_DIR,
    "figures"
)

MLFLOW_DB = os.path.join(
    BASE_DIR,
    "mlflow.db"
)


os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ==============================================================
# CHECK REQUIRED FILES
# ==============================================================

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


# ==============================================================
# LOAD DATA
# ==============================================================

print("\n" + "=" * 70)
print("LOADING SPOTIFY TRACK DATASET")
print("=" * 70)

tracks_df = pd.read_csv(TRACKS_PATH)

print(f"Tracks Dataset Shape: {tracks_df.shape}")

print("\nTrack Dataset Columns:")
print(tracks_df.columns.tolist())


print("\n" + "=" * 70)
print("LOADING STREAMING HISTORY")
print("=" * 70)

history_df = pd.read_csv(HISTORY_PATH)

print(f"Streaming History Shape: {history_df.shape}")

print("\nHistory Columns:")
print(history_df.columns.tolist())


# ==============================================================
# REQUIRED COLUMNS
# ==============================================================

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

for column in required_track_columns:
    if column not in tracks_df.columns:
        raise ValueError(
            f"Required track column missing: {column}"
        )

if "spotify_track_uri" not in history_df.columns:
    raise ValueError(
        "spotify_track_uri column missing from streaming history."
    )

if "skipped" not in history_df.columns:
    raise ValueError(
        "skipped column missing from streaming history."
    )


# ==============================================================
# CLEAN TRACK IDS
# ==============================================================

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

tracks_df = tracks_df[
    tracks_df["track_id"].notna()
    & (tracks_df["track_id"] != "")
].copy()

history_df = history_df[
    history_df["spotify_track_uri"].notna()
    & (history_df["spotify_track_uri"] != "")
].copy()

history_df["track_id"] = history_df["spotify_track_uri"]

print(f"Valid track rows: {len(tracks_df)}")
print(f"Valid history rows: {len(history_df)}")


# ==============================================================
# CHECK OVERLAP
# ==============================================================

print("\n" + "=" * 70)
print("CHECKING DATASET OVERLAP")
print("=" * 70)

track_ids = set(
    tracks_df["track_id"].dropna()
)

history_track_ids = set(
    history_df["track_id"].dropna()
)

matched_track_ids = track_ids.intersection(
    history_track_ids
)

print(
    f"Unique track IDs in dataset.csv: "
    f"{len(track_ids)}"
)

print(
    f"Unique track IDs in spotify_history.csv: "
    f"{len(history_track_ids)}"
)

print(
    f"Matched track IDs: "
    f"{len(matched_track_ids)}"
)

if len(matched_track_ids) == 0:
    raise ValueError(
        "No matching track IDs were found."
    )


# ==============================================================
# FEATURE DEFINITIONS
# ==============================================================

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


# ==============================================================
# PREPROCESSING PIPELINE
# ==============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
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
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==============================================================
# PART A - REGRESSION
# ==============================================================
# Target:
# log_stream_count = log(1 + stream_count)
#
# Stream count is the number of listening-history events
# associated with each track.
# ==============================================================

print("\n" + "=" * 70)
print("PART A - REGRESSION")
print("=" * 70)

print("\nCreating stream count...")

stream_counts = (
    history_df
    .groupby("track_id")
    .size()
    .reset_index(name="stream_count")
)

print(
    f"Unique tracks with streaming history: "
    f"{len(stream_counts)}"
)

print(
    f"Total listening events: "
    f"{len(history_df)}"
)


# --------------------------------------------------------------
# REMOVE DUPLICATE TRACK ROWS FOR REGRESSION
# --------------------------------------------------------------

print("\nPreparing track-level regression dataset...")

tracks_for_regression = (
    tracks_df[
        tracks_df["track_id"].isin(
            matched_track_ids
        )
    ]
    .drop_duplicates(
        subset=["track_id"]
    )
    .copy()
)

regression_df = tracks_for_regression.merge(
    stream_counts,
    on="track_id",
    how="inner"
)

print(
    f"Regression dataset shape: "
    f"{regression_df.shape}"
)

print(
    f"Unique regression tracks: "
    f"{regression_df['track_id'].nunique()}"
)


# --------------------------------------------------------------
# LOG TARGET
# --------------------------------------------------------------

regression_df["log_stream_count"] = np.log1p(
    regression_df["stream_count"]
)

regression_df = regression_df.replace(
    [np.inf, -np.inf],
    np.nan
)

regression_df = regression_df.dropna(
    subset=["log_stream_count"]
)

print(
    "\nRegression target:"
)

print(
    "log_stream_count = "
    "log(1 + stream_count)"
)


# --------------------------------------------------------------
# REGRESSION FEATURES
# --------------------------------------------------------------

X_reg = regression_df[
    numeric_features + categorical_features
].copy()

y_reg = regression_df[
    "log_stream_count"
].copy()


# --------------------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------------------

X_train_reg, X_test_reg, y_train_reg, y_test_reg = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)

print(
    f"\nRegression training samples: "
    f"{len(X_train_reg)}"
)

print(
    f"Regression testing samples: "
    f"{len(X_test_reg)}"
)


# --------------------------------------------------------------
# PREPROCESS
# --------------------------------------------------------------

print("\nPreprocessing regression data...")

X_train_reg_processed = preprocessor.fit_transform(
    X_train_reg
)

X_test_reg_processed = preprocessor.transform(
    X_test_reg
)

X_train_reg_processed = np.asarray(
    X_train_reg_processed,
    dtype=np.float64
)

X_test_reg_processed = np.asarray(
    X_test_reg_processed,
    dtype=np.float64
)

print(
    "Processed regression training shape:",
    X_train_reg_processed.shape
)

print(
    "Processed regression testing shape:",
    X_test_reg_processed.shape
)


# ==============================================================
# RIDGE REGRESSION
# ==============================================================

print("\n" + "=" * 70)
print("MODEL 1 - RIDGE REGRESSION")
print("=" * 70)

ridge_model = Ridge(
    alpha=1.0
)

ridge_model.fit(
    X_train_reg_processed,
    y_train_reg
)

ridge_predictions = ridge_model.predict(
    X_test_reg_processed
)

ridge_mae = mean_absolute_error(
    y_test_reg,
    ridge_predictions
)

ridge_rmse = np.sqrt(
    mean_squared_error(
        y_test_reg,
        ridge_predictions
    )
)

ridge_r2 = r2_score(
    y_test_reg,
    ridge_predictions
)

print("\nRidge Results:")
print(f"MAE  : {ridge_mae:.6f}")
print(f"RMSE : {ridge_rmse:.6f}")
print(f"R²   : {ridge_r2:.6f}")


# ==============================================================
# LASSO REGRESSION
# ==============================================================

print("\n" + "=" * 70)
print("MODEL 2 - LASSO REGRESSION")
print("=" * 70)

lasso_model = Lasso(
    alpha=0.001,
    max_iter=10000
)

lasso_model.fit(
    X_train_reg_processed,
    y_train_reg
)

lasso_predictions = lasso_model.predict(
    X_test_reg_processed
)

lasso_mae = mean_absolute_error(
    y_test_reg,
    lasso_predictions
)

lasso_rmse = np.sqrt(
    mean_squared_error(
        y_test_reg,
        lasso_predictions
    )
)

lasso_r2 = r2_score(
    y_test_reg,
    lasso_predictions
)

print("\nLasso Results:")
print(f"MAE  : {lasso_mae:.6f}")
print(f"RMSE : {lasso_rmse:.6f}")
print(f"R²   : {lasso_r2:.6f}")


# ==============================================================
# REGRESSION COMPARISON
# ==============================================================

print("\n" + "=" * 70)
print("REGRESSION MODEL COMPARISON")
print("=" * 70)

regression_metrics = pd.DataFrame(
    [
        {
            "model": "Ridge",
            "MAE": ridge_mae,
            "RMSE": ridge_rmse,
            "R2": ridge_r2
        },
        {
            "model": "Lasso",
            "MAE": lasso_mae,
            "RMSE": lasso_rmse,
            "R2": lasso_r2
        }
    ]
)

print(regression_metrics.to_string(index=False))

regression_metrics.to_csv(
    os.path.join(
        REPORT_DIR,
        "regression_metrics.csv"
    ),
    index=False
)


# --------------------------------------------------------------
# REGRESSION PREDICTIONS
# --------------------------------------------------------------

regression_predictions = pd.DataFrame(
    {
        "actual_log_stream_count":
            y_test_reg.to_numpy(),

        "ridge_prediction":
            ridge_predictions,

        "lasso_prediction":
            lasso_predictions
    }
)

regression_predictions.to_csv(
    os.path.join(
        REPORT_DIR,
        "regression_predictions.csv"
    ),
    index=False
)


# --------------------------------------------------------------
# SAVE REGRESSION MODELS
# --------------------------------------------------------------

joblib.dump(
    ridge_model,
    os.path.join(
        MODEL_DIR,
        "ridge_model.pkl"
    )
)

joblib.dump(
    lasso_model,
    os.path.join(
        MODEL_DIR,
        "lasso_model.pkl"
    )
)

joblib.dump(
    preprocessor,
    os.path.join(
        MODEL_DIR,
        "regression_preprocessor.pkl"
    )
)


# ==============================================================
# RIDGE VS LASSO PLOT
# ==============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    regression_metrics["model"],
    regression_metrics["R2"]
)

plt.xlabel("Model")
plt.ylabel("R²")
plt.title("Week 4 Regression Model Comparison")

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "regression_model_comparison.png"
    ),
    dpi=300
)

plt.close()


# ==============================================================
# PART B - CLASSIFICATION
# ==============================================================

print("\n" + "=" * 70)
print("PART B - SKIP CLASSIFICATION")
print("=" * 70)

print(
    "\nCreating skip_30s target from streaming history..."
)


# --------------------------------------------------------------
# CREATE SKIP TARGET
# --------------------------------------------------------------

classification_history = history_df.copy()

classification_history["skip_30s"] = (
    pd.to_numeric(
        classification_history["skipped"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

classification_history["skip_30s"] = (
    classification_history["skip_30s"]
    .clip(0, 1)
)


print("\nSkip target distribution:")

print(
    classification_history[
        "skip_30s"
    ].value_counts()
    .sort_index()
)


# --------------------------------------------------------------
# MERGE TRACK FEATURES WITH HISTORY
# --------------------------------------------------------------

print(
    "\nMerging track features with listening history..."
)

classification_df = classification_history.merge(
    tracks_df[
        [
            "track_id"
        ]
        + numeric_features
        + categorical_features
    ].drop_duplicates(
        subset=["track_id"]
    ),
    on="track_id",
    how="inner"
)

print(
    "Classification dataset shape:",
    classification_df.shape
)


if len(classification_df) == 0:
    raise ValueError(
        "No classification records matched "
        "the track dataset."
    )


# --------------------------------------------------------------
# PREPARE CLASSIFICATION DATA
# --------------------------------------------------------------

X_cls = classification_df[
    numeric_features + categorical_features
].copy()

y_cls = classification_df[
    "skip_30s"
].copy()


print(
    "\nClassification samples:",
    len(X_cls)
)

print(
    "Skipped:",
    int(y_cls.sum())
)

print(
    "Not skipped:",
    int((y_cls == 0).sum())
)


# --------------------------------------------------------------
# CLASSIFICATION TRAIN / TEST SPLIT
# --------------------------------------------------------------

X_train_cls, X_test_cls, y_train_cls, y_test_cls = (
    train_test_split(
        X_cls,
        y_cls,
        test_size=0.20,
        random_state=42,
        stratify=y_cls
    )
)

print(
    "\nClassification training samples:",
    len(X_train_cls)
)

print(
    "Classification testing samples:",
    len(X_test_cls)
)


# --------------------------------------------------------------
# CLASSIFICATION PREPROCESSOR
# --------------------------------------------------------------

classification_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


print("\nPreprocessing classification data...")

X_train_cls_processed = (
    classification_preprocessor.fit_transform(
        X_train_cls
    )
)

X_test_cls_processed = (
    classification_preprocessor.transform(
        X_test_cls
    )
)

X_train_cls_processed = np.asarray(
    X_train_cls_processed,
    dtype=np.float64
)

X_test_cls_processed = np.asarray(
    X_test_cls_processed,
    dtype=np.float64
)

print(
    "Processed classification training shape:",
    X_train_cls_processed.shape
)

print(
    "Processed classification testing shape:",
    X_test_cls_processed.shape
)


# ==============================================================
# LOGISTIC REGRESSION
# ==============================================================

print("\n" + "=" * 70)
print("MODEL 3 - LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

print("\nTraining Logistic Regression...")

logistic_model.fit(
    X_train_cls_processed,
    y_train_cls
)

logistic_predictions = (
    logistic_model.predict(
        X_test_cls_processed
    )
)

logistic_probabilities = (
    logistic_model.predict_proba(
        X_test_cls_processed
    )[:, 1]
)


# ==============================================================
# CLASSIFICATION METRICS
# ==============================================================

accuracy = accuracy_score(
    y_test_cls,
    logistic_predictions
)

precision = precision_score(
    y_test_cls,
    logistic_predictions,
    zero_division=0
)

recall = recall_score(
    y_test_cls,
    logistic_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test_cls,
    logistic_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test_cls,
    logistic_probabilities
)


print("\nLogistic Regression Results:")
print(f"Accuracy  : {accuracy:.6f}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1 Score  : {f1:.6f}")
print(f"ROC-AUC   : {roc_auc:.6f}")


# ==============================================================
# CLASSIFICATION METRICS FILE
# ==============================================================

classification_metrics = pd.DataFrame(
    [
        {
            "model": "Logistic Regression",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc
        }
    ]
)

print("\nClassification metrics:")
print(
    classification_metrics.to_string(
        index=False
    )
)

classification_metrics.to_csv(
    os.path.join(
        REPORT_DIR,
        "classification_metrics.csv"
    ),
    index=False
)


# ==============================================================
# CLASSIFICATION PREDICTIONS
# ==============================================================

classification_predictions = pd.DataFrame(
    {
        "actual_skip_30s":
            y_test_cls.to_numpy(),

        "predicted_skip_30s":
            logistic_predictions,

        "skip_probability":
            logistic_probabilities
    }
)

classification_predictions.to_csv(
    os.path.join(
        REPORT_DIR,
        "classification_predictions.csv"
    ),
    index=False
)


# ==============================================================
# SAVE CLASSIFICATION MODEL
# ==============================================================

joblib.dump(
    logistic_model,
    os.path.join(
        MODEL_DIR,
        "logistic_regression_model.pkl"
    )
)

joblib.dump(
    classification_preprocessor,
    os.path.join(
        MODEL_DIR,
        "classification_preprocessor.pkl"
    )
)


# ==============================================================
# CONFUSION MATRIX
# ==============================================================

cm = confusion_matrix(
    y_test_cls,
    logistic_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Logistic Regression - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "logistic_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ==============================================================
# ROC CURVE
# ==============================================================

fpr, tpr, _ = roc_curve(
    y_test_cls,
    logistic_probabilities
)

plt.figure(figsize=(8, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "logistic_roc_curve.png"
    ),
    dpi=300
)

plt.close()


# ==============================================================
# MLFLOW
# ==============================================================

print("\n" + "=" * 70)
print("MLFLOW EXPERIMENT TRACKING")
print("=" * 70)

mlflow.set_tracking_uri(
    "sqlite:///" + MLFLOW_DB
)

mlflow.set_experiment(
    "PlaylistPulse-Week4-Regularization-Classification"
)


# --------------------------------------------------------------
# RIDGE MLFLOW
# --------------------------------------------------------------

print("\nLogging Ridge experiment...")

with mlflow.start_run(
    run_name="Week4-Ridge"
):

    mlflow.log_param(
        "model",
        "Ridge"
    )

    mlflow.log_param(
        "alpha",
        1.0
    )

    mlflow.log_param(
        "target",
        "log_stream_count"
    )

    mlflow.log_param(
        "test_size",
        0.20
    )

    mlflow.log_metric(
        "MAE",
        ridge_mae
    )

    mlflow.log_metric(
        "RMSE",
        ridge_rmse
    )

    mlflow.log_metric(
        "R2",
        ridge_r2
    )

    mlflow.sklearn.log_model(
        ridge_model,
        "ridge_model"
    )

    mlflow.log_artifact(
        os.path.join(
            REPORT_DIR,
            "regression_metrics.csv"
        )
    )


# --------------------------------------------------------------
# LASSO MLFLOW
# --------------------------------------------------------------

print("Logging Lasso experiment...")

with mlflow.start_run(
    run_name="Week4-Lasso"
):

    mlflow.log_param(
        "model",
        "Lasso"
    )

    mlflow.log_param(
        "alpha",
        0.001
    )

    mlflow.log_param(
        "target",
        "log_stream_count"
    )

    mlflow.log_param(
        "test_size",
        0.20
    )

    mlflow.log_metric(
        "MAE",
        lasso_mae
    )

    mlflow.log_metric(
        "RMSE",
        lasso_rmse
    )

    mlflow.log_metric(
        "R2",
        lasso_r2
    )

    mlflow.sklearn.log_model(
        lasso_model,
        "lasso_model"
    )

    mlflow.log_artifact(
        os.path.join(
            REPORT_DIR,
            "regression_metrics.csv"
        )
    )


# --------------------------------------------------------------
# LOGISTIC MLFLOW
# --------------------------------------------------------------

print("Logging Logistic Regression experiment...")

with mlflow.start_run(
    run_name="Week4-Logistic-Regression"
):

    mlflow.log_param(
        "model",
        "Logistic Regression"
    )

    mlflow.log_param(
        "max_iter",
        2000
    )

    mlflow.log_param(
        "class_weight",
        "balanced"
    )

    mlflow.log_param(
        "target",
        "skip_30s"
    )

    mlflow.log_param(
        "test_size",
        0.20
    )

    mlflow.log_metric(
        "Accuracy",
        accuracy
    )

    mlflow.log_metric(
        "Precision",
        precision
    )

    mlflow.log_metric(
        "Recall",
        recall
    )

    mlflow.log_metric(
        "F1",
        f1
    )

    mlflow.log_metric(
        "ROC_AUC",
        roc_auc
    )

    mlflow.sklearn.log_model(
        logistic_model,
        "logistic_regression_model"
    )

    mlflow.log_artifact(
        os.path.join(
            REPORT_DIR,
            "classification_metrics.csv"
        )
    )


# ==============================================================
# SAVE WEEK 4 DATASET
# ==============================================================

regression_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "regression_dataset.csv"
    ),
    index=False
)

classification_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "classification_dataset.csv"
    ),
    index=False
)


# ==============================================================
# FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("WEEK 4 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nRegression models:")
print("1. Ridge Regression")
print("2. Lasso Regression")

print("\nClassification model:")
print("3. Logistic Regression")

print("\nGenerated files:")

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
        "regression_metrics.csv"
    )
)

print(
    "3.",
    os.path.join(
        REPORT_DIR,
        "regression_predictions.csv"
    )
)

print(
    "4.",
    os.path.join(
        REPORT_DIR,
        "classification_dataset.csv"
    )
)

print(
    "5.",
    os.path.join(
        REPORT_DIR,
        "classification_metrics.csv"
    )
)

print(
    "6.",
    os.path.join(
        REPORT_DIR,
        "classification_predictions.csv"
    )
)

print(
    "7.",
    os.path.join(
        MODEL_DIR,
        "ridge_model.pkl"
    )
)

print(
    "8.",
    os.path.join(
        MODEL_DIR,
        "lasso_model.pkl"
    )
)

print(
    "9.",
    os.path.join(
        MODEL_DIR,
        "logistic_regression_model.pkl"
    )
)

print(
    "10.",
    os.path.join(
        PLOT_DIR,
        "regression_model_comparison.png"
    )
)

print(
    "11.",
    os.path.join(
        PLOT_DIR,
        "logistic_confusion_matrix.png"
    )
)

print(
    "12.",
    os.path.join(
        PLOT_DIR,
        "logistic_roc_curve.png"
    )
)

print(
    "\nMLflow database:"
)

print(MLFLOW_DB)

print("\nAll Week 4 tasks completed.")
print("=" * 70)