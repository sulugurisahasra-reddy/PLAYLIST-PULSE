import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso

import mlflow

warnings.filterwarnings("ignore")


# ==============================================================
# HEADER
# ==============================================================

print("=" * 70)
print("PLAYLIST PULSE - WEEK 5")
print("CROSS-VALIDATION + MODEL SELECTION")
print("RIDGE + LASSO + ONE-STANDARD-ERROR RULE")
print("=" * 70)


# ==============================================================
# PATHS
# ==============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TRACK_FILE = os.path.join(
    BASE_DIR,
    "src",
    "data",
    "dataset.csv"
)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "src",
    "data",
    "spotify_history.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "week5"
)

FIGURE_DIR = os.path.join(
    OUTPUT_DIR,
    "figures"
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "models"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================================================
# CHECK REQUIRED FILES
# ==============================================================

print("\nChecking required files...")

if not os.path.exists(TRACK_FILE):
    raise FileNotFoundError(
        f"Track dataset not found:\n{TRACK_FILE}"
    )

if not os.path.exists(HISTORY_FILE):
    raise FileNotFoundError(
        f"Streaming history not found:\n{HISTORY_FILE}"
    )

print("Track dataset found.")
print("Streaming history found.")


# ==============================================================
# LOAD TRACK DATASET
# ==============================================================

print("\n" + "=" * 70)
print("LOADING SPOTIFY TRACK DATASET")
print("=" * 70)

tracks = pd.read_csv(TRACK_FILE)

print(f"Tracks Dataset Shape: {tracks.shape}")

print("\nTrack Dataset Columns:")
print(tracks.columns.tolist())


# ==============================================================
# LOAD STREAMING HISTORY
# ==============================================================

print("\n" + "=" * 70)
print("LOADING STREAMING HISTORY")
print("=" * 70)

history = pd.read_csv(HISTORY_FILE)

print(f"Streaming History Shape: {history.shape}")

print("\nHistory Columns:")
print(history.columns.tolist())


# ==============================================================
# CLEAN TRACK IDS
# ==============================================================

print("\n" + "=" * 70)
print("CLEANING TRACK IDS")
print("=" * 70)

tracks["track_id"] = (
    tracks["track_id"]
    .astype(str)
    .str.strip()
)

history["spotify_track_uri"] = (
    history["spotify_track_uri"]
    .astype(str)
    .str.strip()
)

tracks = tracks[
    tracks["track_id"].notna()
    & (tracks["track_id"] != "")
    & (tracks["track_id"] != "nan")
].copy()

history = history[
    history["spotify_track_uri"].notna()
    & (history["spotify_track_uri"] != "")
    & (history["spotify_track_uri"] != "nan")
].copy()

print(f"Valid track rows: {len(tracks)}")
print(f"Valid history rows: {len(history)}")


# ==============================================================
# DATASET OVERLAP
# ==============================================================

print("\n" + "=" * 70)
print("CHECKING DATASET OVERLAP")
print("=" * 70)

dataset_ids = set(
    tracks["track_id"].unique()
)

history_ids = set(
    history["spotify_track_uri"].unique()
)

matched_ids = dataset_ids.intersection(
    history_ids
)

print(
    f"Unique track IDs in dataset.csv: "
    f"{len(dataset_ids)}"
)

print(
    f"Unique track IDs in spotify_history.csv: "
    f"{len(history_ids)}"
)

print(
    f"Matched track IDs: "
    f"{len(matched_ids)}"
)


# ==============================================================
# CREATE STREAM COUNT
# ==============================================================

print("\n" + "=" * 70)
print("CREATING STREAM COUNT")
print("=" * 70)

stream_counts = (
    history
    .groupby("spotify_track_uri")
    .size()
    .reset_index(name="stream_count")
)

stream_counts = stream_counts.rename(
    columns={
        "spotify_track_uri": "track_id"
    }
)

print(
    f"Unique tracks with streaming history: "
    f"{len(stream_counts)}"
)

print(
    f"Total listening events: "
    f"{len(history)}"
)


# ==============================================================
# PREPARE REGRESSION DATASET
# ==============================================================

print("\n" + "=" * 70)
print("PREPARING TRACK-LEVEL REGRESSION DATASET")
print("=" * 70)

track_features = tracks.drop_duplicates(
    subset=["track_id"]
).copy()

regression_df = track_features.merge(
    stream_counts,
    on="track_id",
    how="inner"
)

regression_df["log_stream_count"] = np.log1p(
    regression_df["stream_count"]
)

print(
    f"Regression dataset shape: "
    f"{regression_df.shape}"
)

print(
    f"Unique regression tracks: "
    f"{regression_df['track_id'].nunique()}"
)


# ==============================================================
# SAVE REGRESSION DATASET
# ==============================================================

regression_dataset_path = os.path.join(
    OUTPUT_DIR,
    "regression_dataset.csv"
)

regression_df.to_csv(
    regression_dataset_path,
    index=False
)


# ==============================================================
# DEFINE TARGET AND FEATURES
# ==============================================================

target_column = "log_stream_count"

excluded_columns = [
    "track_id",
    "stream_count",
    "log_stream_count",
    "Unnamed: 0"
]

feature_columns = [
    column
    for column in regression_df.columns
    if column not in excluded_columns
]

X = regression_df[
    feature_columns
].copy()

y = regression_df[
    target_column
].copy()


# ==============================================================
# IDENTIFY FEATURE TYPES
# ==============================================================

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "bool"]
).columns.tolist()

print("\nFeature configuration:")
print(
    f"Numeric features: "
    f"{len(numeric_columns)}"
)

print(
    f"Categorical features: "
    f"{len(categorical_columns)}"
)


# ==============================================================
# PREPROCESSOR
# ==============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
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
            ),
            numeric_columns
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            ),
            categorical_columns
        )
    ]
)


# ==============================================================
# CROSS-VALIDATION CONFIGURATION
# ==============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}


# ==============================================================
# ALPHA VALUES
# ==============================================================

alpha_values = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]

results = []


# ==============================================================
# RIDGE CROSS-VALIDATION
# ==============================================================

print("\n" + "=" * 70)
print("RIDGE CROSS-VALIDATION")
print("=" * 70)

for alpha in alpha_values:

    ridge_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                Ridge(
                    alpha=alpha
                )
            )
        ]
    )

    cv_results = cross_validate(
        ridge_pipeline,
        X,
        y,
        cv=kf,
        scoring=scoring,
        n_jobs=-1
    )

    mae_scores = -cv_results[
        "test_MAE"
    ]

    rmse_scores = -cv_results[
        "test_RMSE"
    ]

    r2_scores = cv_results[
        "test_R2"
    ]

    results.append(
        {
            "model": "Ridge",
            "alpha": alpha,
            "MAE_mean": mae_scores.mean(),
            "MAE_std": mae_scores.std(),
            "RMSE_mean": rmse_scores.mean(),
            "RMSE_std": rmse_scores.std(),
            "R2_mean": r2_scores.mean(),
            "R2_std": r2_scores.std()
        }
    )

    print(
        f"Alpha={alpha:<8} "
        f"MAE={mae_scores.mean():.6f} "
        f"RMSE={rmse_scores.mean():.6f} "
        f"R2={r2_scores.mean():.6f}"
    )


# ==============================================================
# LASSO CROSS-VALIDATION
# ==============================================================

print("\n" + "=" * 70)
print("LASSO CROSS-VALIDATION")
print("=" * 70)

for alpha in alpha_values:

    lasso_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                Lasso(
                    alpha=alpha,
                    max_iter=20000
                )
            )
        ]
    )

    cv_results = cross_validate(
        lasso_pipeline,
        X,
        y,
        cv=kf,
        scoring=scoring,
        n_jobs=-1
    )

    mae_scores = -cv_results[
        "test_MAE"
    ]

    rmse_scores = -cv_results[
        "test_RMSE"
    ]

    r2_scores = cv_results[
        "test_R2"
    ]

    results.append(
        {
            "model": "Lasso",
            "alpha": alpha,
            "MAE_mean": mae_scores.mean(),
            "MAE_std": mae_scores.std(),
            "RMSE_mean": rmse_scores.mean(),
            "RMSE_std": rmse_scores.std(),
            "R2_mean": r2_scores.mean(),
            "R2_std": r2_scores.std()
        }
    )

    print(
        f"Alpha={alpha:<8} "
        f"MAE={mae_scores.mean():.6f} "
        f"RMSE={rmse_scores.mean():.6f} "
        f"R2={r2_scores.mean():.6f}"
    )


# ==============================================================
# SAVE CROSS-VALIDATION RESULTS
# ==============================================================

cv_df = pd.DataFrame(
    results
)

cv_results_path = os.path.join(
    OUTPUT_DIR,
    "cross_validation_results.csv"
)

cv_df.to_csv(
    cv_results_path,
    index=False
)


# ==============================================================
# FIND BEST MODELS
# ==============================================================

print("\n" + "=" * 70)
print("BEST CROSS-VALIDATION RESULTS")
print("=" * 70)

best_mae = cv_df.loc[
    cv_df["MAE_mean"].idxmin()
]

best_rmse = cv_df.loc[
    cv_df["RMSE_mean"].idxmin()
]

best_r2 = cv_df.loc[
    cv_df["R2_mean"].idxmax()
]

print("\nBest by MAE:")
print(best_mae.to_string())

print("\nBest by RMSE:")
print(best_rmse.to_string())

print("\nBest by R2:")
print(best_r2.to_string())


# ==============================================================
# ONE-STANDARD-ERROR RULE
# ==============================================================

print("\n" + "=" * 70)
print("ONE-STANDARD-ERROR RULE")
print("=" * 70)

best_row = cv_df.loc[
    cv_df["MAE_mean"].idxmin()
]

best_mae_mean = best_row[
    "MAE_mean"
]

best_mae_std = best_row[
    "MAE_std"
]

standard_error = (
    best_mae_std / np.sqrt(5)
)

threshold = (
    best_mae_mean + standard_error
)

print(
    f"Best CV MAE: "
    f"{best_mae_mean:.6f}"
)

print(
    f"Best CV MAE STD: "
    f"{best_mae_std:.6f}"
)

print(
    f"Standard Error: "
    f"{standard_error:.6f}"
)

print(
    f"1-SE Threshold: "
    f"{threshold:.6f}"
)


# ==============================================================
# ELIGIBLE MODELS
# ==============================================================

eligible_models = cv_df[
    cv_df["MAE_mean"] <= threshold
].copy()

print("\nModels within one standard error:")

print(
    eligible_models[
        [
            "model",
            "alpha",
            "MAE_mean",
            "MAE_std",
            "RMSE_mean",
            "R2_mean"
        ]
    ].to_string(index=False)
)


# ==============================================================
# ONE-SE MODEL SELECTION
# ==============================================================

# Among models whose MAE is within one standard error
# of the best model, select the simpler regularized model.
#
# Larger alpha means stronger regularization.
# Therefore, choose the largest alpha among eligible models.

one_se_model = eligible_models.sort_values(
    by=[
        "alpha",
        "MAE_mean"
    ],
    ascending=[
        False,
        True
    ]
).iloc[0]

print(
    "\nModel selected by One-Standard-Error Rule:"
)

print(
    one_se_model.to_string()
)


# ==============================================================
# SAVE MODEL SELECTION RESULTS
# ==============================================================

selection_df = pd.DataFrame(
    [
        {
            "selection_method": "Best MAE",
            "model": best_mae["model"],
            "alpha": best_mae["alpha"],
            "MAE_mean": best_mae["MAE_mean"],
            "RMSE_mean": best_mae["RMSE_mean"],
            "R2_mean": best_mae["R2_mean"]
        },
        {
            "selection_method": "Best RMSE",
            "model": best_rmse["model"],
            "alpha": best_rmse["alpha"],
            "MAE_mean": best_rmse["MAE_mean"],
            "RMSE_mean": best_rmse["RMSE_mean"],
            "R2_mean": best_rmse["R2_mean"]
        },
        {
            "selection_method": "Best R2",
            "model": best_r2["model"],
            "alpha": best_r2["alpha"],
            "MAE_mean": best_r2["MAE_mean"],
            "RMSE_mean": best_r2["RMSE_mean"],
            "R2_mean": best_r2["R2_mean"]
        },
        {
            "selection_method": "One Standard Error",
            "model": one_se_model["model"],
            "alpha": one_se_model["alpha"],
            "MAE_mean": one_se_model["MAE_mean"],
            "RMSE_mean": one_se_model["RMSE_mean"],
            "R2_mean": one_se_model["R2_mean"]
        }
    ]
)

selection_path = os.path.join(
    OUTPUT_DIR,
    "model_selection.csv"
)

selection_df.to_csv(
    selection_path,
    index=False
)


# ==============================================================
# TRAIN FINAL SELECTED MODEL
# ==============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL SELECTED MODEL")
print("=" * 70)

selected_model_name = (
    one_se_model["model"]
)

selected_alpha = (
    one_se_model["alpha"]
)


if selected_model_name == "Ridge":

    final_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                Ridge(
                    alpha=selected_alpha
                )
            )
        ]
    )

else:

    final_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                Lasso(
                    alpha=selected_alpha,
                    max_iter=20000
                )
            )
        ]
    )


final_model.fit(
    X,
    y
)


# ==============================================================
# SAVE FINAL MODEL
# ==============================================================

model_path = os.path.join(
    MODEL_DIR,
    "selected_regression_model.pkl"
)

with open(
    model_path,
    "wb"
) as file:

    pickle.dump(
        final_model,
        file
    )

print(
    f"Selected model: "
    f"{selected_model_name}"
)

print(
    f"Selected alpha: "
    f"{selected_alpha}"
)

print(
    f"Model saved: "
    f"{model_path}"
)


# ==============================================================
# PLOT 1 - CROSS-VALIDATION MAE
# ==============================================================

print("\nCreating cross-validation MAE plot...")

plt.figure(
    figsize=(10, 6)
)

for model_name in [
    "Ridge",
    "Lasso"
]:

    model_data = cv_df[
        cv_df["model"] == model_name
    ]

    plt.errorbar(
        model_data["alpha"],
        model_data["MAE_mean"],
        yerr=model_data["MAE_std"],
        marker="o",
        capsize=4,
        label=model_name
    )

plt.xscale("log")

plt.xlabel(
    "Alpha"
)

plt.ylabel(
    "Cross-Validation MAE"
)

plt.title(
    "Week 5 - Cross-Validation MAE"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

mae_plot_path = os.path.join(
    FIGURE_DIR,
    "cross_validation_mae.png"
)

plt.savefig(
    mae_plot_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==============================================================
# PLOT 2 - CROSS-VALIDATION R2
# ==============================================================

print(
    "Creating cross-validation R² plot..."
)

plt.figure(
    figsize=(10, 6)
)

for model_name in [
    "Ridge",
    "Lasso"
]:

    model_data = cv_df[
        cv_df["model"] == model_name
    ]

    plt.plot(
        model_data["alpha"],
        model_data["R2_mean"],
        marker="o",
        label=model_name
    )

plt.xscale("log")

plt.xlabel(
    "Alpha"
)

plt.ylabel(
    "Cross-Validation R²"
)

plt.title(
    "Week 5 - Cross-Validation R²"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

r2_plot_path = os.path.join(
    FIGURE_DIR,
    "cross_validation_r2.png"
)

plt.savefig(
    r2_plot_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==============================================================
# PLOT 3 - MODEL COMPARISON
# ==============================================================

print(
    "Creating model comparison plot..."
)

plt.figure(
    figsize=(10, 6)
)

for model_name in [
    "Ridge",
    "Lasso"
]:

    model_data = cv_df[
        cv_df["model"] == model_name
    ]

    plt.plot(
        model_data["alpha"],
        model_data["RMSE_mean"],
        marker="o",
        label=model_name
    )

plt.xscale("log")

plt.xlabel(
    "Alpha"
)

plt.ylabel(
    "Cross-Validation RMSE"
)

plt.title(
    "Week 5 - Cross-Validation RMSE"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

rmse_plot_path = os.path.join(
    FIGURE_DIR,
    "cross_validation_rmse.png"
)

plt.savefig(
    rmse_plot_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==============================================================
# MLFLOW
# ==============================================================

print("\n" + "=" * 70)
print("MLFLOW EXPERIMENT TRACKING")
print("=" * 70)

MLFLOW_DB = os.path.join(
    BASE_DIR,
    "mlflow.db"
)

mlflow.set_tracking_uri(
    "sqlite:///" + MLFLOW_DB
)

mlflow.set_experiment(
    "PlaylistPulse-Week5-CrossValidation"
)


with mlflow.start_run(
    run_name="Week5-Selected-Model"
):

    mlflow.log_param(
        "selected_model",
        selected_model_name
    )

    mlflow.log_param(
        "selected_alpha",
        float(selected_alpha)
    )

    mlflow.log_param(
        "cv_folds",
        5
    )

    mlflow.log_param(
        "selection_method",
        "One Standard Error Rule"
    )

    mlflow.log_metric(
        "best_cv_mae",
        float(best_mae["MAE_mean"])
    )

    mlflow.log_metric(
        "best_cv_rmse",
        float(best_rmse["RMSE_mean"])
    )

    mlflow.log_metric(
        "best_cv_r2",
        float(best_r2["R2_mean"])
    )

    mlflow.log_metric(
        "one_se_mae",
        float(one_se_model["MAE_mean"])
    )

    mlflow.log_artifact(
        cv_results_path
    )

    mlflow.log_artifact(
        selection_path
    )

    mlflow.log_artifact(
        mae_plot_path
    )

    mlflow.log_artifact(
        r2_plot_path
    )

    mlflow.log_artifact(
        rmse_plot_path
    )


# ==============================================================
# FINAL OUTPUT
# ==============================================================

print("\n" + "=" * 70)
print("WEEK 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCross-validation:")
print("- 5-fold K-Fold cross-validation")
print("- Ridge tested with multiple alpha values")
print("- Lasso tested with multiple alpha values")

print("\nModel selection:")
print(
    f"- Best MAE: "
    f"{best_mae['model']} "
    f"(alpha={best_mae['alpha']})"
)

print(
    f"- Best RMSE: "
    f"{best_rmse['model']} "
    f"(alpha={best_rmse['alpha']})"
)

print(
    f"- Best R²: "
    f"{best_r2['model']} "
    f"(alpha={best_r2['alpha']})"
)

print(
    f"- One-SE model: "
    f"{one_se_model['model']} "
    f"(alpha={one_se_model['alpha']})"
)

print("\nGenerated files:")

print(
    "1.",
    regression_dataset_path
)

print(
    "2.",
    cv_results_path
)

print(
    "3.",
    selection_path
)

print(
    "4.",
    model_path
)

print(
    "5.",
    mae_plot_path
)

print(
    "6.",
    r2_plot_path
)

print(
    "7.",
    rmse_plot_path
)

print("\nMLflow database:")
print(MLFLOW_DB)

print("=" * 70)