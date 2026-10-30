import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
WEEK8_DIR = os.path.join(REPORTS_DIR, "week8")

FIGURES_DIR = os.path.join(WEEK8_DIR, "figures")
MODELS_DIR = os.path.join(WEEK8_DIR, "models")

os.makedirs(WEEK8_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_read_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Could not read {path}: {e}")
    return None


def copy_if_exists(source, destination):
    if os.path.exists(source):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(source, destination)
        print(f"Copied: {os.path.basename(source)}")
        return True
    return False


def save_dataframe(df, filename):
    path = os.path.join(WEEK8_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ============================================================
# PROJECT INFORMATION
# ============================================================

print("=" * 70)
print("PLAYLISTPULSE - WEEK 8 FINAL INTEGRATION")
print("=" * 70)

print(f"Project directory: {BASE_DIR}")
print(f"Reports directory: {REPORTS_DIR}")
print(f"Week 8 directory: {WEEK8_DIR}")


# ============================================================
# VERIFIED FINAL MODEL RESULTS
# ============================================================

final_models = pd.DataFrame({
    "Model": [
        "Lasso Regression",
        "Gradient Boosting",
        "Random Forest",
        "Decision Tree"
    ],
    "Week": [
        "Week 5 / Week 7",
        "Week 6",
        "Week 6",
        "Week 6"
    ],
    "MAE": [
        0.679296,
        0.927447,
        0.934842,
        0.995756
    ],
    "RMSE": [
        1.018544,
        1.115416,
        1.158535,
        1.247293
    ],
    "R2": [
        0.478816,
        0.374963,
        0.325705,
        0.218429
    ]
})

final_models = final_models.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)

save_dataframe(
    final_models,
    "final_model_comparison.csv"
)


# ============================================================
# FINAL MODEL SUMMARY
# ============================================================

best_model = final_models.iloc[0]

summary = pd.DataFrame({
    "Metric": [
        "Final Model",
        "Regularization",
        "MAE",
        "RMSE",
        "R2",
        "Training Strategy",
        "Evaluation Type"
    ],
    "Value": [
        "Lasso Regression",
        "L1 regularization, alpha = 0.0001",
        f"{best_model['MAE']:.6f}",
        f"{best_model['RMSE']:.6f}",
        f"{best_model['R2']:.6f}",
        "80/20 train-test split, random_state=42",
        "Held-out test set"
    ]
})

save_dataframe(
    summary,
    "final_metrics_summary.csv"
)


# ============================================================
# MODEL COMPARISON - RMSE
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    final_models["Model"],
    final_models["RMSE"]
)

plt.title("Final Model Comparison - RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.xticks(rotation=20)
plt.tight_layout()

rmse_path = os.path.join(
    FIGURES_DIR,
    "final_model_rmse_comparison.png"
)

plt.savefig(rmse_path, dpi=300)
plt.close()

print(f"Saved: {rmse_path}")


# ============================================================
# MODEL COMPARISON - MAE
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    final_models["Model"],
    final_models["MAE"]
)

plt.title("Final Model Comparison - MAE")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.xticks(rotation=20)
plt.tight_layout()

mae_path = os.path.join(
    FIGURES_DIR,
    "final_model_mae_comparison.png"
)

plt.savefig(mae_path, dpi=300)
plt.close()

print(f"Saved: {mae_path}")


# ============================================================
# MODEL COMPARISON - R2
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    final_models["Model"],
    final_models["R2"]
)

plt.title("Final Model Comparison - R²")
plt.xlabel("Model")
plt.ylabel("R²")
plt.xticks(rotation=20)
plt.tight_layout()

r2_path = os.path.join(
    FIGURES_DIR,
    "final_model_r2_comparison.png"
)

plt.savefig(r2_path, dpi=300)
plt.close()

print(f"Saved: {r2_path}")


# ============================================================
# COPY WEEK 7 FIGURES
# ============================================================

WEEK7_FIGURES = os.path.join(
    REPORTS_DIR,
    "week7",
    "figures"
)

week7_figures = [
    "actual_vs_predicted.png",
    "residual_distribution.png",
    "prediction_errors.png",
    "feature_importance.png",
    "shap_bar.png",
    "shap_summary.png",
    "shap_waterfall.png"
]

print("\nCopying Week 7 figures...")

for filename in week7_figures:

    source = os.path.join(
        WEEK7_FIGURES,
        filename
    )

    destination = os.path.join(
        FIGURES_DIR,
        filename
    )

    copy_if_exists(
        source,
        destination
    )


# ============================================================
# COPY FINAL MODEL FILES
# ============================================================

WEEK7_MODELS = os.path.join(
    REPORTS_DIR,
    "week7",
    "models"
)

print("\nCopying final model files...")

model_files = [
    "final_selected_model.pkl",
    "final_preprocessor.pkl"
]

for filename in model_files:

    source = os.path.join(
        WEEK7_MODELS,
        filename
    )

    destination = os.path.join(
        MODELS_DIR,
        filename
    )

    copy_if_exists(
        source,
        destination
    )


# ============================================================
# COPY IMPORTANT WEEK 7 CSV FILES
# ============================================================

WEEK7_DIR = os.path.join(
    REPORTS_DIR,
    "week7"
)

week7_csv_files = [
    "shap_feature_importance.csv",
    "feature_importance.csv",
    "final_predictions.csv",
    "residual_analysis.csv",
    "processed_feature_names.csv",
    "selected_final_model.csv",
    "final_metrics.csv",
    "final_regression_dataset.csv"
]

print("\nCopying Week 7 CSV outputs...")

for filename in week7_csv_files:

    source = os.path.join(
        WEEK7_DIR,
        filename
    )

    destination = os.path.join(
        WEEK8_DIR,
        filename
    )

    copy_if_exists(
        source,
        destination
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

dataset_summary = pd.DataFrame({
    "Dataset": [
        "Spotify Track Dataset",
        "Spotify Listening History",
        "Matched Regression Dataset",
        "Final Training Dataset",
        "Final Test Dataset"
    ],
    "Rows": [
        114000,
        149860,
        1710,
        1368,
        342
    ],
    "Description": [
        "Track-level Spotify audio and metadata",
        "User listening history",
        "Tracks matched with listening history",
        "80% training split",
        "20% held-out test split"
    ]
})

save_dataframe(
    dataset_summary,
    "dataset_summary.csv"
)


# ============================================================
# METHODOLOGY
# ============================================================

methodology = pd.DataFrame({
    "Stage": [
        "Week 1",
        "Week 2",
        "Week 3",
        "Week 4",
        "Week 5",
        "Week 6",
        "Week 7",
        "Week 8"
    ],
    "Method": [
        "Dataset understanding and exploratory data analysis",
        "Data cleaning and preprocessing",
        "OLS regression and manual Gradient Descent",
        "Ridge, Lasso and Logistic Regression",
        "K-fold cross-validation and one-standard-error model selection",
        "Decision Tree, Random Forest and Gradient Boosting",
        "Final evaluation and SHAP explainability",
        "Final integration, reporting and presentation outputs"
    ]
})

save_dataframe(
    methodology,
    "methodology.csv"
)


# ============================================================
# PRESENTATION SUMMARY
# ============================================================

presentation_summary = pd.DataFrame({
    "Slide": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
    ],
    "Title": [
        "PlaylistPulse",
        "Problem Statement",
        "Dataset",
        "Data Preprocessing",
        "Regression Models",
        "Tree Ensemble Models",
        "Final Model Selection",
        "Model Performance",
        "Explainability",
        "Conclusion"
    ],
    "Content": [
        "Spotify listening behavior prediction system",
        "Predict expected stream count and analyze listening behavior",
        "Spotify track metadata and listening history",
        "Cleaning, matching, encoding and feature preprocessing",
        "OLS, Gradient Descent, Ridge and Lasso",
        "Decision Tree, Random Forest and Gradient Boosting",
        "Lasso Regression with alpha = 0.0001",
        "MAE = 0.679296, RMSE = 1.018544, R2 = 0.478816",
        "Feature importance and SHAP analysis",
        "Lasso provides the best final held-out test performance"
    ]
})

save_dataframe(
    presentation_summary,
    "presentation_summary.csv"
)


# ============================================================
# PROJECT SUMMARY
# ============================================================

project_summary = pd.DataFrame({
    "Item": [
        "Project",
        "Primary Task",
        "Regression Target",
        "Final Model",
        "Alpha",
        "MAE",
        "RMSE",
        "R2",
        "Best Model Reason",
        "Explainability"
    ],
    "Value": [
        "PlaylistPulse",
        "Spotify listening behavior prediction",
        "log_stream_count = log(1 + stream_count)",
        "Lasso Regression",
        "0.0001",
        "0.679296",
        "1.018544",
        "0.478816",
        "Lowest MAE and RMSE with highest R2 among final evaluated models",
        "SHAP LinearExplainer and model feature importance"
    ]
})

save_dataframe(
    project_summary,
    "project_summary.csv"
)


# ============================================================
# FINAL REPORT
# ============================================================

report_text = f"""
============================================================
PLAYLISTPULSE - FINAL PROJECT REPORT
============================================================

PROJECT
PlaylistPulse - Spotify Listening Behavior Prediction

============================================================
1. PROJECT OBJECTIVE
============================================================

PlaylistPulse is a machine learning project designed to analyze
Spotify listening behavior and predict expected stream counts.

The project combines Spotify track metadata with listening history
to create a machine learning dataset.

The main regression target is:

log_stream_count = log(1 + stream_count)

============================================================
2. DATASETS
============================================================

Spotify Track Dataset:
114000 rows

Spotify Listening History:
149860 rows

Matched Regression Dataset:
1710 rows

Training samples:
1368

Testing samples:
342

The data was matched using Spotify track identifiers.

============================================================
3. DATA PREPROCESSING
============================================================

The preprocessing pipeline included:

- Data validation
- Missing-value handling
- Duplicate handling
- Spotify track ID extraction
- Dataset merging
- Numerical feature preprocessing
- Categorical feature encoding
- Feature scaling where required
- Train/test splitting

============================================================
4. MODELS EVALUATED
============================================================

Classical regression:

- Ordinary Least Squares
- Manual Gradient Descent
- Ridge Regression
- Lasso Regression

Tree and ensemble models:

- Decision Tree
- Random Forest
- Gradient Boosting

Cross-validation and one-standard-error model selection
were also performed.

============================================================
5. FINAL MODEL
============================================================

Final selected model:

Lasso Regression

Regularization:

L1 regularization

Alpha:

0.0001

============================================================
6. FINAL TEST PERFORMANCE
============================================================

MAE:
{best_model['MAE']:.6f}

RMSE:
{best_model['RMSE']:.6f}

R2:
{best_model['R2']:.6f}

============================================================
7. MODEL COMPARISON
============================================================

Lasso Regression
MAE  = 0.679296
RMSE = 1.018544
R2   = 0.478816

Gradient Boosting
MAE  = 0.927447
RMSE = 1.115416
R2   = 0.374963

Random Forest
MAE  = 0.934842
RMSE = 1.158535
R2   = 0.325705

Decision Tree
MAE  = 0.995756
RMSE = 1.247293
R2   = 0.218429

============================================================
8. MODEL SELECTION
============================================================

Lasso Regression was selected as the final model because it
achieved the lowest MAE and RMSE and the highest R2 among the
final evaluated models.

This indicates that the Lasso model provided the strongest
generalization performance on the held-out test dataset.

============================================================
9. EXPLAINABILITY
============================================================

SHAP explainability was performed during Week 7.

For the Lasso model, SHAP LinearExplainer was used.

The project generated:

- SHAP feature importance
- SHAP bar plot
- SHAP summary plot
- SHAP waterfall plot

These outputs help identify which features have the strongest
influence on the model predictions.

============================================================
10. CONCLUSION
============================================================

PlaylistPulse successfully combines Spotify track metadata and
listening history into a machine learning pipeline.

Multiple regression and ensemble models were evaluated.

The final selected model is Lasso Regression with alpha 0.0001.

The final held-out test performance is:

MAE  = 0.679296
RMSE = 1.018544
R2   = 0.478816

The project also includes model explainability through SHAP,
final model comparison charts, prediction analysis, residual
analysis and presentation-ready outputs.

============================================================
END OF REPORT
============================================================
"""

report_path = os.path.join(
    WEEK8_DIR,
    "FINAL_REPORT.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:
    file.write(report_text)

print(f"Saved: {report_path}")


# ============================================================
# README
# ============================================================

readme_text = f"""# PlaylistPulse - Final Project

## Overview

PlaylistPulse is a machine learning project for analyzing Spotify
listening behavior and predicting expected stream count.

## Final Model

**Lasso Regression**

Alpha: `0.0001`

## Final Performance

| Metric | Score |
|---|---:|
| MAE | 0.679296 |
| RMSE | 1.018544 |
| R² | 0.478816 |

## Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Lasso Regression | 0.679296 | 1.018544 | 0.478816 |
| Gradient Boosting | 0.927447 | 1.115416 | 0.374963 |
| Random Forest | 0.934842 | 1.158535 | 0.325705 |
| Decision Tree | 0.995756 | 1.247293 | 0.218429 |

## Dataset

- Spotify tracks: 114,000 rows
- Spotify history: 149,860 rows
- Matched dataset: 1,710 rows
- Training samples: 1,368
- Testing samples: 342

## Methodology

1. Dataset understanding
2. Exploratory data analysis
3. Data cleaning
4. Feature preprocessing
5. OLS regression
6. Manual Gradient Descent
7. Ridge Regression
8. Lasso Regression
9. Logistic Regression
10. Cross-validation
11. One-standard-error model selection
12. Decision Tree
13. Random Forest
14. Gradient Boosting
15. Final model evaluation
16. SHAP explainability
17. Final integration

## Explainability

SHAP was used to explain the final Lasso model.

Generated explainability outputs include:

- SHAP feature importance
- SHAP bar plot
- SHAP summary plot
- SHAP waterfall plot

## Week 8 Outputs

This folder contains:

- Final model comparison
- Final metrics
- Dataset summary
- Methodology
- Final report
- Presentation summary
- Project summary
- Final model
- Final preprocessor
- Prediction results
- Residual analysis
- Feature importance
- SHAP results
- Final comparison charts

## Final Conclusion

Lasso Regression achieved the strongest final performance among
the evaluated models, with an MAE of {best_model['MAE']:.6f},
RMSE of {best_model['RMSE']:.6f}, and R² of {best_model['R2']:.6f}.
"""

readme_path = os.path.join(
    WEEK8_DIR,
    "README.md"
)

with open(
    readme_path,
    "w",
    encoding="utf-8"
) as file:
    file.write(readme_text)

print(f"Saved: {readme_path}")


# ============================================================
# FINAL OUTPUT TREE
# ============================================================

print("\n" + "=" * 70)
print("WEEK 8 OUTPUTS")
print("=" * 70)

for root, dirs, files in os.walk(WEEK8_DIR):

    level = root.replace(WEEK8_DIR, "").count(os.sep)

    indent = "    " * level

    folder_name = os.path.basename(root)

    print(f"{indent}{folder_name}/")

    sub_indent = "    " * (level + 1)

    for filename in sorted(files):
        print(f"{sub_indent}{filename}")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("WEEK 8 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Final model: Lasso Regression")
print(f"Alpha: 0.0001")
print(f"MAE: {best_model['MAE']:.6f}")
print(f"RMSE: {best_model['RMSE']:.6f}")
print(f"R2: {best_model['R2']:.6f}")

print("\nAll final integration outputs are available at:")
print(WEEK8_DIR)

print("\nFinal report:")
print(report_path)

print("\nREADME:")
print(readme_path)

print("\nPlaylistPulse Week 8 integration completed.")