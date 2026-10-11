import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance


# ============================================================
# WEEK 7 - RANDOM FOREST CLASSIFICATION
# Spotify Skip Prediction
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "src/data/spotify_history.csv"
REPORT_DIR = "reports/figures"

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading Spotify history dataset...")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "ms_played",
    "platform",
    "artist_name",
    "album_name",
    "reason_start",
    "reason_end",
    "shuffle"
]

target = "skipped"


# Keep only required columns
df = df[features + [target]].dropna()


print("Dataset after removing missing values:", df.shape)


# ============================================================
# 4. SAMPLE DATA
# ============================================================
# Sampling makes training faster while preserving
# the overall structure of the dataset.

sample_size = min(30000, len(df))

df = df.sample(
    n=sample_size,
    random_state=42
)

print("Dataset used for training:", df.shape)


# ============================================================
# 5. SPLIT FEATURES AND TARGET
# ============================================================

X = df[features]
y = df[target]


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. PREPROCESSING
# ============================================================

categorical_features = [
    "platform",
    "artist_name",
    "album_name",
    "reason_start",
    "reason_end"
]

numeric_features = [
    "ms_played",
    "shuffle"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=20,
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# 8. ENCODE DATA
# ============================================================

print("\nEncoding categorical features...")

X_train_encoded = preprocessor.fit_transform(X_train)

X_test_encoded = preprocessor.transform(X_test)

print(
    "Encoded training shape:",
    X_train_encoded.shape
)

print(
    "Encoded testing shape:",
    X_test_encoded.shape
)


# ============================================================
# 9. RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    oob_score=True,
    random_state=42,
    n_jobs=-1,
    class_weight=None
)


rf_model.fit(
    X_train_encoded,
    y_train
)


# ============================================================
# 10. PREDICTION
# ============================================================

y_pred = rf_model.predict(
    X_test_encoded
)


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n")
print("=" * 60)
print("WEEK 7 - RANDOM FOREST RESULTS")
print("=" * 60)

print(
    "\nAccuracy:",
    round(accuracy, 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 12. OOB SCORE
# ============================================================

oob_score = rf_model.oob_score_

oob_error = 1 - oob_score

print(
    "OOB Score:",
    round(oob_score, 4)
)

print(
    "OOB Error:",
    round(oob_error, 4)
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\nCreating confusion matrix...")

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Week 7 - Random Forest Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W7_Random_Forest_Confusion_Matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. OOB ERROR VS NUMBER OF TREES
# ============================================================

print("\nTesting different numbers of trees...")

# Start from 50 trees so OOB estimates are more reliable
tree_values = [
    50,
    100,
    150,
    200
]

oob_errors = []


for n_trees in tree_values:

    print(
        f"Testing {n_trees} trees..."
    )

    rf_test = RandomForestClassifier(
        n_estimators=n_trees,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    rf_test.fit(
        X_train_encoded,
        y_train
    )

    error = 1 - rf_test.oob_score_

    oob_errors.append(error)


# Plot OOB error

plt.figure(figsize=(8, 5))

plt.plot(
    tree_values,
    oob_errors,
    marker="o"
)

plt.xlabel(
    "Number of Trees"
)

plt.ylabel(
    "OOB Error"
)

plt.title(
    "Week 7 - Random Forest OOB Error"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W7_Random_Forest_OOB_Error.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. FEATURE SUBSAMPLING
# ============================================================

print("\nTesting feature subsampling strategies...")

max_features_values = [
    "sqrt",
    "log2",
    None
]

feature_scores = []


for value in max_features_values:

    if value is None:
        label = "None"
    else:
        label = value

    print(
        f"Testing max_features = {label}"
    )

    rf_test = RandomForestClassifier(
        n_estimators=100,
        max_features=value,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    rf_test.fit(
        X_train_encoded,
        y_train
    )

    feature_scores.append(
        rf_test.oob_score_
    )


# Plot feature subsampling

labels = [
    "sqrt",
    "log2",
    "None"
]

plt.figure(figsize=(8, 5))

plt.bar(
    labels,
    feature_scores
)

plt.xlabel(
    "Feature Subsampling Strategy"
)

plt.ylabel(
    "OOB Score"
)

plt.title(
    "Week 7 - Feature Subsampling Comparison"
)

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W7_Random_Forest_Feature_Subsampling.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. PERMUTATION IMPORTANCE
# ============================================================

print("\nCalculating permutation importance...")

permutation = permutation_importance(
    rf_model,
    X_test_encoded,
    y_test,
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
    scoring="accuracy"
)


# Get encoded feature names

feature_names = (
    preprocessor
    .get_feature_names_out()
)


# Create importance table

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": permutation.importances_mean
    }
)


# Sort by importance

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(15)
)


print("\nTop 15 Permutation Important Features:")

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 17. PERMUTATION IMPORTANCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1]
)

plt.xlabel(
    "Permutation Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Week 7 - Random Forest Permutation Importance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W7_Random_Forest_Permutation_Importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\nCalculating Random Forest feature importance...")

rf_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }
)


rf_importance = (
    rf_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(15)
)


print("\nTop 15 Random Forest Features:")

print(
    rf_importance.to_string(
        index=False
    )
)


# ============================================================
# 19. FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    rf_importance["Feature"][::-1],
    rf_importance["Importance"][::-1]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Week 7 - Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W7_Random_Forest_Feature_Importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 20. SAVE FEATURE IMPORTANCE TABLE
# ============================================================

importance_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "W7_Permutation_Importance.csv"
    ),
    index=False
)


rf_importance.to_csv(
    os.path.join(
        REPORT_DIR,
        "W7_Random_Forest_Importance.csv"
    ),
    index=False
)


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print("WEEK 7 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nModel Accuracy:")
print(round(accuracy, 4))

print("\nOOB Score:")
print(round(oob_score, 4))

print("\nOOB Error:")
print(round(oob_error, 4))

print("\nGenerated files:")

print(
    "1. W7_Random_Forest_Confusion_Matrix.png"
)

print(
    "2. W7_Random_Forest_OOB_Error.png"
)

print(
    "3. W7_Random_Forest_Feature_Subsampling.png"
)

print(
    "4. W7_Random_Forest_Permutation_Importance.png"
)

print(
    "5. W7_Random_Forest_Feature_Importance.png"
)

print(
    "6. W7_Permutation_Importance.csv"
)

print(
    "7. W7_Random_Forest_Importance.csv"
)

print(
    "\nAll files are saved in:"
)

print(
    REPORT_DIR
)