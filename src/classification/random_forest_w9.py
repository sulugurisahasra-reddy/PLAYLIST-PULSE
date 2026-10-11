import os
import pandas as pd
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
# WEEK 9 - RANDOM FOREST CLASSIFICATION
# Spotify Skip Prediction
# ============================================================

DATA_PATH = "src/data/spotify_history.csv"
REPORT_DIR = "reports/figures"

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading Spotify dataset...")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ============================================================
# 2. FEATURES AND TARGET
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

df = df[features + [target]].dropna()


# ============================================================
# 3. SAMPLE DATA
# ============================================================

sample_size = min(30000, len(df))

df = df.sample(
    n=sample_size,
    random_state=42
)

print("Dataset used:", df.shape)


# ============================================================
# 4. X AND Y
# ============================================================

X = df[features]
y = df[target].astype(int)


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. PREPROCESSING
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
# 7. ENCODE DATA
# ============================================================

print("\nEncoding data...")

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

print("Encoded training shape:", X_train_encoded.shape)


# ============================================================
# 8. RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_encoded,
    y_train
)

rf_pred = rf_model.predict(
    X_test_encoded
)


# ============================================================
# 9. MODEL RESULTS
# ============================================================

accuracy = accuracy_score(
    y_test,
    rf_pred
)

print("\n" + "=" * 60)
print("WEEK 9 - RANDOM FOREST RESULTS")
print("=" * 60)

print("Accuracy:", round(accuracy, 4))

print("OOB Score:", round(rf_model.oob_score_, 4))

print("OOB Error:", round(1 - rf_model.oob_score_, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_pred
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    rf_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Week 9 - Random Forest Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W9_Random_Forest_Confusion_Matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. NUMBER OF TREES ANALYSIS
# ============================================================

print("\nTesting different numbers of trees...")

tree_values = [
    50,
    100,
    150,
    200
]

tree_accuracy = []
tree_oob = []

for n_trees in tree_values:

    model = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_encoded,
        y_train
    )

    prediction = model.predict(
        X_test_encoded
    )

    tree_accuracy.append(
        accuracy_score(
            y_test,
            prediction
        )
    )

    tree_oob.append(
        model.oob_score_
    )

tree_results = pd.DataFrame({
    "Number_of_Trees": tree_values,
    "Test_Accuracy": tree_accuracy,
    "OOB_Score": tree_oob
})

print("\nNumber of Trees Results:")

print(
    tree_results.to_string(
        index=False
    )
)


# ============================================================
# 12. TREE COUNT GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    tree_values,
    tree_accuracy,
    marker="o",
    label="Test Accuracy"
)

plt.plot(
    tree_values,
    tree_oob,
    marker="o",
    label="OOB Score"
)

plt.xlabel("Number of Trees")
plt.ylabel("Score")

plt.title(
    "Week 9 - Random Forest Tree Count Analysis"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W9_Random_Forest_Tree_Count.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. FEATURE SUBSAMPLING
# ============================================================

print("\nTesting feature subsampling...")

feature_options = [
    "sqrt",
    "log2",
    None
]

feature_scores = []

for option in feature_options:

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        max_features=option,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_encoded,
        y_train
    )

    prediction = model.predict(
        X_test_encoded
    )

    feature_scores.append(
        accuracy_score(
            y_test,
            prediction
        )
    )

feature_results = pd.DataFrame({
    "Max_Features": [
        "sqrt",
        "log2",
        "all"
    ],
    "Accuracy": feature_scores
})

print("\nFeature Subsampling Results:")

print(
    feature_results.to_string(
        index=False
    )
)


# ============================================================
# 14. FEATURE SUBSAMPLING GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    feature_results["Max_Features"],
    feature_results["Accuracy"]
)

plt.xlabel("Feature Subsampling")
plt.ylabel("Accuracy")

plt.title(
    "Week 9 - Random Forest Feature Subsampling"
)

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W9_Random_Forest_Feature_Subsampling.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(15)
)

print("\nTop Random Forest Features:")

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 16. FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Week 9 - Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W9_Random_Forest_Feature_Importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. PERMUTATION IMPORTANCE
# ============================================================

print("\nCalculating permutation importance...")

perm = permutation_importance(
    rf_model,
    X_test_encoded,
    y_test,
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance_Mean": perm.importances_mean
})

permutation_df = (
    permutation_df
    .sort_values(
        "Importance_Mean",
        ascending=False
    )
    .head(15)
)

print("\nTop Permutation Importance Features:")

print(
    permutation_df.to_string(
        index=False
    )
)


# ============================================================
# 18. PERMUTATION IMPORTANCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    permutation_df["Feature"][::-1],
    permutation_df["Importance_Mean"][::-1]
)

plt.xlabel("Permutation Importance")
plt.ylabel("Feature")

plt.title(
    "Week 9 - Random Forest Permutation Importance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W9_Random_Forest_Permutation_Importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 19. SAVE RESULTS
# ============================================================

tree_results.to_csv(
    os.path.join(
        REPORT_DIR,
        "W9_Random_Forest_Tree_Results.csv"
    ),
    index=False
)

feature_results.to_csv(
    os.path.join(
        REPORT_DIR,
        "W9_Random_Forest_Feature_Subsampling.csv"
    ),
    index=False
)

importance_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "W9_Random_Forest_Importance.csv"
    ),
    index=False
)

permutation_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "W9_Random_Forest_Permutation_Importance.csv"
    ),
    index=False
)


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("WEEK 9 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nRandom Forest Accuracy:", round(accuracy, 4))
print("OOB Score:", round(rf_model.oob_score_, 4))
print("OOB Error:", round(1 - rf_model.oob_score_, 4))

print("\nGenerated files are saved in:")
print(REPORT_DIR)