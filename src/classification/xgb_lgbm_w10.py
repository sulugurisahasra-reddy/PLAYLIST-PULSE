import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import shap


# ============================================================
# WEEK 10 - XGBOOST AND LIGHTGBM
# Spotify Skip Prediction
# ============================================================

DATA_PATH = "src/data/spotify_history.csv"
REPORT_DIR = "reports/figures"

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
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
# 8. XGBOOST
# ============================================================

print("\nTraining XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_encoded,
    y_train
)

xgb_pred = xgb_model.predict(
    X_test_encoded
)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_pred
)

print("\n" + "=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)

print("Accuracy:", round(xgb_accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        xgb_pred
    )
)


# ============================================================
# 9. XGBOOST CONFUSION MATRIX
# ============================================================

cm_xgb = confusion_matrix(
    y_test,
    xgb_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_xgb
)

disp.plot()

plt.title("Week 10 - XGBoost Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W10_XGBoost_Confusion_Matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. LIGHTGBM
# ============================================================

print("\nTraining LightGBM...")

lgbm_model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)

lgbm_model.fit(
    X_train_encoded,
    y_train
)

lgbm_pred = lgbm_model.predict(
    X_test_encoded
)

lgbm_accuracy = accuracy_score(
    y_test,
    lgbm_pred
)

print("\n" + "=" * 60)
print("LIGHTGBM RESULTS")
print("=" * 60)

print("Accuracy:", round(lgbm_accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        lgbm_pred
    )
)


# ============================================================
# 11. LIGHTGBM CONFUSION MATRIX
# ============================================================

cm_lgbm = confusion_matrix(
    y_test,
    lgbm_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_lgbm
)

disp.plot()

plt.title("Week 10 - LightGBM Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W10_LightGBM_Confusion_Matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "XGBoost",
        "LightGBM"
    ],
    "Accuracy": [
        xgb_accuracy,
        lgbm_accuracy
    ]
})

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 13. MODEL COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.title(
    "Week 10 - XGBoost vs LightGBM"
)

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W10_XGBoost_vs_LightGBM.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. XGBOOST FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": xgb_model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(15)
)

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Week 10 - XGBoost Feature Importance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W10_XGBoost_Feature_Importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. SHAP EXPLANATION
# ============================================================

print("\nGenerating SHAP explanation...")

explainer = shap.TreeExplainer(
    xgb_model
)

shap_values = explainer.shap_values(
    X_test_encoded[:1000]
)

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_encoded[:1000],
    feature_names=feature_names,
    show=False,
    max_display=15
)

plt.title(
    "Week 10 - XGBoost SHAP Summary"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W10_XGBoost_SHAP_Summary.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. SAVE RESULTS
# ============================================================

comparison.to_csv(
    os.path.join(
        REPORT_DIR,
        "W10_XGBoost_LightGBM_Comparison.csv"
    ),
    index=False
)

importance_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "W10_XGBoost_Feature_Importance.csv"
    ),
    index=False
)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("WEEK 10 COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nXGBoost Accuracy:",
    round(xgb_accuracy, 4)
)

print(
    "LightGBM Accuracy:",
    round(lgbm_accuracy, 4)
)

print("\nAll W10 files saved in:")
print(REPORT_DIR)