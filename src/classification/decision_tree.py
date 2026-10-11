import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "src/data/spotify_history.csv"
REPORT_DIR = "reports/figures"

os.makedirs(REPORT_DIR, exist_ok=True)


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# -----------------------------
# Select features and target
# -----------------------------
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


# -----------------------------
# Sample data for faster training
# -----------------------------
df = df.sample(
    n=min(30000, len(df)),
    random_state=42
)


X = df[features]
y = df[target]


# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Preprocessing
# -----------------------------
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
                min_frequency=20
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


# -----------------------------
# Gini Decision Tree
# -----------------------------
gini_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=6,
                random_state=42
            )
        )
    ]
)

gini_model.fit(X_train, y_train)

y_pred = gini_model.predict(X_test)

print("\n===== GINI DECISION TREE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()
plt.title("Week 6 - Decision Tree Confusion Matrix")
plt.tight_layout()

plt.savefig(
    f"{REPORT_DIR}/W6_Decision_Tree_Confusion_Matrix.png",
    dpi=300
)

plt.close()


# -----------------------------
# Tree visualization
# -----------------------------
X_train_encoded = gini_model.named_steps[
    "preprocessor"
].transform(X_train)

feature_names = gini_model.named_steps[
    "preprocessor"
].get_feature_names_out()

tree_model = gini_model.named_steps["classifier"]

plt.figure(figsize=(20, 10))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Skipped", "Skipped"],
    filled=True,
    max_depth=3,
    fontsize=7
)

plt.title("Week 6 - Decision Tree")
plt.tight_layout()

plt.savefig(
    f"{REPORT_DIR}/W6_Decision_Tree.png",
    dpi=300
)

plt.close()


# -----------------------------
# Feature importance
# -----------------------------
importance = tree_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
).head(15)

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Week 6 - Decision Tree Feature Importance")

plt.tight_layout()

plt.savefig(
    f"{REPORT_DIR}/W6_Decision_Tree_Feature_Importance.png",
    dpi=300
)

plt.close()


print("\nWeek 6 completed successfully.")
print("Graphs saved in:", REPORT_DIR)