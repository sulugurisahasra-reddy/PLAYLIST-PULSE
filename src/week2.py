import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


TRACKS_PATH = "src/data/dataset.csv"
HISTORY_PATH = "src/data/spotify_history.csv"

REPORT_DIR = "reports/week2"

os.makedirs(REPORT_DIR, exist_ok=True)


print("=" * 70)
print("PLAYLISTPULSE - WEEK 2")
print("Data Cleaning, Validation and Leakage-Safe Pipeline")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("\n[1] Loading datasets...")

tracks = pd.read_csv(TRACKS_PATH)
history = pd.read_csv(HISTORY_PATH)

print(f"Tracks shape: {tracks.shape}")
print(f"History shape: {history.shape}")


# ------------------------------------------------------------
# 2. BASIC SCHEMA VALIDATION
# ------------------------------------------------------------

print("\n[2] Validating schemas...")

required_track_columns = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "popularity",
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

required_history_columns = [
    "spotify_track_uri",
    "ts",
    "platform",
    "ms_played",
    "track_name",
    "artist_name",
    "album_name",
    "reason_start",
    "reason_end",
    "shuffle",
    "skipped"
]

missing_track_columns = [
    col for col in required_track_columns
    if col not in tracks.columns
]

missing_history_columns = [
    col for col in required_history_columns
    if col not in history.columns
]

if missing_track_columns:
    raise ValueError(
        f"Missing track columns: {missing_track_columns}"
    )

if missing_history_columns:
    raise ValueError(
        f"Missing history columns: {missing_history_columns}"
    )

print("Track schema: VALID")
print("History schema: VALID")


# ------------------------------------------------------------
# 3. COPY DATA
# ------------------------------------------------------------

tracks_clean = tracks.copy()
history_clean = history.copy()


# ------------------------------------------------------------
# 4. CLEAN TRACK IDs
# ------------------------------------------------------------

print("\n[3] Cleaning track identifiers...")

tracks_clean["track_id"] = (
    tracks_clean["track_id"]
    .astype("string")
    .str.strip()
)

history_clean["track_id"] = (
    history_clean["spotify_track_uri"]
    .astype("string")
    .str.strip()
)

tracks_clean = tracks_clean.dropna(
    subset=["track_id"]
)

history_clean = history_clean.dropna(
    subset=["track_id"]
)

tracks_clean = tracks_clean[
    tracks_clean["track_id"] != ""
]

history_clean = history_clean[
    history_clean["track_id"] != ""
]

print(
    f"Valid track IDs in tracks: "
    f"{tracks_clean['track_id'].notna().sum()}"
)

print(
    f"Valid track IDs in history: "
    f"{history_clean['track_id'].notna().sum()}"
)


# ------------------------------------------------------------
# 5. REMOVE DUPLICATE TRACK RECORDS
# ------------------------------------------------------------

print("\n[4] Removing duplicate track records...")

track_rows_before = len(tracks_clean)

tracks_clean = tracks_clean.drop_duplicates(
    subset=["track_id"],
    keep="first"
)

track_rows_after = len(tracks_clean)

print(
    f"Removed duplicate track rows: "
    f"{track_rows_before - track_rows_after}"
)

print(
    f"Unique track records remaining: "
    f"{track_rows_after}"
)


# ------------------------------------------------------------
# 6. REMOVE EXACT DUPLICATE HISTORY EVENTS
# ------------------------------------------------------------

print("\n[5] Removing exact duplicate history events...")

history_rows_before = len(history_clean)

history_clean = history_clean.drop_duplicates()

history_rows_after = len(history_clean)

print(
    f"Removed duplicate history rows: "
    f"{history_rows_before - history_rows_after}"
)

print(
    f"History rows remaining: "
    f"{history_rows_after}"
)


# ------------------------------------------------------------
# 7. DATETIME VALIDATION
# ------------------------------------------------------------

print("\n[6] Validating timestamps...")

history_clean["ts"] = pd.to_datetime(
    history_clean["ts"],
    errors="coerce"
)

invalid_timestamps = history_clean["ts"].isna().sum()

print(
    f"Invalid timestamps: {invalid_timestamps}"
)


# ------------------------------------------------------------
# 8. NUMERIC TYPE VALIDATION
# ------------------------------------------------------------

print("\n[7] Validating numeric features...")

numeric_track_columns = [
    "popularity",
    "duration_ms",
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

numeric_history_columns = [
    "ms_played"
]

for col in numeric_track_columns:
    tracks_clean[col] = pd.to_numeric(
        tracks_clean[col],
        errors="coerce"
    )

for col in numeric_history_columns:
    history_clean[col] = pd.to_numeric(
        history_clean[col],
        errors="coerce"
    )

print("Numeric conversion completed.")


# ------------------------------------------------------------
# 9. MISSING VALUE REPORT
# ------------------------------------------------------------

print("\n[8] Creating missing-value report...")

track_missing = (
    tracks_clean.isna()
    .sum()
    .reset_index()
)

track_missing.columns = [
    "column",
    "missing_count"
]

track_missing["missing_percent"] = (
    track_missing["missing_count"]
    / len(tracks_clean)
    * 100
)

history_missing = (
    history_clean.isna()
    .sum()
    .reset_index()
)

history_missing.columns = [
    "column",
    "missing_count"
]

history_missing["missing_percent"] = (
    history_missing["missing_count"]
    / len(history_clean)
    * 100
)

track_missing.to_csv(
    f"{REPORT_DIR}/track_missing_values.csv",
    index=False
)

history_missing.to_csv(
    f"{REPORT_DIR}/history_missing_values.csv",
    index=False
)


# ------------------------------------------------------------
# 10. HANDLE INVALID HISTORY VALUES
# ------------------------------------------------------------

print("\n[9] Cleaning invalid history values...")

history_clean = history_clean[
    history_clean["ms_played"].ge(0)
    | history_clean["ms_played"].isna()
]

history_clean["skipped"] = pd.to_numeric(
    history_clean["skipped"],
    errors="coerce"
)

history_clean = history_clean[
    history_clean["skipped"].isin([0, 1])
    | history_clean["skipped"].isna()
]

print(
    f"History rows after validation: "
    f"{len(history_clean)}"
)


# ------------------------------------------------------------
# 11. AUDIO FEATURE RANGE VALIDATION
# ------------------------------------------------------------

print("\n[10] Validating audio feature ranges...")

range_rules = {
    "danceability": (0, 1),
    "energy": (0, 1),
    "speechiness": (0, 1),
    "acousticness": (0, 1),
    "instrumentalness": (0, 1),
    "liveness": (0, 1),
    "valence": (0, 1)
}

range_results = []

for column, (lower, upper) in range_rules.items():

    invalid_count = (
        (
            (tracks_clean[column] < lower)
            | (tracks_clean[column] > upper)
        )
        & tracks_clean[column].notna()
    ).sum()

    range_results.append({
        "feature": column,
        "minimum_allowed": lower,
        "maximum_allowed": upper,
        "invalid_count": int(invalid_count)
    })

range_validation = pd.DataFrame(range_results)

range_validation.to_csv(
    f"{REPORT_DIR}/feature_range_validation.csv",
    index=False
)

print(range_validation.to_string(index=False))


# ------------------------------------------------------------
# 12. OUTLIER DETECTION USING IQR
# ------------------------------------------------------------

print("\n[11] Detecting outliers using IQR...")

outlier_columns = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

outlier_results = []

for column in outlier_columns:

    series = tracks_clean[column].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    count = (
        (series < lower_bound)
        | (series > upper_bound)
    ).sum()

    outlier_results.append({
        "feature": column,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": int(count),
        "outlier_percent": (
            count / len(series) * 100
            if len(series) > 0
            else 0
        )
    })

outlier_summary = pd.DataFrame(outlier_results)

outlier_summary.to_csv(
    f"{REPORT_DIR}/outlier_summary.csv",
    index=False
)

print(outlier_summary.to_string(index=False))


# ------------------------------------------------------------
# 13. CREATE TARGETS
# ------------------------------------------------------------

print("\n[12] Creating Week 2 targets...")

history_clean["skip_30s"] = (
    (history_clean["skipped"] == 1)
    & (history_clean["ms_played"] <= 30000)
).astype(int)

track_history = (
    history_clean
    .groupby("track_id")
    .agg(
        listen_count=("track_id", "size"),
        stream_count=("track_id", "size"),
        skip_30s_count=("skip_30s", "sum")
    )
    .reset_index()
)

track_history["skip_30s_rate"] = (
    track_history["skip_30s_count"]
    / track_history["listen_count"]
)

track_history["log_stream_count"] = np.log1p(
    track_history["stream_count"]
)

print(
    f"Track-level history records: "
    f"{len(track_history)}"
)


# ------------------------------------------------------------
# 14. MERGE TRACK + HISTORY DATA
# ------------------------------------------------------------

print("\n[13] Creating validated modeling dataset...")

track_columns_for_model = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "popularity",
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

model_data = tracks_clean[
    track_columns_for_model
].merge(
    track_history,
    on="track_id",
    how="inner",
    validate="one_to_one"
)

print(
    f"Final modeling dataset shape: "
    f"{model_data.shape}"
)


# ------------------------------------------------------------
# 15. REMOVE LEAKAGE-PRONE PREDICTORS
# ------------------------------------------------------------

print("\n[14] Applying leakage protection...")

leakage_columns = [
    "track_id",
    "stream_count",
    "skip_30s_count",
    "listen_count",
    "skip_30s_rate",
    "log_stream_count"
]

print("Target/history-derived columns protected from predictors:")

for column in leakage_columns:
    print(f"  - {column}")


# ------------------------------------------------------------
# 16. SAVE CLEAN DATA
# ------------------------------------------------------------

print("\n[15] Saving cleaned datasets...")

tracks_clean.to_csv(
    f"{REPORT_DIR}/clean_tracks.csv",
    index=False
)

history_clean.to_csv(
    f"{REPORT_DIR}/clean_history.csv",
    index=False
)

model_data.to_csv(
    f"{REPORT_DIR}/clean_model_data.csv",
    index=False
)


# ------------------------------------------------------------
# 17. BUILD LEAKAGE-SAFE PREPROCESSING PIPELINE
# ------------------------------------------------------------

print("\n[16] Building preprocessing pipeline...")

feature_columns = [
    "artists",
    "album_name",
    "track_name",
    "popularity",
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

numeric_features = [
    "popularity",
    "duration_ms",
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
    "artists",
    "album_name",
    "track_name",
    "explicit",
    "track_genre"
]

numeric_pipeline = Pipeline(
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

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2
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


# ------------------------------------------------------------
# 18. PIPELINE VALIDATION
# ------------------------------------------------------------

print("\n[17] Validating preprocessing pipeline...")

X = model_data[feature_columns]

preprocessor.fit(X)

X_transformed = preprocessor.transform(X)

print(
    f"Original feature columns: {X.shape[1]}"
)

print(
    f"Transformed feature dimensions: "
    f"{X_transformed.shape}"
)

print(
    "Leakage-safe preprocessing pipeline: READY"
)


# ------------------------------------------------------------
# 19. DATA QUALITY SUMMARY
# ------------------------------------------------------------

summary = {
    "tracks_original_rows": len(tracks),
    "tracks_clean_rows": len(tracks_clean),
    "history_original_rows": len(history),
    "history_clean_rows": len(history_clean),
    "unique_track_ids": tracks_clean["track_id"].nunique(),
    "history_unique_track_ids": history_clean["track_id"].nunique(),
    "matched_track_ids": model_data["track_id"].nunique(),
    "model_rows": len(model_data),
    "model_columns": model_data.shape[1],
    "missing_track_cells": int(tracks_clean.isna().sum().sum()),
    "missing_history_cells": int(history_clean.isna().sum().sum()),
    "skip_30s_events": int(history_clean["skip_30s"].sum()),
    "stream_count_min": int(model_data["stream_count"].min()),
    "stream_count_max": int(model_data["stream_count"].max())
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["metric", "value"]
)

summary_df.to_csv(
    f"{REPORT_DIR}/week2_summary.csv",
    index=False
)


print("\n" + "=" * 70)
print("WEEK 2 COMPLETED")
print("=" * 70)

print("\nDATA QUALITY SUMMARY")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nGenerated reports:")

for file in sorted(os.listdir(REPORT_DIR)):
    print(f"  - {file}")

print("\nLeakage protection:")
print("  Target variables are not used as model predictors.")
print("  ms_played is used only to construct skip_30s.")
print("  History-derived stream/skip aggregates are excluded from X.")

print("\nWeek 2 preprocessing pipeline is ready for Week 3.")

