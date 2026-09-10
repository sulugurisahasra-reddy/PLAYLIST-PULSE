from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRACKS_PATH = "src/data/dataset.csv"
HISTORY_PATH = "src/data/spotify_history.csv"

FIG_DIR = Path("reports/figures")
OUT_DIR = Path("reports/week1")

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PLAYLISTPULSE - WEEK 1")
print("Target Definition + Initial EDA")
print("=" * 70)

# =========================================================
# 1. LOAD DATA
# =========================================================

print("\n[1] Loading datasets...")

tracks = pd.read_csv(TRACKS_PATH)
history = pd.read_csv(HISTORY_PATH)

print(f"Tracks shape   : {tracks.shape}")
print(f"History shape  : {history.shape}")

# =========================================================
# 2. BASIC VALIDATION
# =========================================================

print("\n[2] Basic validation")

print("\nTracks missing values:")
track_missing = tracks.isnull().sum()
print(track_missing[track_missing > 0])

print("\nHistory missing values:")
history_missing = history.isnull().sum()
print(history_missing[history_missing > 0])

print("\nDuplicate track IDs:", tracks["track_id"].duplicated().sum())
print("Duplicate history rows:", history.duplicated().sum())

# =========================================================
# 3. CLEAN TRACK DATA
# =========================================================

print("\n[3] Cleaning track IDs...")

tracks["track_id"] = tracks["track_id"].astype("string").str.strip()

tracks = tracks.dropna(subset=["track_id"])

tracks = tracks.drop_duplicates(
    subset=["track_id"],
    keep="first"
)

print(
    "Unique track IDs after deduplication:",
    tracks["track_id"].nunique()
)

# =========================================================
# 4. CONNECT HISTORY TO TRACK DATA
# =========================================================

print("\n[4] Connecting streaming history to track dataset...")

history["track_id"] = (
    history["spotify_track_uri"]
    .astype("string")
    .str.strip()
)

history = history.dropna(subset=["track_id"])

print(
    "History rows with valid track IDs:",
    len(history)
)

print(
    "Unique track IDs in history:",
    history["track_id"].nunique()
)

print(
    "Unique track IDs in tracks dataset:",
    tracks["track_id"].nunique()
)

matched_ids = set(history["track_id"]) & set(tracks["track_id"])

print(
    "Track IDs present in BOTH datasets:",
    len(matched_ids)
)

print(
    "Percentage of history track IDs matched:",
    round(
        100 * len(matched_ids) / history["track_id"].nunique(),
        2
    ),
    "%"
)

# =========================================================
# 5. CREATE skip_30s TARGET
# =========================================================

print("\n[5] Creating classification target: skip_30s")

history["skipped"] = pd.to_numeric(
    history["skipped"],
    errors="coerce"
)

history["ms_played"] = pd.to_numeric(
    history["ms_played"],
    errors="coerce"
)

history["skip_30s"] = (
    (history["skipped"] == 1)
    & (history["ms_played"] <= 30000)
).astype(int)

print("\nskip_30s distribution:")
print(
    history["skip_30s"]
    .value_counts()
    .sort_index()
)

print("\nskip_30s percentage:")
print(
    (
        history["skip_30s"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)

# =========================================================
# 6. CREATE STREAM COUNT
# =========================================================

print("\n[6] Creating stream count target...")

stream_counts = (
    history
    .groupby("track_id")
    .size()
    .reset_index(name="stream_count")
)

print(
    "Unique tracks with listening history:",
    len(stream_counts)
)

print("\nStream count statistics:")
print(
    stream_counts["stream_count"].describe()
)

# =========================================================
# 7. LOG STREAM COUNT
# =========================================================

stream_counts["log_stream_count"] = np.log1p(
    stream_counts["stream_count"]
)

# =========================================================
# 8. TRACK-LEVEL SKIP BEHAVIOR
# =========================================================

print("\n[7] Creating track-level skip behavior...")

skip_rates = (
    history
    .groupby("track_id")["skip_30s"]
    .agg(
        skip_30s_rate="mean",
        skip_30s_count="sum",
        listen_count="count"
    )
    .reset_index()
)

skip_rates["skip_30s_majority"] = (
    skip_rates["skip_30s_rate"] >= 0.5
).astype(int)

print("\nTrack-level skip rate statistics:")
print(
    skip_rates["skip_30s_rate"].describe()
)

# =========================================================
# 9. MERGE DATA
# =========================================================

print("\n[8] Merging track features with listening behavior...")

track_data = tracks.merge(
    stream_counts,
    on="track_id",
    how="inner",
    validate="one_to_one"
)

track_data = track_data.merge(
    skip_rates[
        [
            "track_id",
            "skip_30s_rate",
            "skip_30s_count",
            "listen_count",
            "skip_30s_majority"
        ]
    ],
    on="track_id",
    how="left",
    validate="one_to_one"
)

print(
    "Final track-level dataset shape:",
    track_data.shape
)

print(
    "Tracks successfully matched:",
    len(track_data)
)

# =========================================================
# 10. LOG STREAM COUNT DISTRIBUTION
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    track_data["log_stream_count"],
    bins=40
)

plt.xlabel("log(1 + stream_count)")
plt.ylabel("Number of Tracks")
plt.title("Distribution of Log Stream Count")

plt.tight_layout()

plt.savefig(
    FIG_DIR / "week1_log_stream_count_distribution.png",
    dpi=150
)

plt.close()

# =========================================================
# 11. SKIP TARGET DISTRIBUTION
# =========================================================

skip_distribution = (
    history["skip_30s"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(8, 5))

plt.bar(
    ["Not Skipped Within 30s", "Skipped Within 30s"],
    [
        skip_distribution.get(0, 0),
        skip_distribution.get(1, 0)
    ]
)

plt.xlabel("Target")
plt.ylabel("Number of Listening Events")
plt.title("Skip Within 30 Seconds Distribution")

plt.tight_layout()

plt.savefig(
    FIG_DIR / "week1_skip_30s_distribution.png",
    dpi=150
)

plt.close()

# =========================================================
# 12. AUDIO FEATURE DISTRIBUTIONS
# =========================================================

audio_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

available_audio = [
    feature
    for feature in audio_features
    if feature in track_data.columns
]

print("\nAudio features being analyzed:")
print(available_audio)

for feature in available_audio:

    values = pd.to_numeric(
        track_data[feature],
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        continue

    plt.figure(figsize=(8, 5))

    plt.hist(
        values,
        bins=40
    )

    plt.xlabel(feature)
    plt.ylabel("Number of Tracks")
    plt.title(f"Distribution of {feature}")

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / f"week1_{feature}_distribution.png",
        dpi=150
    )

    plt.close()

# =========================================================
# 13. AUDIO FEATURES VS LOG STREAM COUNT
# =========================================================

correlation_data = track_data[
    available_audio + ["log_stream_count"]
].copy()

correlations = (
    correlation_data
    .corr(numeric_only=True)["log_stream_count"]
    .drop("log_stream_count")
    .dropna()
    .sort_values()
)

print("\nAudio feature correlations with log_stream_count:")
print(correlations)

if len(correlations) > 0:

    plt.figure(figsize=(10, 6))

    correlations.plot(kind="bar")

    plt.xlabel("Audio Feature")
    plt.ylabel("Correlation")
    plt.title("Audio Features vs Log Stream Count")

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "week1_audio_vs_log_stream_correlation.png",
        dpi=150
    )

    plt.close()

# =========================================================
# 14. TOP ARTISTS
# =========================================================

print("\n[9] Artist analysis")

artist_data = track_data.dropna(
    subset=["artists", "stream_count"]
).copy()

artist_data["artists"] = (
    artist_data["artists"]
    .astype(str)
    .str.strip()
)

artist_data = artist_data[
    artist_data["artists"] != ""
]

artist_streams = (
    artist_data
    .groupby("artists")["stream_count"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

print("\nTop 15 artists by total stream count:")
print(artist_streams)

if len(artist_streams) > 0:

    plt.figure(figsize=(10, 7))

    artist_streams.sort_values().plot(
        kind="barh"
    )

    plt.xlabel("Total Stream Count")
    plt.ylabel("Artist")
    plt.title("Top 15 Artists by Total Stream Count")

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "week1_top_artists_stream_count.png",
        dpi=150
    )

    plt.close()

# =========================================================
# 15. POPULARITY VS STREAM COUNT
# =========================================================

if "popularity" in track_data.columns:

    plot_data = track_data[
        ["popularity", "log_stream_count"]
    ].dropna()

    plt.figure(figsize=(8, 6))

    plt.scatter(
        plot_data["popularity"],
        plot_data["log_stream_count"],
        alpha=0.25
    )

    plt.xlabel("Spotify Popularity")
    plt.ylabel("log(1 + stream_count)")
    plt.title("Popularity vs Log Stream Count")

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "week1_popularity_vs_stream_count.png",
        dpi=150
    )

    plt.close()

# =========================================================
# 16. OUTLIER ANALYSIS
# =========================================================

print("\n[10] Outlier analysis")

outlier_features = available_audio + ["stream_count"]

outlier_summary = []

for feature in outlier_features:

    values = pd.to_numeric(
        track_data[feature],
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (
        (values < lower)
        | (values > upper)
    ).sum()

    outlier_summary.append(
        {
            "feature": feature,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(outliers),
            "outlier_percentage":
                100 * outliers / len(values)
        }
    )

outlier_df = pd.DataFrame(
    outlier_summary
)

print(outlier_df)

outlier_df.to_csv(
    OUT_DIR / "outlier_summary.csv",
    index=False
)

# =========================================================
# 17. MISSING VALUES
# =========================================================

missing_summary = pd.DataFrame(
    {
        "column": track_data.columns,
        "missing_count":
            track_data.isnull().sum().values,
        "missing_percentage":
            track_data.isnull().mean().values * 100
    }
)

missing_summary = missing_summary[
    missing_summary["missing_count"] > 0
]

print("\nMissing-value summary:")
print(missing_summary)

missing_summary.to_csv(
    OUT_DIR / "missing_value_summary.csv",
    index=False
)

# =========================================================
# 18. SUMMARY
# =========================================================

summary = pd.DataFrame(
    {
        "metric": [
            "tracks_rows",
            "tracks_columns",
            "history_rows",
            "history_columns",
            "unique_track_ids",
            "history_unique_track_ids",
            "matched_track_ids",
            "tracks_with_history",
            "skip_30s_events",
            "non_skip_30s_events",
            "mean_stream_count",
            "median_stream_count",
            "mean_log_stream_count",
            "median_log_stream_count"
        ],
        "value": [
            len(tracks),
            len(tracks.columns),
            len(history),
            len(history.columns),
            tracks["track_id"].nunique(),
            history["track_id"].nunique(),
            len(matched_ids),
            len(track_data),
            int(history["skip_30s"].sum()),
            int((history["skip_30s"] == 0).sum()),
            track_data["stream_count"].mean(),
            track_data["stream_count"].median(),
            track_data["log_stream_count"].mean(),
            track_data["log_stream_count"].median()
        ]
    }
)

summary.to_csv(
    OUT_DIR / "week1_summary.csv",
    index=False
)

# =========================================================
# 19. SAVE DATASETS
# =========================================================

track_data.to_csv(
    OUT_DIR / "track_level_dataset.csv",
    index=False
)

history.to_csv(
    OUT_DIR / "history_with_targets.csv",
    index=False
)

# =========================================================
# 20. UNCERTAINTY PROXY
# =========================================================

mixed_tracks = skip_rates[
    (skip_rates["skip_30s_rate"] > 0)
    & (skip_rates["skip_30s_rate"] < 1)
]

if len(skip_rates) > 0:

    weighted_uncertainty = (
        mixed_tracks["listen_count"].sum()
        / skip_rates["listen_count"].sum()
    )

else:

    weighted_uncertainty = np.nan

print("\n[11] Initial uncertainty analysis")

print(
    "Tracks with both skip and non-skip outcomes:",
    len(mixed_tracks)
)

if not pd.isna(weighted_uncertainty):

    print(
        "Fraction of listening events belonging to mixed-outcome tracks:",
        round(weighted_uncertainty * 100, 2),
        "%"
    )

else:

    print(
        "Fraction of listening events belonging to mixed-outcome tracks: N/A"
    )

print(
    "\nNote: This is an empirical uncertainty proxy, "
    "not a mathematically exact irreducible-error estimate."
)

# =========================================================
# 21. FINAL
# =========================================================

print("\n" + "=" * 70)
print("WEEK 1 COMPLETE")
print("=" * 70)

print("\nCreated:")
print("- reports/week1/track_level_dataset.csv")
print("- reports/week1/history_with_targets.csv")
print("- reports/week1/week1_summary.csv")
print("- reports/week1/missing_value_summary.csv")
print("- reports/week1/outlier_summary.csv")

print("\nGraphs saved in: reports/figures/")

print("\nMain targets:")
print("- Classification: skip_30s")
print("- Regression: log_stream_count")

print("\nLeakage protection:")
print("- ms_played is used only to construct skip_30s.")
print("- skipped is used only to construct skip_30s.")
print("- stream_count is the regression target, not a predictor.")

print("\nWEEK 1 ANALYSIS FINISHED.")
