import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.cluster import KMeans


# ============================================================
# WEEK 11 - K-MEANS IMAGE SEGMENTATION
# ============================================================

IMAGE_PATH = "src/data/input_image.jpg"
REPORT_DIR = "reports/figures"

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD IMAGE
# ============================================================

print("\nLoading image...")

image = Image.open(IMAGE_PATH).convert("RGB")

image_array = np.array(image)

print("Image shape:", image_array.shape)


# ============================================================
# 2. PREPARE PIXELS
# ============================================================

pixels = image_array.reshape(
    -1,
    3
)

print("Number of pixels:", len(pixels))


# ============================================================
# 3. TEST DIFFERENT K VALUES
# ============================================================

k_values = [2, 4, 6, 8]

inertias = []


for k in k_values:

    print("\n" + "=" * 50)
    print("Processing K =", k)
    print("=" * 50)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(pixels)

    centers = kmeans.cluster_centers_

    segmented_pixels = centers[labels]

    segmented_image = segmented_pixels.reshape(
        image_array.shape
    ).astype(np.uint8)


    # ========================================================
    # SAVE SEGMENTED IMAGE
    # ========================================================

    output_path = os.path.join(
        REPORT_DIR,
        f"W11_KMeans_Segmented_K{k}.png"
    )

    Image.fromarray(
        segmented_image
    ).save(output_path)

    print(
        "Saved:",
        output_path
    )


    # ========================================================
    # STORE INERTIA
    # ========================================================

    inertias.append(
        kmeans.inertia_
    )


# ============================================================
# 4. ELBOW GRAPH
# ============================================================

print("\nCreating elbow graph...")

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.title(
    "Week 11 - K-Means Elbow Method"
)

plt.xticks(k_values)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W11_KMeans_Elbow.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. COMPARISON IMAGE
# ============================================================

print("\nCreating comparison image...")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

for ax, k in zip(
    axes.ravel(),
    k_values
):

    segmented_path = os.path.join(
        REPORT_DIR,
        f"W11_KMeans_Segmented_K{k}.png"
    )

    segmented = Image.open(
        segmented_path
    )

    ax.imshow(segmented)

    ax.set_title(
        f"K = {k}"
    )

    ax.axis("off")


plt.suptitle(
    "Week 11 - K-Means Image Segmentation"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "W11_KMeans_Comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. SAVE RESULTS
# ============================================================

results = []

for k, inertia in zip(
    k_values,
    inertias
):

    results.append({
        "K": k,
        "Inertia": inertia
    })


import pandas as pd

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "W11_KMeans_Results.csv"
    ),
    index=False
)


# ============================================================
# 7. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("WEEK 11 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nK-Means Results:")

print(
    results_df.to_string(
        index=False
    )
)

print("\nGenerated files:")

print("1. W11_KMeans_Segmented_K2.png")
print("2. W11_KMeans_Segmented_K4.png")
print("3. W11_KMeans_Segmented_K6.png")
print("4. W11_KMeans_Segmented_K8.png")
print("5. W11_KMeans_Elbow.png")
print("6. W11_KMeans_Comparison.png")
print("7. W11_KMeans_Results.csv")

print("\nAll files saved in:")
print(REPORT_DIR)