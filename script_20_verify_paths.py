import pandas as pd
import os

# ===============================
# 1. LOAD DATASET
# ===============================
df = pd.read_csv("train_dataset.csv")

# ===============================
# 2. DEFINE IMAGE FOLDER
# ===============================
image_folder = "images_resized"

# ===============================
# 3. VERIFY IMAGE PATHS
# ===============================
missing_images = []

for img_name in df["Image Name"]:
    img_path = os.path.join(image_folder, img_name)

    if not os.path.exists(img_path):
        missing_images.append(img_name)

# ===============================
# 4. REPORT RESULTS
# ===============================
total_images = len(df)
missing_count = len(missing_images)

print("Total images checked:", total_images)
print("Missing images:", missing_count)

if missing_count > 0:
    print("\nMissing image files:")
    for img in missing_images[:20]:   # show first 20 only
        print(img)
else:
    print("\nAll image files exist.")

# ===============================
# 5. SAVE REPORT
# ===============================
if missing_count > 0:
    pd.DataFrame(missing_images, columns=["Missing Images"]).to_csv(
        "missing_images_report.csv",
        index=False
    )
    print("\nMissing image report saved to missing_images_report.csv")
