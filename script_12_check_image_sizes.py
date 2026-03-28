import pandas as pd
from PIL import Image
import os

# ===============================
# 1. LOAD DATASET
# ===============================
df = pd.read_csv("Labels.csv")

# Remove unnecessary column if present
if "Unnamed: 4" in df.columns:
    df.drop(columns=["Unnamed: 4"], inplace=True)

# ===============================
# 2. CREATE IMAGE PATH
# ===============================
image_folder = "images"
df["image_path"] = df["Image Name"].apply(lambda x: os.path.join(image_folder, x))

# ===============================
# 3. CHECK FIRST 20 IMAGE SIZES
# ===============================
image_sizes = []

for img_path in df["image_path"].head(20):
    if os.path.exists(img_path):
        with Image.open(img_path) as img:
            image_sizes.append(img.size)   # (width, height)
    else:
        image_sizes.append("Missing")

# ===============================
# 4. DISPLAY RESULTS
# ===============================
print("First 20 image sizes:\n")
print(image_sizes)

# ===============================
# 5. OPTIONAL: UNIQUE SIZE SUMMARY
# ===============================
unique_sizes = set([s for s in image_sizes if s != "Missing"])

print("\nUnique image sizes detected:")
print(unique_sizes)
