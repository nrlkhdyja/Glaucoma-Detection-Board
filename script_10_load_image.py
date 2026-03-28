import pandas as pd
from PIL import Image
import os
 
df = pd.read_csv("Labels.csv")
if "Unnamed: 4" in df.columns:
    df = df.drop(columns=["Unnamed: 4"])
df["image_path"] = df["Image Name"].apply(
    lambda x: os.path.join("images", x)
)
 
# Select first image
image_path = df.iloc[0]["image_path"]
img = Image.open(image_path)
 
print("Image path:", image_path)
print("Image size:", img.size)
print("Image mode:", img.mode)
