import pandas as pd
import os
 
df = pd.read_csv("Labels.csv")
if "Unnamed: 4" in df.columns:
    df = df.drop(columns=["Unnamed: 4"])
 
image_folder = "images"
df["image_path"] = df["Image Name"].apply(
lambda x: os.path.join(image_folder, x)
)
 
# Check if image exists
df["file_exists"] = df["image_path"].apply(os.path.exists)
 
print("Image file check summary:\n")
print(df["file_exists"].value_counts())
