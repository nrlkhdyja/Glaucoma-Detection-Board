import pandas as pd
import os
 
# Load dataset
df = pd.read_csv("Labels.csv")
 
# Remove unnecessary column if present
"Unnamed: 4" in df.columns and df.drop(columns=["Unnamed: 4"], inplace=True)
 
# Define image folder
image_folder = "images"

# Create full image path
df["image_path"] = df["Image Name"].apply(
lambda x: os.path.join(image_folder, x)
)
 
print("Dataset with image paths:\n")
print(df.head())
