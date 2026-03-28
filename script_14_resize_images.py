import pandas as pd
import os
from PIL import Image
 
# Load dataset
df = pd.read_csv("Labels.csv")
if "Unnamed: 4" in df.columns:
    df = df.drop(columns=["Unnamed: 4"])
 
image_folder = "images"
resized_folder = "images_resized"

if not os.path.exists(resized_folder):
    os.makedirs(resized_folder)
 
# Resize images
for index, row in df.iterrows():
    image_path = os.path.join(image_folder, row["Image Name"])
    img = Image.open(image_path)
    img = img.resize((224, 224))
    save_path = os.path.join(resized_folder, row["Image Name"])
    img.save(save_path)
 
print("All images resized successfully.")
