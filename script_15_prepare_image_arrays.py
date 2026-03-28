import pandas as pd
import numpy as np
import os
from PIL import Image
 
df = pd.read_csv("Labels.csv")
if "Unnamed: 4" in df.columns:
    df = df.drop(columns=["Unnamed: 4"])
 
image_folder = "images_resized"
label_map = {"GON+": 1, "GON-": 0}
 
images = []
labels = []
 
for index, row in df.iterrows():
    image_path = os.path.join(image_folder, row["Image Name"])
    img = Image.open(image_path)
    img_array = np.array(img) / 255.0
    images.append(img_array)
    labels.append(label_map[row["Label"]])
 
images = np.array(images)
labels = np.array(labels)
 
print("Image dataset shape:", images.shape)
print("Label dataset shape:", labels.shape)
