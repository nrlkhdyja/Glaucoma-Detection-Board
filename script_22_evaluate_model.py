import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
 
# Load trained model
model = tf.keras.models.load_model("glaucoma_model.h5")
 
# Load test dataset
test_df = pd.read_csv("test_dataset.csv")
image_folder = "images_resized"
 
X_test, y_test = [], []
for index, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row["Image Name"])
    img = Image.open(img_path)
    img_array = np.array(img) / 255.0
    X_test.append(img_array)
    y_test.append(row["label_numeric"])
X_test = np.array(X_test)
y_test = np.array(y_test)
 
# Generate predictions
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)
 
# Confusion matrix
cm = confusion_matrix(y_test, predicted_labels)
print("Confusion Matrix:")
print(cm)
 
# Classification report
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels))
plt.figure(figsize=(6, 5))
sns.heatmap(cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Normal', 'Glaucoma'],
    yticklabels=['Normal', 'Glaucoma'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

test_df["prediction"] = predicted_labels
test_df["probability"] = predictions
test_df.to_csv("test_predictions.csv", index=False)
print("Predictions saved to test_predictions.csv")
