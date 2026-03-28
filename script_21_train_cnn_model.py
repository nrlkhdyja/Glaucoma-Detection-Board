import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# Load datasets
train_df = pd.read_csv("train_dataset.csv")
test_df = pd.read_csv("test_dataset.csv")
image_folder = "images_resized"

# ---------------------------
# Prepare training data
# ---------------------------
X_train, y_train = [], []

for index, row in train_df.iterrows():
    img_path = os.path.join(image_folder, row["Image Name"])
    img = Image.open(img_path).convert("RGB").resize((224,224))  # ✅ FIXED
    img_array = np.array(img) / 255.0
    X_train.append(img_array)
    y_train.append(row["label_numeric"])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Shuffle data
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# ---------------------------
# Prepare testing data
# ---------------------------
X_test, y_test = [], []

for index, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row["Image Name"])
    img = Image.open(img_path).convert("RGB").resize((224,224))
    img_array = np.array(img) / 255.0
    X_test.append(img_array)
    y_test.append(row["label_numeric"])

X_test = np.array(X_test)
y_test = np.array(y_test)

# ---------------------------
# Split training & validation
# ---------------------------
from sklearn.model_selection import train_test_split

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

# ---------------------------
# Build CNN model
# ---------------------------
model = models.Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------
# Train model
# ---------------------------
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop]
)

# ---------------------------
# Evaluate model
# ---------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# ---------------------------
# Save model
# ---------------------------
model.save("glaucoma_model.h5")
print("Model saved successfully.")