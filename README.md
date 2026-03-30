# Glaucoma Detection Dashboard

## Project Overview
This project is a deep learning-based web application for detecting glaucoma from retinal fundus images using a Convolutional Neural Network (CNN).

The system allows users to upload an eye image and receive:
- Prediction (Glaucoma / Normal)
- Confidence score
- Grad-CAM visualization for model explainability

## Features
- Upload retinal fundus images
- Automated glaucoma detection
- Visual explanation using Grad-CAM
- User-friendly interface with Streamlit

## Model Information
- Model type: CNN (TensorFlow / Keras)
- Input size: 224 × 224 images
- Output: Binary classification (Glaucoma / Normal)

## Dataset
This project uses the **Hillel Yaffe Glaucoma Dataset (HYGD)** from PhysioNet:

https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/

> Note: The full dataset (train, validation, and test sets) is not uploaded due to GitHub size limitations.

## Project Structure
Glaucoma-Detection-Board/
│
├── app.py
├── glaucoma_model.h5
├── requirements.txt
├── README.md
│
├── scripts/
├── sample_images/

## 📖 References
Abramovich, O., et al. (2026). *Hillel Yaffe Glaucoma Dataset (HYGD) (Version 1.1.0).* PhysioNet.  
https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/

Akada, M., et al. (2026). *Myopic and glaucomatous optic neuropathy in highly myopic eyes: Current concepts and clinical implications.* Journal of Clinical Medicine, 15(7), 2491.  
https://doi.org/10.3390/jcm15072491
