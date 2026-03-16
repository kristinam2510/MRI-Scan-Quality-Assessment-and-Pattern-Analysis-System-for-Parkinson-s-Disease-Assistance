# MRI Scan Quality Assessment and Pattern Analysis System for Parkinson’s Disease Assistance

## Overview

This project presents an intelligent system designed to analyze **MRI brain scans** and assist in identifying patterns associated with **Parkinson’s Disease**. The system evaluates the quality of MRI scans, performs image preprocessing, and uses **deep learning techniques** to analyze patterns in brain images.

The goal of this project is to support **medical image analysis and research** by applying artificial intelligence to neurological imaging data.

## Features

* MRI scan quality assessment
* Image preprocessing and normalization
* Feature extraction from MRI images
* Pattern analysis using Convolutional Neural Networks (CNN)
* Visualization of results and model performance
* Confusion matrix and classification report for evaluation

## Technologies Used

* **Python**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **TensorFlow / Keras**

## Project Workflow

1. **Data Collection** – MRI brain scan dataset is collected.
2. **Preprocessing** – Images are resized, normalized, and cleaned.
3. **Dataset Splitting** – Data is divided into training and testing sets.
4. **Model Building** – A CNN model is built using TensorFlow/Keras.
5. **Training** – The model is trained on MRI scan data.
6. **Evaluation** – Model performance is evaluated using accuracy, confusion matrix, and classification report.

## Model Architecture

The system uses a **Convolutional Neural Network (CNN)** consisting of:

* Convolutional layers for feature extraction
* MaxPooling layers for dimensionality reduction
* Flatten layer to convert feature maps into vectors
* Dense layers for classification
* Dropout layers to prevent overfitting

## Applications

* Parkinson’s Disease research
* Medical image processing
* AI-based healthcare analysis
* Educational and academic projects

## Results

The model analyzes MRI scans and provides predictions along with performance metrics such as:

* Accuracy
* Confusion Matrix
* Classification Report

## Disclaimer

This project is intended **for research and educational purposes only**. It is not designed to replace professional medical diagnosis or clinical evaluation.

## Author

Developed as part of an academic project on **medical image analysis and machine learning**.
