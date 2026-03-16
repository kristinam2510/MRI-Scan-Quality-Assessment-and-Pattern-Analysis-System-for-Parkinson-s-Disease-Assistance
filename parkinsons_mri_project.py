# ==========================================================
# MRI Scan Quality Assessment and Pattern Analysis System
# for Parkinson's Disease
# ==========================================================

# Import Required Libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ==========================================================
# PARAMETERS
# ==========================================================

DATASET_DIRECTORY = "dataset"
IMAGE_SIZE = 128
CLASS_NAMES = ["Normal", "Parkinson"]


# ==========================================================
# IMAGE QUALITY ASSESSMENT FUNCTIONS
# ==========================================================

def calculate_image_quality_parameters(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sharpness measurement using Laplacian variance
    sharpness_value = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    # Brightness measurement
    brightness_value = np.mean(gray_image)

    # Contrast measurement
    contrast_value = np.std(gray_image)

    return sharpness_value, brightness_value, contrast_value


def determine_image_quality(sharpness, brightness, contrast):

    if sharpness < 50:
        return "Blurry Image"

    if brightness < 40:
        return "Low Brightness"

    if brightness > 220:
        return "Overexposed Image"

    if contrast < 20:
        return "Low Contrast"

    return "Good Quality Image"


# ==========================================================
# LOAD MRI DATASET
# ==========================================================

image_data = []
image_labels = []
image_quality_information = []

for class_index, class_name in enumerate(CLASS_NAMES):

    class_folder_path = os.path.join(DATASET_DIRECTORY, class_name)

    for file_name in os.listdir(class_folder_path):

        image_path = os.path.join(class_folder_path, file_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Quality Assessment
        sharpness, brightness, contrast = calculate_image_quality_parameters(image)

        quality_label = determine_image_quality(sharpness, brightness, contrast)

        image_quality_information.append({
            "file_name": file_name,
            "class_name": class_name,
            "sharpness": sharpness,
            "brightness": brightness,
            "contrast": contrast,
            "quality": quality_label
        })

        # Resize image
        resized_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # Normalize image
        normalized_image = resized_image / 255.0

        image_data.append(normalized_image)
        image_labels.append(class_index)


image_data = np.array(image_data)
image_labels = np.array(image_labels)

print("Dataset Loaded Successfully")
print("Total Images:", len(image_data))


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

train_images, test_images, train_labels, test_labels = train_test_split(
    image_data,
    image_labels,
    test_size=0.2,
    random_state=42,
    stratify=image_labels
)

train_labels = to_categorical(train_labels, 2)
test_labels = to_categorical(test_labels, 2)

print("Training Samples:", len(train_images))
print("Testing Samples:", len(test_images))


# ==========================================================
# CONVOLUTIONAL NEURAL NETWORK MODEL
# ==========================================================

cnn_model = Sequential()

cnn_model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Conv2D(64, (3,3), activation='relu'))
cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Conv2D(128, (3,3), activation='relu'))
cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Flatten())

cnn_model.add(Dense(128, activation='relu'))

cnn_model.add(Dropout(0.5))

cnn_model.add(Dense(2, activation='softmax'))

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()


# ==========================================================
# MODEL TRAINING
# ==========================================================

training_history = cnn_model.fit(
    train_images,
    train_labels,
    epochs=15,
    batch_size=16,
    validation_split=0.1
)


# ==========================================================
# MODEL EVALUATION
# ==========================================================

test_loss, test_accuracy = cnn_model.evaluate(test_images, test_labels)

print("Test Accuracy:", test_accuracy*100)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

predicted_probabilities = cnn_model.predict(test_images)

predicted_classes = np.argmax(predicted_probabilities, axis=1)

true_classes = np.argmax(test_labels, axis=1)

confusion_matrix_result = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(6,5))

sns.heatmap(
    confusion_matrix_result,
    annot=True,
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.show()


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES))


# ==========================================================
# TRAINING HISTORY VISUALIZATION
# ==========================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(training_history.history['accuracy'])
plt.plot(training_history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])

plt.subplot(1,2,2)
plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train","Validation"])

plt.show()


# ==========================================================
# SAVE TRAINED MODEL
# ==========================================================

cnn_model.save("parkinson_mri_classification_model.h5")

print("Model Saved Successfully")


# ==========================================================
# MRI IMAGE PREDICTION FUNCTION
# ==========================================================

def predict_mri_image(image_path):

    image = cv2.imread(image_path)

    sharpness, brightness, contrast = calculate_image_quality_parameters(image)

    quality_label = determine_image_quality(sharpness, brightness, contrast)

    resized_image = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE))

    normalized_image = resized_image / 255.0

    input_image = np.expand_dims(normalized_image,axis=0)

    prediction = cnn_model.predict(input_image)

    predicted_class_index = np.argmax(prediction)

    predicted_class_name = CLASS_NAMES[predicted_class_index]

    confidence_score = prediction[0][predicted_class_index] * 100

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.title(
        "Prediction: " + predicted_class_name +
        "\nConfidence: " + str(round(confidence_score,2)) + "%" +
        "\nQuality: " + quality_label
    )

    plt.axis("off")

    plt.show()


# Example Prediction
# predict_mri_image("dataset/Normal/sample_image.jpg")