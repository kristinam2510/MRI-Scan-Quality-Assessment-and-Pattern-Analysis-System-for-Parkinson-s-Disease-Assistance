import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("parkinson_mri_classification_model.h5")

IMAGE_SIZE = 128
CLASS_NAMES = ["Normal", "Parkinson"]


def calculate_image_quality_parameters(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpness_value = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    brightness_value = np.mean(gray_image)
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


def predict_mri_image(image_path):
    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found")
        return None

    # Quality assessment
    sharpness, brightness, contrast = calculate_image_quality_parameters(image)
    quality_label = determine_image_quality(sharpness, brightness, contrast)

    # Resize image
    resized_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # Normalize image
    normalized_image = resized_image / 255.0

    # Expand dimensions
    input_image = np.expand_dims(normalized_image, axis=0)

    # Predict
    prediction = model.predict(input_image, verbose=0)

    predicted_class_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    # Print result
    print("Prediction:", predicted_class)
    print(f"Confidence: {confidence:.2f}%")
    print("Quality:", quality_label)
    print(f"Sharpness: {sharpness:.2f}")
    print(f"Brightness: {brightness:.2f}")
    print(f"Contrast: {contrast:.2f}")

    # Show image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2f}%\n"
        f"Quality: {quality_label}"
    )
    plt.axis("off")
    plt.show()

    # Return result
    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "quality": quality_label,
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2)
    }


# Example
result = predict_mri_image("T2W_TSE_016.png")
print(result)