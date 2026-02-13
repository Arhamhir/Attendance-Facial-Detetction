# face_operations.py
import cv2
import numpy as np
import os
from config import IMAGE_SIZE

# --- Load the pre-trained Haar Cascade model for face detection ---
# This classifier is an XML file that contains the trained model data.
# Ensure 'haarcascade_frontalface_default.xml' is in the same directory.
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    print("Error loading cascade file. Make sure 'haarcascade_frontalface_default.xml' is available.")

def detect_and_crop_face(image: np.ndarray) -> np.ndarray | None:
    """
    Detects the largest face in an image and returns the cropped face.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray | None: The cropped face image if a face is found, otherwise None.
    """
    # Convert the image to grayscale, as the face detector works on grayscale images.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using the Haar Cascade classifier.
    # scaleFactor=1.1 and minNeighbors=5 are good defaults for robust detection.
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None  # No faces were found

    # Assume the largest face (by area) is the target.
    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Crop the original color image to get only the face.
    cropped_face = image[y:y + h, x:x + w]
    
    return cropped_face

def preprocess_face(face_image: np.ndarray) -> np.ndarray:
    """
    Preprocesses a cropped face image for model training/inference.

    Args:
        face_image (np.ndarray): The cropped face image.

    Returns:
        np.ndarray: The preprocessed face image.
    """
    # 1. Convert to grayscale for feature extraction.
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize to a standard size to ensure consistent feature vector length.
    resized_face = cv2.resize(gray_face, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # 3. Apply Histogram Equalization to improve contrast.
    # This is a key step for high accuracy, as it normalizes brightness and
    # enhances facial features, making the model more robust to lighting changes.
    equalized_face = cv2.equalizeHist(resized_face)
    
    return equalized_face

def extract_features(processed_face: np.ndarray) -> np.ndarray:
    """
    Converts a preprocessed face image into a 1D feature vector.

    Args:
        processed_face (np.ndarray): The preprocessed face image.

    Returns:
        np.ndarray: A 1D NumPy array representing the face features.
    """
    # 1. Flatten the 2D image array into a 1D vector.
    flattened_face = processed_face.flatten()
    
    # 2. Normalize the pixel values to a range of [0, 1].
    # Normalization helps the ML model (especially SVM) converge faster and perform better.
    normalized_features = flattened_face / 255.0
    
    return normalized_features