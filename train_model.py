# model_trainer.py (Version 3 - Training Only)

import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from config import (
    DATASET_PATH, 
    MODELS_DIR, 
    ENCODER_FILENAME,
    AUGMENTATIONS_PER_IMAGE,
    ROTATION_RANGE,
    BRIGHTNESS_RANGE
)
# Ensure you have the correct filename for face operations
import face_configure as fo

# --- (The augment_image function remains the same as before) ---
def augment_image(image: np.ndarray) -> list[np.ndarray]:
    """Applies random rotation and brightness adjustments to an image."""
    augmented_images = []
    h, w = int(image.shape[0]), int(image.shape[1])
    center = (w // 2, h // 2)
    for _ in range(AUGMENTATIONS_PER_IMAGE):
        angle = float(np.random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1]))
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rot_mat, (w, h))
        brightness_factor = np.random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
        hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        brightness_change = int((brightness_factor - 1.0) * 255)
        v_channel = np.add(v_channel.astype(np.int16), brightness_change)
        v_channel = np.clip(v_channel, 0, 255).astype(np.uint8)
        final_hsv = cv2.merge((h_channel, s_channel, v_channel))
        brightened = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        augmented_images.append(brightened)
    return augmented_images

def train_all_models():
    """
    Loads data, trains multiple models, and saves the models and split data.
    """
    print("🚀 Starting model training process...")

    # --- 1. Load, Augment, and Process Data ---
    features, labels = [], []
    os.makedirs(MODELS_DIR, exist_ok=True)
    person_names = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    for name in tqdm(person_names, desc="Processing Persons"):
        person_dir = os.path.join(DATASET_PATH, name)
        for filename in os.listdir(person_dir):
            image = cv2.imread(os.path.join(person_dir, filename))
            if image is None: continue
            cropped_face = fo.detect_and_crop_face(image)
            if cropped_face is not None:
                processed_face = fo.preprocess_face(cropped_face)
                features.append(fo.extract_features(processed_face))
                labels.append(name)
                for aug_face in augment_image(cropped_face):
                    processed_aug_face = fo.preprocess_face(aug_face)
                    features.append(fo.extract_features(processed_aug_face))
                    labels.append(name)

    if not features:
        print("Error: No faces found in the dataset. Make sure haarcascade file is present.")
        return
    print(f"\n✅ Data loading and augmentation complete. Total samples: {len(features)}")

    # --- 2. Label Encoding and Stratified Data Splitting ---
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    X, y = np.array(features), np.array(encoded_labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # --- 3. Save the Split Data for the Evaluator ---
    # THIS IS THE MISSING PART
    split_data_path = os.path.join(MODELS_DIR, 'split_data.pkl')
    with open(split_data_path, 'wb') as f:
        pickle.dump({'X_test': X_test, 'y_test': y_test, 'label_encoder': label_encoder}, f)
    print(f"📊 Test data and encoder saved to: {split_data_path}")

    # --- 4. Define Models and Hyperparameter Grids ---
    models = {
        'SVM': (SVC(probability=True), {
            'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']
        }),
        'KNN': (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']
        }),
        'ANN': (MLPClassifier(max_iter=1000, early_stopping=True), {
            'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]
        })
    }
    
    # --- 5. Loop Through Models to Train and Save ---
    print("\n💾 Training and saving all models...")
    for model_name, (model, params) in models.items():
        print(f"\n---\nTuning and training {model_name}...")
        grid = GridSearchCV(model, params, refit=True, verbose=1, cv=3)
        grid.fit(X_train, y_train)
        
        # Save the trained model
        model_filename = f"{model_name.lower()}_model.pkl"
        model_path = os.path.join(MODELS_DIR, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(grid.best_estimator_, f)
        print(f"✅ Model '{model_name}' saved to: {model_path}")

    print("\nTraining complete!")

if __name__ == "__main__":
    train_all_models()