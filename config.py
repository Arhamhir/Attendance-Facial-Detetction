# config.py

import os

# Get the absolute path of the directory where this file is located.
# This makes all other paths relative to the project's root folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Directory Paths ---
# This is where your raw dataset of face images is stored.
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')

# This is where we will save the trained model and the label encoder.
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# This is where the final attendance CSV file will be stored.
LOGS_DIR = os.path.join(BASE_DIR, 'attendance_logs')


# --- Image Preprocessing Parameters ---
# The standard size (width, height) to which all face images will be resized.
# This ensures that all feature vectors have the same dimensions.
IMAGE_SIZE = (100, 100)


# --- Data Augmentation Settings ---
# Since the dataset is small, we will generate augmented images to improve model performance.

# The number of new, augmented images to create for each original image.
AUGMENTATIONS_PER_IMAGE = 8

# The range of rotation in degrees to apply. Simulates slight head tilts.
ROTATION_RANGE = (-10, 10)

# The range for adjusting image brightness. Simulates different lighting conditions.
# 1.0 is original brightness. < 1.0 is darker, > 1.0 is brighter.
BRIGHTNESS_RANGE = (0.8, 1.2)


# --- Model & File Naming ---
# The filenames for the saved model and the corresponding label encoder.
ENCODER_FILENAME = 'label_encoder.pkl'

# The name of the CSV file that will log attendance.
ATTENDANCE_LOG_FILE = 'attendance.csv'