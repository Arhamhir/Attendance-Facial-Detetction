# Face Detection and Attendance (Python)

A compact, learning-focused face recognition pipeline that covers data preparation, classical ML model training, evaluation, and attendance logging. The project is intentionally small but complete enough to teach end-to-end workflow design.

## Highlights
- End-to-end pipeline: face detection, preprocessing, feature extraction, model training, evaluation, and logging.
- Multiple model comparison (SVM, KNN, ANN) with simple hyperparameter search.
- Data augmentation for small datasets (rotation and brightness).
- Practical CSV-based attendance logging with duplicate-per-day protection.

## Why this is great to learn from
- Clear separation of concerns: configuration, face ops, training, evaluation, and logging.
- Teaches classical ML workflows without heavy deep learning dependencies.
- Emphasizes reproducibility (saved models, split data, and evaluation summary).
- Highlights real-world constraints: small data, lighting variance, and idempotent logging.

## Project Structure
- app.py: model training script (same logic as train_model.py).
- train_model.py: trains SVM, KNN, and ANN models with grid search.
- evaluate.py: evaluates saved models and writes comparison results.
- face_configure.py: face detection, preprocessing, and feature extraction.
- attendance_manager.py: attendance CSV logging and retrieval.
- config.py: central configuration for paths and preprocessing.
- models/: saved models, label encoder, and evaluation output.
- attendance_logs/: attendance CSV file output.

## Requirements
- Python 3.10+
- Packages: opencv-python, numpy, pandas, scikit-learn, tqdm

Install dependencies:

```bash
pip install opencv-python numpy pandas scikit-learn tqdm
```

## Dataset Layout
Place images in the dataset folder with one subfolder per person:

```
dataset/
  PersonA/
    img1.jpg
    img2.jpg
  PersonB/
    img1.jpg
```

## Configure
Edit config values in config.py if needed:
- IMAGE_SIZE
- AUGMENTATIONS_PER_IMAGE
- ROTATION_RANGE
- BRIGHTNESS_RANGE
- Paths for dataset, models, and logs

## Train Models
Run either training script:

```bash
python train_model.py
```

or

```bash
python app.py
```

Outputs:
- models/svm_model.pkl
- models/knn_model.pkl
- models/ann_model.pkl
- models/split_data.pkl
- models/label_encoder.pkl

## Evaluate Models
After training:

```bash
python evaluate.py
```

Outputs:
- Console report for each model
- models/model_comparison_results.json

## Attendance Logging
Use attendance_manager.py from your own inference loop to log attendance:
- log_attendance(name): adds a single daily entry per person
- get_attendance_df(): reads and sorts logs

The log file is written to:
- attendance_logs/attendance.csv

## Notes
- Face detection uses OpenCV Haar cascades.
- If no faces are found, training exits early with a warning.
- This project focuses on learning and experimentation rather than production scale.

## Learning Extensions (Optional)
- Add a real-time webcam inference loop.
- Compare classical features with embeddings from a deep model.
- Export metrics to a dashboard or notebook.
- Add unit tests for preprocessing and logging.
