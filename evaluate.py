# evaluate_models.py

import os
import pickle
import pandas as pd
import json

from sklearn.metrics import classification_report, accuracy_score

from config import MODELS_DIR, ENCODER_FILENAME

def evaluate_all_models():
    """
    Loads saved models and test data to perform a comparative evaluation.
    """
    print("🚀 Starting model evaluation process...")

    # --- 1. Load the Split Data ---
    split_data_path = os.path.join(MODELS_DIR, 'split_data.pkl')
    try:
        with open(split_data_path, 'rb') as f:
            data = pickle.load(f)
        X_test = data['X_test']
        y_test = data['y_test']
        label_encoder = data['label_encoder']
        print(f"✅ Test data loaded successfully ({len(X_test)} samples).")
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {split_data_path}. Please run the model_trainer.py script first.")
        return

    # --- 2. Find All Saved Models ---
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl') and f not in ['split_data.pkl', ENCODER_FILENAME]]
    if not model_files:
        print("❌ Error: No trained models found in the 'models' directory.")
        return

    results = {}

    # --- 3. Loop Through Models to Evaluate ---
    for model_filename in model_files:
        model_name = model_filename.split('_')[0].upper()
        model_path = os.path.join(MODELS_DIR, model_filename)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        
        results[model_name] = {'accuracy': accuracy, 'report': report}

        print(f"\n{'='*20}\n📊 Results for {model_name}\n{'='*20}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(report)

    # --- 4. Final Comparison and Saving Results ---
    print(f"\n{'='*20}\n🏆 Final Comparison\n{'='*20}")
    
    summary_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [f"{res['accuracy'] * 100:.2f}%" for res in results.values()]
    }).sort_values('Accuracy', ascending=False)
    
    print(summary_df)

    # Save the numerical results to a JSON file for the app
    results_for_json = {model: data['accuracy'] for model, data in results.items()}
    results_path = os.path.join(MODELS_DIR, 'model_comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_for_json, f, indent=4)
    print(f"\n📊 Model comparison results saved to: {results_path}")
    print("\nEvaluation complete!")

if __name__ == "__main__":
    evaluate_all_models()