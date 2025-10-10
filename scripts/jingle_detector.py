# Train the audio model which will detect if jingle is present in the audio file
import argparse
import os
import warnings

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")


def extract_features(file_path, sr=22050, n_mfcc=13):
    """Extract MFCC features from an audio file"""
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)  # Average over time frames
    return mfccs


def load_data(jingle_dir, non_jingle_dir):
    """Load audio files and extract features and labels"""
    features = []
    labels = []

    # Load jingle files
    for file_name in tqdm(os.listdir(jingle_dir), desc="Loading jingle files"):
        if file_name.endswith(('.wav', '.opus')):
            file_path = os.path.join(jingle_dir, file_name)
            mfccs = extract_features(file_path)
            features.append(mfccs)
            labels.append(1)  # Jingle present

    # Load non-jingle files
    for file_name in tqdm(os.listdir(non_jingle_dir), desc="Loading non-jingle files"):
        if file_name.endswith(('.wav', '.opus')):
            file_path = os.path.join(non_jingle_dir, file_name)
            mfccs = extract_features(file_path)
            features.append(mfccs)
            labels.append(0)  # Jingle absent

    return np.array(features), np.array(labels)


def train_model(jingle_dir='data/train_data/jingle', non_jingle_dir='data/train_data/non_jingle', model_path='jingle_detector_model.pkl', test_size=0.2, random_state=42, n_estimators=100):
    """Train the jingle detector and save the model to model_path."""
    # Load data
    X, y = load_data(jingle_dir, non_jingle_dir)
    if len(X) == 0:
        raise ValueError("No training data found. Ensure WAV files exist in the specified train_data directories.")
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    # Save the trained model
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    print("Training complete.")


def predict_jingle(file_path, model_path='jingle_detector_model.pkl'):
    """Predict whether a given audio file contains a jingle."""
    clf = joblib.load(model_path)
    mfccs = extract_features(file_path)
    mfccs = mfccs.reshape(1, -1)  # Reshape for prediction
    prediction = clf.predict(mfccs)
    return "Jingle Present" if prediction[0] == 1 else "Jingle Absent"


def main():
    parser = argparse.ArgumentParser(description="Train and run a simple jingle detector using MFCC + RandomForest.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_p = subparsers.add_parser("train", help="Train the model")
    train_p.add_argument("--jingle_dir", default="data/train_data/jingle", help="Directory with jingle WAV files")
    train_p.add_argument("--non_jingle_dir", default="data/train_data/non_jingle", help="Directory with non-jingle WAV files")
    train_p.add_argument("--model_path", default="jingle_detector_model.pkl", help="Where to save the trained model")
    train_p.add_argument("--test_size", type=float, default=0.2, help="Test split size (0-1)")
    train_p.add_argument("--n_estimators", type=int, default=100, help="RandomForest n_estimators")
    train_p.add_argument("--random_state", type=int, default=42, help="Random seed")

    # Predict subcommand
    pred_p = subparsers.add_parser("predict", help="Predict on a file or all files in a folder")
    pred_p.add_argument("--input", required=True, help="Path to an audio file or a directory of audio files")
    pred_p.add_argument("--model_path", default="jingle_detector_model.pkl", help="Path to a trained model file")

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            jingle_dir=args.jingle_dir,
            non_jingle_dir=args.non_jingle_dir,
            model_path=args.model_path,
            test_size=args.test_size,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )
    elif args.command == "predict":
        target = args.input
        if os.path.isdir(target):
            files = [f for f in os.listdir(target) if os.path.isfile(os.path.join(target, f)) and f.lower().endswith(('.wav', '.opus'))]
            if not files:
                print(f"No WAV or OPUS files found in {target}")
                return
            for fname in files:
                fpath = os.path.join(target, fname)
                try:
                    result = predict_jingle(fpath, model_path=args.model_path)
                    print(f"Prediction for {fpath}: {result}")
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")
        else:
            try:
                result = predict_jingle(target, model_path=args.model_path)
                print(f"Prediction for {target}: {result}")
            except Exception as e:
                print(f"Error processing {target}: {e}")


if __name__ == "__main__":
    main()

# The model can be further improved with more data and hyperparameter tuning.
# Ensure you have the required libraries installed:
# pip install numpy librosa scikit-learn soundfile joblib tqdm
# Place your training audio files in 'train_data/jingle' and 'train_data/non_jingle' directories.
# Test files can be placed in 'test_data' directory.
# The audio files should be in WAV or OPUS format for this example.
# You can modify the code to handle other formats by converting them to WAV first.
# Adjust parameters like n_mfcc, model type, and hyperparameters based on your specific use case and data.
# This is a basic implementation and can be expanded with more sophisticated feature extraction and models.
# Always validate the model with a separate validation set for better performance assessment.
# Make sure to handle exceptions and edge cases in production code for robustness.
# Further improvements can include data augmentation, cross-validation, and exploring deep learning models for better accuracy.
# Note: This is a simplified example for educational purposes. Real-world applications may require more complex handling and optimizations.
# Always consider the ethical implications and privacy concerns when working with audio data.
# Happy coding!
