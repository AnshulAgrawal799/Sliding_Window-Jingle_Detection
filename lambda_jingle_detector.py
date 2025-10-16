# lambda_jingle_detector.py
import argparse
import os
import sys
import logging
import time
import warnings

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to load .env using python-dotenv if it's installed. This is opportunistic:
# failure to import dotenv is non-fatal and we simply fall back to existing os.environ.
try:
    from dotenv import find_dotenv, load_dotenv

    _dotenv_path = find_dotenv()  # searches upwards for a .env file
    if _dotenv_path:
        # load into os.environ without overriding existing env vars
        load_dotenv(_dotenv_path, override=False)
except Exception:
    # ignore any errors and continue using os.environ
    pass

from db_mysql import get_connection, get_audio_file_id_by_filename

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

def insert_jingle_prediction(audio_file_path, prediction_result, conn=None):
    """
    Insert a jingle detection prediction into the jingle_detection table and delete the audio file.

    Args:
        audio_file_path (str): Path to the audio file
        prediction_result (str): Either 'Present' or 'Absent'
        conn: Optional database connection (will create one if not provided)

    Returns:
        bool: True if insertion was successful and file was deleted, False otherwise
    """
    own_conn = False
    if conn is None:
        try:
            conn = get_connection()
            own_conn = True
        except Exception as e:
            logging.error(f"Failed to connect to database: {e}")
            return False

    try:
        # Get audio file ID from database
        audio_file_id = get_audio_file_id_by_filename(audio_file_path, conn)
        if audio_file_id is None:
            logging.warning(f"Could not find audio_file_id for {audio_file_path}")
            return False

        # Convert prediction to enum value
        jingle_value = 'Present' if prediction_result == 'Jingle Present' else 'Absent'

        # Insert prediction into database
        cursor = conn.cursor()
        sql = "INSERT INTO jingle_detection (audio_file_id, jingle) VALUES (%s, %s)"
        cursor.execute(sql, (audio_file_id, jingle_value))
        conn.commit()
        cursor.close()

        logging.info(f"Successfully inserted prediction for {audio_file_path}: {jingle_value}")

        # Delete the audio file after successful database insertion
        try:
            os.remove(audio_file_path)
            logging.info(f"Successfully deleted audio file: {audio_file_path}")
        except Exception as e:
            logging.error(f"Failed to delete audio file {audio_file_path}: {e}")
            return False

        return True

    except Exception as e:
        logging.error(f"Error inserting prediction for {audio_file_path}: {e}")
        return False
    finally:
        if own_conn:
            try:
                conn.close()
            except Exception:
                pass

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

def predict_jingle(file_path, model_path='jingle_detector_model.pkl', save_to_db=True):
    """Predict whether a given audio file contains a jingle."""
    clf = joblib.load(model_path)
    mfccs = extract_features(file_path)
    mfccs = mfccs.reshape(1, -1)  # Reshape for prediction
    prediction = clf.predict(mfccs)
    result = "Jingle Present" if prediction[0] == 1 else "Jingle Absent"

    # Save prediction to database if requested
    if save_to_db:
        success = insert_jingle_prediction(file_path, result)
        if success:
            logging.info(f"Successfully processed and archived {file_path}")
        else:
            logging.error(f"Failed to process {file_path} - prediction not saved and file not deleted")

    return result

def lambda_handler(event, context):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Determine mode from event
    mode = event.get('mode', 'predict')

    if mode == 'train':
        # Extract training parameters from event
        jingle_dir = event.get('jingle_dir', 'data/train_data/jingle')
        non_jingle_dir = event.get('non_jingle_dir', 'data/train_data/non_jingle')
        model_path = event.get('model_path', '/tmp/jingle_detector_model.pkl')
        test_size = event.get('test_size', 0.2)
        random_state = event.get('random_state', 42)
        n_estimators = event.get('n_estimators', 100)

        try:
            train_model(
                jingle_dir=jingle_dir,
                non_jingle_dir=non_jingle_dir,
                model_path=model_path,
                test_size=test_size,
                random_state=random_state,
                n_estimators=n_estimators,
            )
            return {
                'statusCode': 200,
                'body': f'Model trained and saved to {model_path}'
            }
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return {
                'statusCode': 500,
                'body': f'Training failed: {str(e)}'
            }

    elif mode == 'predict':
        # Extract prediction parameters from event
        input_path = event.get('input')
        model_path = event.get('model_path', '/tmp/jingle_detector_model.pkl')
        save_to_db = event.get('save_to_db', True)

        if not input_path:
            return {
                'statusCode': 400,
                'body': 'Input path is required for prediction'
            }

        # Use /tmp for temporary files if needed
        if os.path.isdir(input_path):
            files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.lower().endswith(('.wav', '.opus'))]
            if not files:
                return {
                    'statusCode': 404,
                    'body': f'No WAV or OPUS files found in {input_path}'
                }

            logging.info(f"Processing {len(files)} files in {input_path}")
            success_count = 0
            error_count = 0

            for fname in files:
                fpath = os.path.join(input_path, fname)
                try:
                    result = predict_jingle(fpath, model_path=model_path, save_to_db=save_to_db)
                    print(f"Prediction for {fpath}: {result}")
                    success_count += 1
                except Exception as e:
                    logging.error(f"Error processing {fpath}: {e}")
                    print(f"Error processing {fpath}: {e}")
                    error_count += 1

            return {
                'statusCode': 200,
                'body': f'Processing complete. Success: {success_count}, Errors: {error_count}'
            }
        else:
            try:
                result = predict_jingle(input_path, model_path=model_path, save_to_db=save_to_db)
                return {
                    'statusCode': 200,
                    'body': f"Prediction for {input_path}: {result}"
                }
            except Exception as e:
                logging.error(f"Error processing {input_path}: {e}")
                return {
                    'statusCode': 500,
                    'body': f"Error processing {input_path}: {str(e)}"
                }

    else:
        return {
            'statusCode': 400,
            'body': 'Invalid mode. Use "train" or "predict"'
        }
