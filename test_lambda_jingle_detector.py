# test_lambda_jingle.py
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Lambda handler
from lambda_jingle_detector import lambda_handler

# Mock AWS Lambda context (not used in the handler, but required for signature)
class MockContext:
    pass

# Define a sample event for testing
# Example 1: Predict on a single file
event = {
    'mode': 'predict',
    'input': 'audio/',  # Directory with audio files
    'model_path': 'jingle_detector_model.pkl',  # Or your local model path
    'save_to_db': True
}

# Example 2: Predict on a directory
# event = {
#     'mode': 'predict',
#     'input': 'audio/',  # Directory with audio files
#     'model_path': 'jingle_detector_model.pkl',
#     'save_to_db': True
# }

# Example 3: Train the model
# event = {
#     'mode': 'train',
#     'jingle_dir': 'data/train_data/jingle',
#     'non_jingle_dir': 'data/train_data/non_jingle',
#     'model_path': 'jingle_detector_model.pkl'
# }

# Run the handler
context = MockContext()
response = lambda_handler(event, context)

# Print the response
print(response)