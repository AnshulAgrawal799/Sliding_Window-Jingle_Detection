# test_lambda_download.py
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Lambda handler
from lambda_download_pending import lambda_handler

# Mock AWS Lambda context (not used in the handler, but required for signature)
class MockContext:
    pass

# Define a sample event for testing
# Example: Set limit for fetching pending files
event = {
    'limit': 5  # Number of pending files to process (default: 5)
}

# Run the handler
context = MockContext()
response = lambda_handler(event, context)

# Print the response
print(response)