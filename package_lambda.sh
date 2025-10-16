#!/bin/bash
# package_lambda.sh - Script to create Lambda deployment packages

# Install dependencies in a virtual environment
python -m venv lambda_env
source lambda_env/bin/activate

pip install -r requirements.txt

# Create directories for packages
mkdir -p lambda_packages

# Package for download_pending
cd lambda_packages
mkdir lambda_download_pending
cp ../lambda_download_pending.py lambda_download_pending/
cp ../db_mysql.py lambda_download_pending/  # Assuming db_mysql.py exists
cd lambda_download_pending
zip -r ../lambda_download_pending.zip .
cd ..

# Package for jingle_detector
mkdir lambda_jingle_detector
cp ../lambda_jingle_detector.py lambda_jingle_detector/
cp ../db_mysql.py lambda_jingle_detector/  # Assuming db_mysql.py exists
cd lambda_jingle_detector
zip -r ../lambda_jingle_detector.zip .
cd ../..

# Deactivate environment
deactivate

echo "Lambda packages created: lambda_packages/lambda_download_pending.zip and lambda_packages/lambda_jingle_detector.zip"
