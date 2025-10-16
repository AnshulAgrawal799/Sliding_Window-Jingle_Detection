@echo off
REM package_lambda.bat - Script to create Lambda deployment packages

REM Create virtual environment
python -m venv lambda_env

REM Activate environment
call lambda_env\Scripts\activate.bat

REM Install dependencies
pip install -r requirements.txt

REM Create directories for packages
mkdir lambda_packages

REM Package for download_pending
mkdir lambda_packages\lambda_download_pending
copy lambda_download_pending.py lambda_packages\lambda_download_pending\
copy db_mysql.py lambda_packages\lambda_download_pending\  REM Assuming db_mysql.py exists
cd lambda_packages\lambda_download_pending
powershell -command "Compress-Archive -Path . -DestinationPath ../lambda_download_pending.zip"
cd ..\..

REM Package for jingle_detector
mkdir lambda_packages\lambda_jingle_detector
copy lambda_jingle_detector.py lambda_packages\lambda_jingle_detector\
copy db_mysql.py lambda_packages\lambda_jingle_detector\  REM Assuming db_mysql.py exists
cd lambda_packages\lambda_jingle_detector
powershell -command "Compress-Archive -Path . -DestinationPath ../lambda_jingle_detector.zip"
cd ..\..

REM Package for db_mysql
mkdir lambda_packages\lambda_db_mysql
copy lambda_db_mysql.py lambda_packages\lambda_db_mysql\
copy db_mysql.py lambda_packages\lambda_db_mysql\  REM Assuming db_mysql.py exists
cd lambda_packages\lambda_db_mysql
powershell -command "Compress-Archive -Path . -DestinationPath ../lambda_db_mysql.zip"
cd ..\..

REM Deactivate environment
call lambda_env\Scripts\deactivate.bat

echo Lambda packages created: lambda_packages/lambda_download_pending.zip, lambda_packages/lambda_jingle_detector.zip, and lambda_packages/lambda_db_mysql.zip
