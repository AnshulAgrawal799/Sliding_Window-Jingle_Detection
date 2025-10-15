# Sliding Window Jingle Detection

This project provides a simple MFCC + RandomForest pipeline to train a model that detects whether a short audio clip contains a jingle. The main entrypoint is `scripts/jingle_detector.py`, which supports training and prediction via command-line with **MySQL database integration** for storing predictions.

## Project Structure

- `audio/` — Audio data files after downloading
- `scripts/jingle_detector.py` — Train and predict CLI with database integration
- `db_mysql.py` — Database connection utilities for MySQL integration
- `data/` — Place data here
  - `data/train_data/jingle/` — WAVs that contain a jingle
  - `data/train_data/non_jingle/` — WAVs that do not contain a jingle
  - `data/test_data/` — WAVs to run predictions on
- `requirements.txt` — Python dependencies
- `.env` — Environment variables for database configuration

## Prerequisites

- Python 3.9+ recommended
- Windows PowerShell or Command Prompt
- MySQL server running and accessible
- Audio files must be registered in the `audio_file` table before running predictions

## Database Setup

1. **Create the database table** (run this SQL in your MySQL database):

```sql
CREATE TABLE jingle_detection (
  id BIGINT UNSIGNED NOT NULL PRIMARY KEY AUTO_INCREMENT,
  audio_file_id BIGINT NOT NULL,
  jingle ENUM('Absent','Present') NOT NULL DEFAULT 'Absent',
  row_created_at BIGINT NOT NULL DEFAULT (UNIX_TIMESTAMP())
);
```

2. **Create a `.env` file** in the project root with your database credentials:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_username
DB_PASS=your_password
DB_NAME=your_database_name
```

3. **Ensure audio files are in the database** - Before running predictions, make sure your audio files are registered in the `audio_file` table with their file paths in the `s3_uri` column.

## Setup

1. Create and activate a virtual environment

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies

   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## (Optional) Generate Dummy Data For Quick Testing

You can generate a couple of small WAV files to quickly test training and inference:

```powershell
python scripts\generate_dummy_audio.py
```

This will create:
- `data/train_data/jingle/jingle_tone.wav`
- `data/train_data/non_jingle/noise.wav`
- `data/test_data/test_tone.wav`

These are tiny synthetic files intended only to verify the end-to-end flow.

## Train the Model

By default, `scripts/jingle_detector.py` expects training data under `data/train_data/`.

```powershell
python scripts\jingle_detector.py train \
  --jingle_dir data/train_data/jingle \
  --non_jingle_dir data/train_data/non_jingle \
  --model_path jingle_detector_model.pkl \
  --test_size 0.2 \
  --n_estimators 100 \
  --random_state 42
```

- The trained model will be saved to `jingle_detector_model.pkl` by default.
- You can change `--model_path` to any desired location.

## Run Predictions

You can run predictions on a single WAV file or an entire folder containing WAVs. **Predictions are automatically saved to the MySQL database.**

- Predict for a folder (example uses the dummy test file folder):

  ```powershell
  python scripts\jingle_detector.py predict --input data\test_data --model_path jingle_detector_model.pkl
  ```

- Predict for a single file:

  ```powershell
  python scripts\jingle_detector.py predict --input data\test_data\test_tone.wav --model_path jingle_detector_model.pkl
  ```

- Predict without saving to database:

  ```powershell
  python scripts\jingle_detector.py predict --input data\test_data --model_path jingle_detector_model.pkl --no_db
  ```

Output lines will look like:

```
Prediction for data\test_data\test_tone.wav: Jingle Present
```

## Database Integration

When running predictions, the script will:

1. **Look up audio file ID** - Find the corresponding `audio_file_id` in the `audio_file` table using the filename
2. **Store prediction results** - Insert a row into the `jingle_detection` table with:
   - `audio_file_id`: The ID from the `audio_file` table
   - `jingle`: 'Present' or 'Absent' based on the prediction
   - `row_created_at`: Automatic timestamp
3. **Delete audio file** - Remove the processed audio file from the filesystem after successful database insertion

**⚠️ Important**: Audio files are permanently deleted after successful processing. Ensure you have backups if you need to retain the original files.

### Database Error Handling

- If database connection fails, the script logs a warning but continues processing
- If an audio file ID cannot be found, the prediction is logged but not saved to database
- If database insertion fails, the audio file is not deleted (to allow for retry)
- If file deletion fails after successful database insertion, the operation is considered failed
- All database operations are handled gracefully without stopping the prediction process

## Notes

- Input format: this example pipeline expects WAV files. If you have other formats (e.g., MP3), convert them to WAV or modify the loader accordingly.
- If you have your own data, place it in the `data/train_data/jingle/` and `data/train_data/non_jingle/` directories and retrain the model.
- If the script needs a configuration/model file: the only required artifact is the trained model file produced during training (`jingle_detector_model.pkl` by default). Place it wherever you like and reference it via `--model_path` during prediction.
- Database credentials are read from environment variables defined in `.env` file
- Make sure your `.env` file is properly configured before running predictions that save to database
