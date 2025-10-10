# Sliding Window Jingle Detection

This project provides a simple MFCC + RandomForest pipeline to train a model that detects whether a short audio clip contains a jingle. The main entrypoint is `scripts/jingle_detector.py`, which supports training and prediction via command-line.

## Project Structure

- `scripts/jingle_detector.py` — Train and predict CLI
- `scripts/generate_dummy_audio.py` — Quickly generate tiny WAV files to test the pipeline
- `data/` — Place data here
  - `data/train_data/jingle/` — WAVs that contain a jingle
  - `data/train_data/non_jingle/` — WAVs that do not contain a jingle
  - `data/test_data/` — WAVs to run predictions on
- `requirements.txt` — Python dependencies

## Prerequisites

- Python 3.9+ recommended
- Windows PowerShell or Command Prompt

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

You can run predictions on a single WAV file or an entire folder containing WAVs.

- Predict for a folder (example uses the dummy test file folder):

  ```powershell
  python scripts\jingle_detector.py predict --input data\test_data --model_path jingle_detector_model.pkl
  ```

- Predict for a single file:

  ```powershell
  python scripts\jingle_detector.py predict --input data\test_data\test_tone.wav --model_path jingle_detector_model.pkl
  ```

Output lines will look like:

```
Prediction for data\test_data\test_tone.wav: Jingle Present
```

## Notes

- Input format: this example pipeline expects WAV files. If you have other formats (e.g., MP3), convert them to WAV or modify the loader accordingly.
- If you have your own data, place it in the `data/train_data/jingle/` and `data/train_data/non_jingle/` directories and retrain the model.
- If the script needs a configuration/model file: the only required artifact is the trained model file produced during training (`jingle_detector_model.pkl` by default). Place it wherever you like and reference it via `--model_path` during prediction.
