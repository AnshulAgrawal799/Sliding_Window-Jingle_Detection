# Sliding-Window Jingle Detection — Project Plan

**Objective**

Design and implement a robust sliding-window detection system that locates repeated jingle plays inside long vehicle recordings and maps each detection to an absolute timestamp. The system must work reliably on phone recordings captured in noisy Indian village environments and produce a verifiable CSV of detection.

---

## Changes in this version

This version incorporates the **actual jingle audio file** you supplied and adds concrete steps that use it for: canonical reference creation, synthetic-data generation (mixing the jingle into noisy backgrounds at controlled SNRs), fast prefiltering (matched-filter / cross-correlation / fingerprinting), and as an optional synchronization anchor. New deliverables and scripts were added: `data/reference/jingle.wav`, `scripts/make_synthetic.py`, and `scripts/prefilter.py`.

---

## 1. Scope

## This plan covers everything from use of the reference jingle audio for: dataset expansion, prefiltering, and verification.

## 2. Deliverables

1. `plan.md` (this document)
2. Training scripts and model artifact (TensorFlow/PyTorch) that detect jingle in short windows
3. Inference script that scans long recordings, outputs `detections.csv`, and exports short audio clips per detection
4. Evaluation report (precision/recall/confusion matrix) on a labeled holdout set
5. README with folder structure, dependency list, and usage examples
6. **Reference jingle** stored at `data/reference/jingle.wav` and utilities to use it:

   - `scripts/make_synthetic.py` — generates synthetic long recordings by mixing jingle into backgrounds at controlled SNRs and timestamps
   - `scripts/prefilter.py` — computes fast cross-correlation / fingerprint prefilter to produce candidate windows

---

## 3. Data & Folder Layout (recommended)

```
project-root/
  data/
    reference/
      jingle.wav           # canonical reference audio of the exact jingle
    train/                 # labeled short chunks for training
      jingle/
      no_jingle/
    val/                   # validation set (optional)
    holdout/               # test set for final evaluation
  recordings/              # long route recordings (wav/mp3)
  models/
  outputs/
    clips/
    detections/
  notebooks/
  scripts/
    train.py
    infer_sliding.py
    export_clips.py
    make_synthetic.py
    prefilter.py
  README.md
  requirements.txt
```

Notes:

- Keep the original raw `jingle.wav` immutable; create a normalized canonical version for algorithmic use.
- Create a small `reference/` README describing the jingle: nominal duration, sample rate, any known variations (e.g., alternate mixes).

---

## 4. Preprocessing pipeline (updated)

1. **Canonical reference creation**: load `data/reference/jingle.wav`, convert to mono, resample to the project SR (e.g., 22050 Hz), normalize peak amplitude, trim leading/trailing silence and save `data/reference/jingle_canonical.wav`.
2. **Fingerprint & template assets**: compute and store a few artifacts from the canonical jingle:

   - log–mel spectrogram (same parameters as training)
   - short-time Fourier magnitude template (for matched-filter / FFT cross-correlation)
   - an audio fingerprint (e.g., chroma/chromaprint or a learned embedding) to use in fast candidate detection and de-duplication

3. **Resampling & channels**: convert recordings to mono and resample to a stable sample rate (e.g., 22050 Hz).
4. **Silence trimming & padding**: for fixed-window training, pad short files to window length with zeros; trim excessive leading silence in chunks if needed.
5. **Normalization**: peak-normalize each file (divide by max(abs)).
6. **Feature extraction**: compute log–mel spectrograms: `n_mels = 64`, `n_fft = 2048`, `hop_length = 512`. Convert to dB and standardize (mean/std) per-sample or per-batch.
7. **Augmentation (expanded)**:

   - Use the canonical jingle to create synthetic positive examples: mix jingle into long background recordings at a range of SNRs (e.g., +10 dB down to -10 dB) and at random start times.
   - Add recorded or synthetic background noises (crowd, engine, other music) at random SNRs.
   - Time-stretch (±5–10%) and pitch shift (±1–2 semitones) the jingle to mimic playback speed/pitch changes in cheap phones or variable playback.
   - Encode some training files as low-bitrate MP3/AMR to simulate phone compression.
   - Random gain, small clipping, and lowpass/highpass filtering to mimic phone artifacts.

Rationale: using the actual jingle to synthesize many controlled examples dramatically improves the model's ability to generalize and lets you measure performance vs. SNR.

---

## 5. Model & Sliding-Window Design (updated)

### 5.1. Two-stage detection (recommended)

To balance precision, recall, and runtime, use a **two-stage pipeline**:

**Stage A — Fast prefilter (candidate generation)**

- Use the canonical jingle to compute a matched-filter / normalized cross-correlation (time-domain or FFT-accelerated) against long recordings, or compute fingerprints and perform fast fingerprint matching.
- The prefilter emits candidate timestamps (peaks) where correlation/fingerprint similarity is high. Keep a low threshold to prioritize recall.
- Prefiltering reduces the number of windows passed to the heavier classifier by >90% in practice.

**Stage B — Classifier verification**

- For each candidate, extract a window centered on the candidate (window length ≈ jingle length or slightly larger) and run the CNN classifier to produce final probability and refined timing.
- Optionally run the classifier on small offsets around the candidate and pick the best-scoring center to improve timing.

Benefits: much faster than brute-force sliding-window over every possible stride, lower false positives when classifier confirms candidates.

### 5.2. Window & stride (updated)

- **Window length**: set to jingle length + small margin (e.g. jingle_length + 0.5s) so the window can contain the whole jingle even with time-stretch changes.
- **Stride**: if running full brute-force sliding, use small stride (0.2–0.5s). In two-stage design, the prefilter determines candidate locations and you can use denser local sampling only around candidates.

### 5.3. Input representation

- Log–mel spectrogram of the window. Store the jingle’s spectrogram as a template for direct spectrogram cross-correlation experiments.

### 5.4. Model architecture (suggested)

- Lightweight CNN (Conv2D -> Pool -> Conv2D -> Pool -> GlobalAvgPool -> Dense -> Sigmoid) tuned to the jingle’s spectral footprint.
- Optionally train a small embedding network and use cosine-similarity scoring between window embeddings and jingle embedding as an additional signal.

### 5.5. Fingerprinting fallback

- If classifier confidence is low or latency is critical, provide an optional fingerprint-based fallback (e.g., chromaprint / acoustic fingerprint) to provide high-precision detections or to cluster duplicates.

---

## 6. Training workflow (updated)

1. Create a training set that mixes real positive examples (manually-labeled `wow*` chunks) and synthetic positives generated by inserting `jingle_canonical.wav` into background audio at various SNRs and variants.
2. Assemble balanced minibatches with on-the-fly augmentation (including variable SNRs and time-stretch/pitch-shift of the jingle).
3. Use `Adam` optimizer, initial LR (e.g. 1e-3) with ReduceLROnPlateau.
4. Early stopping on validation loss with `restore_best_weights=True`.
5. Save best model to `models/jingle_detector.h5` (or `pt` for PyTorch).
6. Keep a training log and final model metadata (training parameters, dataset counts, augmentation settings), and export the canonical jingle embedding for inference.

Curriculum suggestion: start training with high-SNR synthetic positives (easy), progressively add lower-SNR and real noisy positives.

---

## 7. Inference pipeline (sliding-window) — (updated with jingle-aware steps)

1. **Load long recording** (mono, sr=22050).
2. **Prefilter stage (new)**:

   - Compute normalized cross-correlation (time or spectrogram domain) between the recording and `jingle_canonical` (FFT-based to be fast) or compute fingerprints and scan for matches.
   - Detect peaks above a permissive threshold and emit candidate midpoints. Optionally apply median filtering to the correlation curve to remove spurious narrow peaks.

3. **Candidate window generation**: for each candidate, extract a window of `window_s` centered on candidate (pad if needed). Optionally extract ±0.5s neighboring windows for refinement.
4. **Batch predict**: convert each window to log-mel, batch and predict probabilities using the classifier.
5. **Threshold & peak selection**:

   - For classifier outputs, apply detection threshold (tunable; e.g., 0.5 initial).
   - Merge adjacent positive candidate windows if they overlap into a single detection span. Use NMS and a min-gap (fraction of jingle length).
   - Record detection center time (midpoint of merged span) and confidence (max prob in span). Also record whether the detection was prefiltered and/or classifier-confirmed.

6. **Output**: `detections.csv` with columns: `recording_file, detection_idx, start_s, end_s, mid_s, confidence, absolute_time, method` (method = "prefilter+classifier" or "classifier-only" or "fingerprint").
7. **Export clips**: for each detection, export a short audio clip (e.g., ±1s around midpoint) to `outputs/clips/` and include the canonical jingle spectrogram/metadata for quick human review.

Performance optimization: prefiltering usually reduces total inference time by an order of magnitude while keeping recall high.

---

## 8. Timestamp mapping & synchronization (updated)

- **Best-effort mapping**: Add recording file creation time (filesystem ctime) to compute `absolute_time = file_ctime + mid_s`.
- **Jingle-as-anchor**: If the system operator can guarantee the jingle is _intentionally_ played at known times (for example: a calibration jingle at the start of the route), use the jingle detection to bootstrap or validate file timestamps. If not, treat it as a content event only.
- **Recommended for reliability**: request a short spoken sync phrase or audible beep at the start of each recording; detect that explicitly to get reliable start time mapping. If that is impossible, require uploader to include `YYYYMMDD_HHMMSS` in the filename.

Note: do not rely on detection of arbitrary jingle plays for absolute synchronization unless the recording process is controlled.

---

## 9. Evaluation & validation (updated)

- **Holdout set**: keep a labeled holdout set of long recordings and/or chunks not used in training.
- **Synthetic SNR sweep**: use `make_synthetic.py` to create test recordings where the canonical jingle is inserted at known times across a range of SNRs (e.g., +10, +5, 0, -5, -10 dB). Report detection rate vs SNR.
- **Metrics**:

  - **Detection-level**: precision, recall, F1 (match if predicted mid within ± tolerance of ground-truth jingle time)
  - **Timing error**: mean and median absolute timing error (seconds)
  - **False positives per hour** and **miss rate**

- **A/B** comparing two-stage (prefilter+classifier) vs brute-force sliding-window baseline.
- **Manual audit**: provide a zipped sample of detection clips for human verification; iterate on threshold / NMS.

---

## 10. Practical considerations & robustness (updated)

- **Use the canonical jingle to create cover variations** (pitch/time-stretch) to make the system robust to playback speed/pitch differences.
- **Phone codecs**: include compressed versions of the jingle (MP3/AMR) in synthetic training to simulate real-world degradation.
- **Fallbacks**: when classifier confidence is borderline, consult fingerprint matching or require human verification for those cases.

---

## 11. Deployment & operations (updated)

- Add `scripts/prefilter.py` to the inference container; when deployed, allow configuration flags `--prefilter {none,xcorr,fingerprint}` and `--prefilter-threshold` so operators can tune recall vs compute.
- Containerization: include the canonical jingle and its precomputed embeddings/fingerprint in the container image or in a mounted `data/reference/` directory.

---

## 12. Risks & mitigations (updated)

- **Risk**: Overfitting to the exact jingle rendering (model learns measurement artifacts). — _Mitigation_: heavy augmentation, compressions, and synthetic mixing with diverse backgrounds.
- **Risk**: False positives where ambient audio resembles parts of the jingle. — _Mitigation_: classifier verification stage and fingerprint checks.
- **Risk**: Using detected jingles for timestamp synchronization when phone clocks are wrong. — _Mitigation_: do not use jingle detections alone to correct clock drift; require explicit sync beep or filename timestamp.

---

## 13. Next steps (action items — updated)

1. Save canonical jingle to `data/reference/jingle.wav` (if not already) and create `data/reference/jingle_canonical.wav` with normalization and trimming.
2. Run `scripts/make_synthetic.py` to create synthetic training/validation sets mixing the canonical jingle into representative background recordings across SNRs.
3. Implement `scripts/prefilter.py` (FFT cross-correlation and fingerprint options) and run it over a few long recordings to inspect candidate peaks.
4. Implement training script (`scripts/train.py`) and run a baseline model trained on the synthetic + real labeled chunks.
5. Implement inference script (`scripts/infer_sliding.py`) that supports a `--prefilter` option and exports `detections.csv` and clips.
6. Tune threshold and NMS parameters using holdout set and SNR sweep; generate evaluation report.
7. Iterate on augmentation and retrain if needed; re-evaluate.

---

## 14. Example commands (updated)

```bash
# Install minimal dependencies
pip install numpy librosa tensorflow soundfile pandas pydub scipy

# Make canonical jingle
python scripts/make_synthetic.py --mode canonical --input data/reference/jingle.wav --out data/reference/jingle_canonical.wav

# Generate synthetic long recordings (SNR sweep)
python scripts/make_synthetic.py --background recordings/background1.wav --jingle data/reference/jingle_canonical.wav --out data/synthetic/ --snrs 10 5 0 -5 -10 --count 100

# Prefilter a long recording (generate candidate peaks)
python scripts/prefilter.py --method xcorr --jingle data/reference/jingle_canonical.wav --input recordings/route1.wav --out outputs/prefilter/route1_peaks.csv

# Train (example)
python scripts/train.py --data-dir data --out models/jingle_detector.h5

# Run sliding-window inference with prefilter
python scripts/infer_sliding.py --model models/jingle_detector.h5 --input recordings/route1.wav --out outputs/detections/route1.csv --clips outputs/clips/ --prefilter xcorr --prefilter-threshold 0.3
```

---

## 15. Acceptance criteria (updated)

- Detections exported to a CSV with at least `recording_file, mid_s, absolute_time, confidence, method` columns.
- Short audio clips for >95% of true positive detections are audibly correct on manual inspection of a sample set.
- Performance benchmarks on synthetic SNR sweep: recall >= target at SNR >= 0 dB (tune to your operational target). Record the SNR level corresponding to the target recall.
- Two-stage pipeline achieves similar recall to brute-force sliding-window while using substantially fewer classifier inferences (measured in inference count).

---

## 16. Contact & handoff

When ready, provide:

- A sample long recording (one or two routes) and
- A small labeled holdout of \~20 jingles with ground-truth timestamps

I will use these plus your canonical `data/reference/jingle.wav` to run the first end-to-end evaluation and return the detection CSV + a brief results report.

---

_End of plan._
