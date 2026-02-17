# Multimodal Misinformation Detector

This project predicts whether an image-text pair is likely misinformation or likely consistent.

The app uses:
- CLIP (`ViT-B/32`) to extract image and text embeddings.
- A trained `LogisticRegression` classifier (`model.pkl`) to produce the final prediction.
- Streamlit for the web interface and deployment.

## What This Project Does

Given:
- an uploaded image (`jpg`, `jpeg`, `png`)
- a text input (caption, post text, claim)

the app:
1. Encodes image and text with CLIP.
2. Builds a feature vector that matches the training pipeline.
3. Runs the logistic regression classifier.
4. Returns:
- predicted class (`Likely Misinformation` or `Likely Consistent`)
- misinformation probability
- confidence band (`Low`, `Medium`, `High`)

## Model Pipeline

The classifier was trained on a feature vector with **1537 features**:
- `1` cosine similarity between image and text embeddings
- `512` absolute difference (`|img_emb - txt_emb|`)
- `1024` concatenation (`[img_emb, txt_emb]`)

Total: `1 + 512 + 1024 = 1537`

This is important: inference must build the same 1537-dimensional vector or the classifier will fail.

## Project Structure

```text
MBD_Multimodal_Misinformation/
├─ app.py                     # Streamlit app (main entry point)
├─ model.pkl                  # Trained LogisticRegression model
├─ requirements.txt           # Python dependencies
├─ packages.txt               # Apt dependencies for Streamlit Cloud
├─ runtime.txt                # Python runtime for deployment
├─ .streamlit/config.toml     # Streamlit server/runner config
├─ notebooks/
│  └─ MBD_Technical_Group.ipynb   # Training and experimentation notebook
├─ data/                      # Data assets (project-specific)
├─ models/                    # Additional model artifacts (if used)
├─ outputs/                   # Generated outputs/results
└─ src/                       # Python package area (currently minimal)
```

## Local Development

### 1. Create and activate a virtual environment

Windows (PowerShell):

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Deployment (Streamlit Community Cloud)

### Required app settings

- Repository: your GitHub repo
- Branch: `main` (or your target branch)
- Main file path: `app.py`

### Dependency files used by Streamlit Cloud

- `requirements.txt` for Python packages
- `packages.txt` for system packages (`build-essential`, `git`)
- `runtime.txt` for Python version

### Current deployment notes

- `numpy` is pinned to `1.26.4` for Streamlit compatibility.
- `pillow` is pinned to `10.4.0` for Streamlit compatibility.
- CLIP is installed via `clip-anytorch==2.6.0` (PyPI package), which is more stable for Streamlit Cloud than installing from a GitHub URL during build.

## Inference Flow in `app.py`

1. Load CLIP model and preprocessing function.
2. Load classifier (`model.pkl`) with `joblib`.
3. Validate user input (image + non-empty text).
4. Extract CLIP image/text embeddings.
5. Build 1537-dim feature vector.
6. Predict class and probability with logistic regression.
7. Display prediction, risk bar, and explanation.

## Troubleshooting

### Feature shape mismatch error

Error example:
- `X has 1024 features, but LogisticRegression is expecting 1537 features`

Cause:
- Inference features do not match training features.

Fix:
- Ensure inference uses:
- cosine similarity (`1`)
- absolute difference (`512`)
- concatenation (`1024`)

### `ModuleNotFoundError: No module named 'clip'`

Cause:
- CLIP package missing in deployment environment.

Fix:
- Keep `clip-anytorch==2.6.0` in `requirements.txt`.
- Reboot app after deploy.
- If needed, clear Streamlit cache and reboot.

### Dependency resolver conflicts on Streamlit Cloud

Cause:
- Incompatible pinned versions (for example `numpy==2.x` with older Streamlit).

Fix:
- Keep compatible pins in `requirements.txt`.
- Redeploy after pushing changes.

## Reproducibility Notes

- Model file: `model.pkl`
- Expected model type: `LogisticRegression`
- Expected input dimension: `1537`
- CLIP backbone in app: `ViT-B/32`
- Inference device: CPU

## Limitations

- This system estimates risk; it does not verify truth in a fact-checking sense.
- Output quality depends on training data quality and coverage.
- Out-of-domain or adversarial content may reduce reliability.

## Suggested Next Improvements

1. Save full training artifacts (scaler/preprocessor + classifier) in one pipeline object.
2. Add evaluation section in README with metrics and dataset split details.
3. Add unit tests for feature-shape checks and model loading.
4. Add CI to validate dependencies and app startup on each push.

## License

Add your preferred license here (for example MIT, Apache-2.0, or proprietary).
