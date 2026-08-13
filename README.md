# Face Recognition

A Face Verification System built with:
- **Haar Cascade** for face detection
- **ArcFace ResNet100** (via OpenVINO) for face verification

Upload two face photos and the app reports whether they belong to the same
person, using cosine similarity between ArcFace embeddings.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will be available at `http://localhost:8501`. On first run it
automatically downloads the Haar cascade XML and the ArcFace ONNX model
(~250MB) into `models/` — no manual setup needed.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (models are gitignored — `app.py`
   downloads them at startup, so nothing large needs to be committed).
2. Go to https://share.streamlit.io, sign in, and click **New app**.
3. Point it at this repo, the branch, and `app.py` as the entry point.
4. Deploy. The first boot takes a bit longer while the ~250MB ArcFace
   model downloads; it's cached afterward via `@st.cache_resource`.

## File structure

```
.
├── app.py              # Streamlit app (auto-downloads models on startup)
├── requirements.txt    # Python dependencies
├── models/              # Downloaded at runtime (gitignored)
└── stored-faces/        # Runtime scratch output (gitignored)
```
