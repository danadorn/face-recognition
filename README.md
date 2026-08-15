# Face Recognition

A Face Verification System built with:
- **Haar Cascade** for face detection
- **ArcFace ResNet100** (via OpenVINO) for face verification

Enter a reference name, upload or scan Face A, then upload or scan Face B to
check. If Face B has a cosine similarity score of at least `0.70` against
Face A, the app displays the reference name. Otherwise, it displays
**NOT MATCH**.

Each face can use either:
- **Upload Image**
- **Scan with Camera** using `streamlit-webrtc`

The camera option captures from the user's browser instead of the server
camera. The WebRTC frame processor corrects mirrored camera frames before both
the live preview and the saved frame enter the face detection and ArcFace
pipeline. The UI uses a custom dark-mode card layout with styled controls and
result cards. Comparison results open in a native Streamlit dialog modal so the
decision is shown immediately after clicking Compare.

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
