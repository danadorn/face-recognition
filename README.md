---
title: Face Verification
emoji: 🧑‍🤝‍🧑
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.23.1
app_file: app.py
pinned: false
---

# Face Recognition

A Face Verification System built with:
- **Haar Cascade** for face detection
- **ArcFace ResNet100** (via OpenVINO) for face verification

Upload two face photos and the app reports whether they belong to the same
person, using cosine similarity between ArcFace embeddings.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The app will be available at `http://localhost:7860`. On first run it
automatically downloads the Haar cascade XML and the ArcFace ONNX model
(~250MB) into `models/` — no manual setup needed.

## Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space, selecting the
   **Gradio** SDK.
2. Push this repository to the Space's git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
3. The Space builds from `requirements.txt` and launches `app.py`
   automatically. The frontmatter at the top of this file configures the
   Space (title, SDK, entry point).

No model files need to be committed — `app.py` downloads them at startup.

## File structure

```
.
├── app.py              # Gradio app (auto-downloads models on startup)
├── requirements.txt    # Python dependencies
├── models/             # Downloaded at runtime (gitignored)
└── stored-faces/        # Runtime scratch output (gitignored)
```
