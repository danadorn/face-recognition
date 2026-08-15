# Face Recognition Project Detail

## Project Title

Face Verification System

## Project Overview

This project is a web-based face verification application built with Streamlit. It allows a user to enter a reference name, upload or scan a labeled reference face image as Image A, upload or scan a second face image as Image B, extract identity embeddings using an ArcFace ResNet100 model, and compare the embeddings with cosine similarity.

The final output tells the user whether Image B matches the reference image. If the similarity score is high enough, the app displays the entered reference name; otherwise, it displays `NOT MATCH`. The application displays both input images with detected face bounding boxes and a styled verification result card inside a modal dialog.

## Main Objective

The main objective of this project is to demonstrate a simple artificial intelligence workflow for face verification:

1. Accept a user-entered reference name, a Face A reference image, and a Face B comparison image from upload or camera capture.
2. Detect the face region in each image.
3. Generate a numerical face embedding for each detected face.
4. Compare the embeddings using cosine similarity.
5. Display the reference name when Face B matches the reference threshold, or `NOT MATCH` when it does not.

## Technology Stack

| Technology | Purpose |
| --- | --- |
| Python | Main programming language |
| Streamlit | Web application interface |
| OpenCV | Image processing and Haar Cascade face detection |
| NumPy | Numerical operations and similarity calculation |
| OpenVINO | Model loading and inference for ArcFace |
| ArcFace ResNet100 | Face embedding and verification model |
| Haar Cascade | Frontal face detection |

## Project Files

```text
.
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── models/
└── stored-faces/
```

### app.py

This is the main application file. It contains the full Streamlit interface, model loading logic, face detection, face embedding extraction, similarity calculation, and result display.

Main responsibilities:

- Creates required runtime folders.
- Downloads the Haar Cascade XML file if it is missing.
- Downloads the ArcFace ONNX model if it is missing.
- Loads the ArcFace model with OpenVINO.
- Detects the largest face in each uploaded or camera-captured image.
- Preprocesses faces for ArcFace.
- Generates normalized 512-dimensional embeddings.
- Computes cosine similarity.
- Displays the reference name or `NOT MATCH` in the Streamlit UI.

### README.md

The README gives a short explanation of the project, local setup instructions, deployment instructions for Streamlit Community Cloud, and a basic file structure.

### requirements.txt

This file lists the Python packages required to run the application:

```text
opencv-python-headless==4.13.0.92
numpy
streamlit
openvino
```

### .gitignore

This file excludes generated, temporary, and large runtime files from Git. Important ignored project paths include:

- `models/`
- `stored-faces/`
- `*.onnx`
- `__pycache__/`
- `.env`
- `.pytest_cache/`

## Runtime Folders

### models/

The `models/` folder is created automatically when the app starts. It stores downloaded model files:

- `haarcascade_frontalface_default.xml`
- `arcfaceresnet100-8.onnx`

These files are not committed to Git because the ArcFace model is large. The app downloads them automatically on first run.

### stored-faces/

The `stored-faces/` folder is also created automatically. In the current implementation, it is prepared as a runtime folder but is not actively used for permanent face storage.

## System Workflow

### 1. Application Startup

When `app.py` starts, it creates two folders if they do not already exist:

- `models/`
- `stored-faces/`

Then the application loads required machine learning resources through the cached `load_models()` function.

### 2. Model Download and Loading

The app checks whether the Haar Cascade file and ArcFace ONNX model are available locally.

If they are missing:

- The Haar Cascade XML is downloaded from the OpenCV GitHub repository.
- The ArcFace ResNet100 ONNX model is downloaded from the ONNX model repository.

After downloading, OpenVINO loads and compiles the ArcFace model for CPU inference.

Streamlit caches this process with `@st.cache_resource`, so the model does not reload on every UI interaction.

### 3. Image Upload

The user provides two images through the Streamlit interface:

- Image A - reference face
- Image B - check match

Each image can be provided independently with:

- Upload Image
- Scan with Camera

Accepted file types:

- JPG
- JPEG
- PNG

The "Compare Faces" button is disabled until both images are available. Captured face data is stored in `st.session_state`, so a user can scan Face A once, then scan or upload Face B separately.

### 4. Face Detection

The application uses OpenCV Haar Cascade detection to find faces in each image.

If multiple faces are detected, the application selects the largest detected face. This is treated as the main face in the image.

If no face is detected in either image, the app shows an error message and stops the comparison.

### 5. Face Preprocessing

Each detected face is prepared for the ArcFace model:

1. Convert from BGR to RGB.
2. Resize to `112 x 112` pixels.
3. Change image shape from height-width-channel format to channel-height-width format.
4. Add a batch dimension.
5. Convert the tensor to `float32`.

### 6. Embedding Generation

The preprocessed face is passed into the ArcFace ResNet100 model through OpenVINO.

The model returns a face embedding. The app normalizes this embedding so that cosine similarity can be calculated reliably.

The result is a 512-dimensional vector that represents the identity features of the detected face.

### 7. Similarity Calculation

The app compares both face embeddings using a dot product:

```python
similarity = float(np.dot(embedding_a, embedding_b))
```

Because both embeddings are normalized, this dot product acts as cosine similarity.

### 8. Decision Rule

The project uses a fixed verification threshold:

```python
THRESHOLD = 0.70
```

Decision logic:

- If similarity is greater than or equal to `0.70`, the app returns the entered reference name.
- If similarity is less than `0.70`, the app returns `NOT MATCH`.

### 9. Result Display

After comparison, the application displays:

- Image A with a face bounding box.
- Image B with a face bounding box.
- Cosine similarity score.
- Current verification threshold.
- ArcFace embedding size.
- Final label decision.
- Short decision analysis.

The result card uses custom HTML and CSS rendered inside Streamlit.

## Key Functions

### load_models()

Downloads missing model files, loads the Haar Cascade detector, loads the ArcFace ONNX model, compiles it with OpenVINO, and returns the loaded resources.

### detect_main_face(image)

Converts an image to grayscale, detects faces with Haar Cascade, chooses the largest detected face, and returns the cropped face and bounding box.

### preprocess_arcface(face_bgr)

Converts a face image into the tensor format required by ArcFace.

### get_face_embedding(face)

Runs ArcFace inference and returns a normalized face embedding.

### draw_face_box(image_bgr, box, label)

Draws a green rectangle and label around the detected face.

### decode_image_file(image_file)

Decodes either an uploaded image or a camera-captured image into RGB format so both input methods use the same processing pipeline.

### process_face_image(image_rgb)

Runs the shared face detection and embedding pipeline for any image source.

### render_detected_face(processed_face, label)

Draws the bounding box and label on a processed face image.

### process_image_input(slot_key, image_file, stored_label)

Decodes, detects, embeds, and stores a captured face in `st.session_state`.

### build_result_html(similarity, is_match, matched_label)

Builds the styled result card shown after face verification.

### verify_stored_faces(face_a, face_b)

Compares already processed Face A and Face B embeddings:

1. Calculate cosine similarity.
2. Decide the reference name or `NOT MATCH`.
3. Render detected images with final labels.
4. Return detected images and result HTML.

## How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Open the app in a browser:

```text
http://localhost:8501
```

On the first run, the application downloads the required model files automatically.

## Deployment

The project is ready to deploy on Streamlit Community Cloud.

Deployment steps:

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select this repository.
5. Set `app.py` as the entry point.
6. Deploy the app.

The first deployment boot may take longer because the ArcFace model is downloaded at startup.

## Strengths

- Simple and clear user interface.
- No manual model setup required.
- Uses a recognized face verification model.
- Uses Streamlit caching to avoid repeated model loading.
- Works with common image formats.
- Displays both visual detection results and numerical similarity.

## Limitations

- Haar Cascade works best on clear, front-facing faces.
- The app only compares the largest detected face in each image.
- The threshold is fixed at `0.70` and may need tuning for different datasets.
- Image quality, lighting, angle, blur, and occlusion can affect the result.
- The app performs verification only; it does not identify a person from a database.
- The current implementation does not store user face data permanently.

## Possible Future Improvements

- Replace Haar Cascade with a stronger face detector such as RetinaFace, MTCNN, or OpenVINO face detection models.
- Add face alignment before ArcFace embedding extraction.
- Allow users to adjust the similarity threshold.
- Support multiple-face selection.
- Add a small face database for identity recognition.
- Add test cases for preprocessing, detection failure, and similarity decisions.
- Improve error handling for model download failures.
- Add privacy notices for uploaded or camera-captured face images.

## Conclusion

This project demonstrates a complete face verification pipeline using Python, Streamlit, OpenCV, OpenVINO, and ArcFace. It provides a simple interface where users can compare Face B against a named reference image and receive a visual and numerical comparison result. The project is suitable for learning about face detection, face embeddings, cosine similarity, and AI application deployment.
