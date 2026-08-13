import os
import cv2
import numpy as np
import urllib.request
from pathlib import Path
import streamlit as st
import openvino as ov

# ==========================================
# SETUP
# ==========================================

BASE_DIR = Path(".")
folders = [
    BASE_DIR / "models",
    BASE_DIR / "stored-faces",
]

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================================
# DOWNLOAD MODELS AND CASCADES
# ==========================================

HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
HAAR_CASCADE_FILE = "models/haarcascade_frontalface_default.xml"

ARCFACE_MODEL_URL = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
ARCFACE_MODEL_PATH = "models/arcfaceresnet100-8.onnx"


@st.cache_resource(show_spinner="Loading face detection and verification models...")
def load_models():
    if not os.path.exists(HAAR_CASCADE_FILE):
        urllib.request.urlretrieve(HAAR_CASCADE_URL, HAAR_CASCADE_FILE)

    haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    if haar_cascade.empty():
        haar_cascade = None

    if not os.path.exists(ARCFACE_MODEL_PATH):
        urllib.request.urlretrieve(ARCFACE_MODEL_URL, ARCFACE_MODEL_PATH)

    core = ov.Core()
    arcface_model = core.read_model(ARCFACE_MODEL_PATH)
    compiled_arcface = core.compile_model(arcface_model, "CPU")
    arcface_output = compiled_arcface.output(0)

    return haar_cascade, compiled_arcface, arcface_output


haar_cascade, compiled_arcface, arcface_output = load_models()

# ==========================================
# FACE VERIFICATION CONSTANTS
# ==========================================

THRESHOLD = 0.50

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def detect_main_face(image):
    """Detect the largest face in an image using Haar Cascade"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None, None

    # Choose largest detected face
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    face = image[y:y+h, x:x+w]

    return face, (x, y, w, h)


def preprocess_arcface(face_bgr):
    """Preprocess face for ArcFace model"""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (112, 112))
    face_tensor = np.transpose(face_rgb, (2, 0, 1))
    face_tensor = np.expand_dims(face_tensor, axis=0)
    return face_tensor.astype(np.float32)


def get_face_embedding(face):
    """Get ArcFace embedding for a face"""
    face_tensor = preprocess_arcface(face)
    result = compiled_arcface([face_tensor])
    embedding = result[arcface_output][0]
    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
    return embedding


def draw_face_box(image_bgr, box, label):
    """Draw a bounding box around detected face"""
    result = image_bgr.copy()
    x, y, w, h = box

    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        result,
        label,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    return result


def build_result_html(similarity, is_same):
    """Build the styled result card shown after verification"""
    if is_same:
        verdict = "SAME PERSON"
        accent = "#22c55e"
        badge_bg = "#052e16"
        badge_text = "#86efac"
        status_icon = "✓"
    else:
        verdict = "DIFFERENT PERSON"
        accent = "#ef4444"
        badge_bg = "#450a0a"
        badge_text = "#fca5a5"
        status_icon = "✕"

    display_score = max(0, min(100, similarity * 100))

    return f"""
    <div style="
        max-width:850px;
        margin:20px auto;
        font-family:Inter,Arial,sans-serif;
        color:#f9fafb;
    ">
        <div style="
            background:#111827;
            border:1px solid #374151;
            border-top:5px solid {accent};
            border-radius:16px;
            padding:28px;
            box-shadow:0 8px 25px rgba(0,0,0,0.25);
        ">
            <div style="text-align:center; margin-bottom:28px;">
                <div style="
                    display:inline-block;
                    padding:7px 16px;
                    border-radius:999px;
                    background:{badge_bg};
                    color:{badge_text};
                    font-size:12px;
                    font-weight:700;
                    letter-spacing:1px;
                ">
                    FACE VERIFICATION RESULT
                </div>
                <div style="
                    width:64px;
                    height:64px;
                    margin:20px auto 12px auto;
                    display:flex;
                    justify-content:center;
                    align-items:center;
                    border-radius:50%;
                    background:{badge_bg};
                    border:2px solid {accent};
                    color:{accent};
                    font-size:32px;
                    font-weight:bold;
                ">
                    {status_icon}
                </div>
                <h1 style="margin:0; color:{accent}; font-size:30px; font-weight:800;">
                    {verdict}
                </h1>
                <p style="margin:8px 0 0 0; color:#9ca3af; font-size:14px;">
                    ArcFace identity verification
                </p>
            </div>
            <div style="
                background:#1f2937;
                border:1px solid #374151;
                border-radius:12px;
                padding:20px;
                margin-bottom:18px;
            ">
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    margin-bottom:12px;
                ">
                    <div>
                        <div style="
                            color:#9ca3af;
                            font-size:12px;
                            text-transform:uppercase;
                            letter-spacing:0.8px;
                        ">
                            Cosine Similarity
                        </div>
                        <div style="color:#f9fafb; font-size:13px; margin-top:3px;">
                            Higher means more similar
                        </div>
                    </div>
                    <div style="color:{accent}; font-size:28px; font-weight:800;">
                        {similarity:.4f}
                    </div>
                </div>
                <div style="
                    width:100%;
                    height:10px;
                    border-radius:999px;
                    background:#374151;
                    overflow:hidden;
                ">
                    <div style="
                        width:{display_score:.1f}%;
                        height:100%;
                        border-radius:999px;
                        background:{accent};
                    "></div>
                </div>
                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-top:7px;
                    color:#6b7280;
                    font-size:11px;
                ">
                    <span>Lower similarity</span>
                    <span>Higher similarity</span>
                </div>
            </div>
            <div style="
                display:grid;
                grid-template-columns:repeat(auto-fit, minmax(140px, 1fr));
                gap:12px;
                margin-bottom:18px;
            ">
                <div style="background:#1f2937; border:1px solid #374151; border-radius:12px; padding:16px; text-align:center;">
                    <div style="color:#9ca3af; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.8px;">Similarity</div>
                    <div style="color:{accent}; font-size:22px; font-weight:800; margin-top:6px;">{similarity:.4f}</div>
                </div>
                <div style="background:#1f2937; border:1px solid #374151; border-radius:12px; padding:16px; text-align:center;">
                    <div style="color:#9ca3af; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.8px;">Threshold</div>
                    <div style="color:#f9fafb; font-size:22px; font-weight:800; margin-top:6px;">{THRESHOLD:.2f}</div>
                </div>
                <div style="background:#1f2937; border:1px solid #374151; border-radius:12px; padding:16px; text-align:center;">
                    <div style="color:#9ca3af; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.8px;">ArcFace</div>
                    <div style="color:#f9fafb; font-size:22px; font-weight:800; margin-top:6px;">512-D</div>
                </div>
            </div>
            <div style="
                background:#0f172a;
                border-left:4px solid {accent};
                border-radius:10px;
                padding:16px 18px;
            ">
                <div style="color:#f9fafb; font-weight:700; margin-bottom:7px;">Decision Analysis</div>
                <div style="color:#cbd5e1; font-size:13px; line-height:1.7;">
                    The cosine similarity score is
                    <strong style="color:#ffffff;">{similarity:.4f}</strong>.
                    This score is
                    <strong style="color:{accent};">{"above" if is_same else "below"}</strong>
                    the current verification threshold of
                    <strong style="color:#ffffff;">{THRESHOLD:.2f}</strong>.
                    Therefore, ArcFace classifies these two faces as
                    <strong style="color:{accent};">{verdict}</strong>.
                </div>
            </div>
            <div style="
                margin-top:16px;
                padding-top:14px;
                border-top:1px solid #374151;
                color:#6b7280;
                text-align:center;
                font-size:11px;
            ">
                Haar Cascade Face Detection
                &nbsp; • &nbsp;
                ArcFace ResNet100
                &nbsp; • &nbsp;
                512-dimensional embedding
            </div>
        </div>
    </div>
    """


def verify_faces(image_a_rgb, image_b_rgb):
    """Run detection + verification, returning (detected_a, detected_b, result_html, error)"""
    image_a = cv2.cvtColor(image_a_rgb, cv2.COLOR_RGB2BGR)
    image_b = cv2.cvtColor(image_b_rgb, cv2.COLOR_RGB2BGR)

    face_a, box_a = detect_main_face(image_a)
    face_b, box_b = detect_main_face(image_b)

    if face_a is None:
        return None, None, None, "No face detected in Image A. Try a clearer front-facing image."
    if face_b is None:
        return None, None, None, "No face detected in Image B. Try a clearer front-facing image."

    embedding_a = get_face_embedding(face_a)
    embedding_b = get_face_embedding(face_b)

    similarity = float(np.dot(embedding_a, embedding_b))
    is_same = similarity >= THRESHOLD

    detected_a = draw_face_box(image_a, box_a, "Face A")
    detected_b = draw_face_box(image_b, box_b, "Face B")

    detected_a_rgb = cv2.cvtColor(detected_a, cv2.COLOR_BGR2RGB)
    detected_b_rgb = cv2.cvtColor(detected_b, cv2.COLOR_BGR2RGB)

    result_html = build_result_html(similarity, is_same)

    return detected_a_rgb, detected_b_rgb, result_html, None


# ==========================================
# STREAMLIT INTERFACE
# ==========================================

st.set_page_config(page_title="Face Verification", page_icon="🧑‍🤝‍🧑", layout="centered")

st.title("Face Verification System")
st.markdown(
    """
    Upload two face images and let the system determine whether they likely
    represent the **same person** or **different people**.

    **Models**
    - Haar Cascade → Face Detection
    - ArcFace ResNet100 → Face Verification
    """
)

st.header("Upload Images")
col_a, col_b = st.columns(2)
with col_a:
    file_a = st.file_uploader("Image A", type=["jpg", "jpeg", "png"], key="image_a")
with col_b:
    file_b = st.file_uploader("Image B", type=["jpg", "jpeg", "png"], key="image_b")

if st.button("Compare Faces", type="primary", disabled=not (file_a and file_b)):
    image_a_rgb = cv2.imdecode(np.frombuffer(file_a.read(), np.uint8), cv2.IMREAD_COLOR_RGB)
    image_b_rgb = cv2.imdecode(np.frombuffer(file_b.read(), np.uint8), cv2.IMREAD_COLOR_RGB)

    detected_a, detected_b, result_html, error = verify_faces(image_a_rgb, image_b_rgb)

    if error:
        st.error(error)
    else:
        st.header("Detected Faces")
        out_a, out_b = st.columns(2)
        with out_a:
            st.image(detected_a, caption="Detected Face A")
        with out_b:
            st.image(detected_b, caption="Detected Face B")

        st.markdown(result_html, unsafe_allow_html=True)
elif not (file_a and file_b):
    st.info("Upload both images to enable comparison.")
