import os
import cv2
import numpy as np
import urllib.request
import hashlib
import threading
from pathlib import Path
import streamlit as st
import openvino as ov

try:
    import av
    from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer
    STREAMLIT_WEBRTC_AVAILABLE = True
except ImportError:
    av = None
    WebRtcMode = None
    VideoProcessorBase = object
    webrtc_streamer = None
    STREAMLIT_WEBRTC_AVAILABLE = False

st.set_page_config(page_title="Face Verification", page_icon=":material/face:", layout="wide")

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

DEFAULT_REFERENCE_NAME = "Reference"
THRESHOLD = 0.70
CAMERA_FRAME_IS_MIRRORED = True
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
CAMERA_MEDIA_CONSTRAINTS = {
    "video": {
        "facingMode": "user",
        "width": {"ideal": 1280, "min": 960},
        "height": {"ideal": 720, "min": 540},
        "frameRate": {"ideal": 30},
    },
    "audio": False,
}
CAMERA_VIDEO_HTML_ATTRS = {
    "autoPlay": True,
    "muted": True,
    "playsInline": True,
    "style": {"width": "100%", "height": "auto"},
}

# ==========================================
# UI STYLING
# ==========================================

def inject_custom_css():
    """Apply a polished dark SaaS visual layer over Streamlit widgets."""
    st.markdown(
        """
        <style>
            :root {
                --bg: #000000;
                --surface: #0b0b0b;
                --surface-strong: #151515;
                --surface-soft: #050505;
                --border: rgba(148, 163, 184, 0.18);
                --border-strong: rgba(255, 145, 0, 0.42);
                --text: #f8fafc;
                --muted: #94a3b8;
                --accent: #FF9100;
                --accent-strong: #FFB14A;
                --accent-soft: rgba(255, 145, 0, 0.16);
                --info: #FF9100;
                --success: #22c55e;
                --danger: #ef4444;
                --radius: 20px;
                --shadow: 0 22px 55px rgba(0, 0, 0, 0.36);
            }

            .stApp {
                background: #000000;
                color: var(--text);
            }

            [data-testid="stAppViewContainer"] > .main .block-container {
                max-width: 1180px;
                padding: 36px 28px 64px;
            }

            h1, h2, h3, p, label, span, div {
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }

            h1, h2, h3 {
                letter-spacing: 0;
            }

            [data-testid="stMarkdownContainer"] p {
                color: var(--muted);
            }

            .app-header {
                margin-bottom: 28px;
                padding: 30px;
                border: 1px solid var(--border);
                border-radius: 28px;
                background: rgba(11, 11, 11, 0.92);
                box-shadow: var(--shadow);
            }

            .eyebrow {
                margin-bottom: 12px;
                color: var(--info);
                font-size: 12px;
                font-weight: 800;
                letter-spacing: 0.14em;
                text-transform: uppercase;
            }

            .app-title {
                margin: 0;
                color: var(--text);
                font-size: clamp(34px, 5vw, 58px);
                font-weight: 850;
                line-height: 1.02;
            }

            .app-subtitle {
                max-width: 760px;
                margin: 16px 0 22px;
                color: #d1d5db;
                font-size: 16px;
                line-height: 1.7;
            }

            .model-pills {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }

            .model-pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 12px;
                border: 1px solid var(--border);
                border-radius: 999px;
                background: rgba(5, 5, 5, 0.82);
                color: #f3f4f6;
                font-size: 13px;
                font-weight: 700;
            }

            .section-title {
                margin: 24px 0 14px;
                color: var(--text);
                font-size: 22px;
                font-weight: 800;
            }

            .st-key-reference_name_panel {
                margin: 4px 0 12px;
                padding: 18px 20px;
                border: 1px solid var(--border);
                border-radius: 18px;
                background: rgba(11, 11, 11, 0.88);
                box-shadow: 0 14px 34px rgba(0, 0, 0, 0.20);
            }

            .st-key-reference_name_panel [data-testid="stTextInput"] {
                width: 100%;
            }

            div[data-testid="stHorizontalBlock"]:has(.st-key-face_a_card):has(.st-key-face_b_card) {
                display: grid !important;
                grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;
                gap: 20px !important;
                align-items: stretch !important;
                justify-items: stretch !important;
                width: 100% !important;
            }

            div[data-testid="stHorizontalBlock"]:has(.st-key-face_a_card):has(.st-key-face_b_card) > div[data-testid="stColumn"] {
                display: flex;
                flex-direction: column;
                width: 100% !important;
                min-width: 0 !important;
                max-width: none !important;
                flex: unset !important;
            }

            div[data-testid="stHorizontalBlock"]:has(.st-key-face_a_card):has(.st-key-face_b_card) > div[data-testid="stColumn"] > div {
                width: 100% !important;
                height: 100%;
                min-width: 0 !important;
                max-width: none !important;
            }

            div[data-testid="stHorizontalBlock"]:has(.st-key-face_a_card):has(.st-key-face_b_card) > div[data-testid="stColumn"] > div > div,
            div[data-testid="stHorizontalBlock"]:has(.st-key-face_a_card):has(.st-key-face_b_card) [data-testid="stVerticalBlock"],
            div[data-testid="stHorizontalBlock"]:has(.st-key-face_a_card):has(.st-key-face_b_card) [data-testid="stElementContainer"] {
                width: 100% !important;
                min-width: 0 !important;
                max-width: none !important;
            }

            .st-key-face_a_card,
            .st-key-face_b_card {
                width: 100% !important;
                min-width: 0 !important;
                max-width: none !important;
                height: 100%;
                padding: 24px;
                border: 1px solid var(--border);
                border-radius: var(--radius);
                background: rgba(11, 11, 11, 0.96);
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.26);
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                transition: border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
            }

            .st-key-face_a_card:hover,
            .st-key-face_b_card:hover {
                border-color: var(--border-strong);
                transform: translateY(-1px);
                box-shadow: 0 24px 55px rgba(0, 0, 0, 0.34);
            }

            .input-card-header {
                display: grid;
                grid-template-columns: 56px minmax(0, 1fr);
                column-gap: 18px;
                align-items: start;
                min-height: 92px;
                margin-bottom: 18px;
            }

            .avatar-token {
                width: 56px;
                height: 56px;
                display: grid;
                place-items: center;
                flex: 0 0 auto;
                border-radius: 16px;
                background: var(--accent-soft);
                border: 1px solid rgba(255, 145, 0, 0.38);
                color: #FFE0B3;
                font-size: 18px;
                font-weight: 850;
            }

            .input-card-heading {
                min-width: 0;
                padding-top: 0;
            }

            .input-card-title {
                margin: 0 !important;
                padding: 0 !important;
                color: var(--text);
                font-size: 34px;
                font-weight: 820;
                line-height: 56px;
            }

            .input-card-caption {
                margin: 8px 0 0 !important;
                padding: 0 !important;
                color: var(--muted);
                font-size: 16px;
                line-height: 1.3;
            }

            [data-testid="stTextInput"] label,
            [data-testid="stFileUploader"] label,
            [data-testid="stRadio"] label {
                color: #f3f4f6 !important;
                font-weight: 720;
            }

            [data-testid="stTextInput"] input {
                border: 1px solid var(--border);
                border-radius: 14px;
                background: rgba(5, 5, 5, 0.90);
                color: var(--text);
                min-height: 44px;
            }

            [data-testid="stTextInput"] input:focus {
                border-color: var(--accent-strong);
                box-shadow: 0 0 0 3px rgba(255, 145, 0, 0.18);
            }

            [data-testid="stRadio"],
            [data-testid="stFileUploader"] {
                width: 100% !important;
                min-width: 0 !important;
                max-width: none !important;
            }

            div[role="radiogroup"] {
                width: 100% !important;
                min-width: 0 !important;
                max-width: none !important;
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 6px;
                padding: 5px;
                margin-top: 4px;
                border: 1px solid var(--border);
                border-radius: 999px;
                background: rgba(5, 5, 5, 0.72);
                box-sizing: border-box;
            }

            div[role="radiogroup"] label {
                width: 100% !important;
                min-width: 0 !important;
                height: 38px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                gap: 0 !important;
                margin: 0 !important;
                min-height: 38px;
                padding: 0 12px;
                border-radius: 999px;
                color: var(--muted) !important;
                font-size: 13px;
                font-weight: 800;
                line-height: 1;
                overflow: hidden;
                position: relative;
                transition: background 150ms ease, color 150ms ease, box-shadow 150ms ease;
            }

            div[role="radiogroup"] label input {
                position: absolute !important;
                width: 0 !important;
                height: 0 !important;
                opacity: 0 !important;
                pointer-events: none !important;
            }

            div[role="radiogroup"] label p {
                margin: 0 !important;
                line-height: 1 !important;
                color: inherit !important;
                text-align: center !important;
                white-space: nowrap;
            }

            div[role="radiogroup"] label > div,
            div[role="radiogroup"] label div:has(> [data-testid="stMarkdownContainer"]) {
                width: 100% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                gap: 0 !important;
            }

            div[role="radiogroup"] label div:has(> [data-testid="stMarkdownContainer"]) > div:not([data-testid="stMarkdownContainer"]) {
                display: none !important;
            }

            div[role="radiogroup"] label:has(input:checked) {
                background: linear-gradient(135deg, var(--accent), var(--accent-strong));
                color: white !important;
                box-shadow: 0 10px 24px rgba(255, 145, 0, 0.28);
            }

            div[role="radiogroup"] label > div:first-child,
            div[role="radiogroup"] label span:first-child,
            div[role="radiogroup"] label svg {
                display: none !important;
            }

            [data-testid="stFileUploader"] section {
                width: 100% !important;
                min-width: 0 !important;
                max-width: none !important;
                min-height: 68px;
                border-radius: 18px;
                border: 1px dashed rgba(148, 163, 184, 0.42);
                background: rgba(8, 8, 8, 0.82);
                box-sizing: border-box;
                transition: border-color 150ms ease, background 150ms ease;
            }

            [data-testid="stFileUploader"] section > div {
                width: 100% !important;
                min-width: 0 !important;
            }

            [data-testid="stFileUploader"] section:hover {
                border-color: var(--accent-strong);
                background: rgba(255, 145, 0, 0.10);
            }

            [data-testid="stImage"] img {
                border-radius: 18px;
                border: 1px solid var(--border);
                box-shadow: 0 16px 34px rgba(0, 0, 0, 0.28);
            }

            [data-testid="stImageCaption"] {
                color: #d1d5db;
                font-weight: 700;
            }

            .stButton > button {
                min-height: 44px;
                border-radius: 999px !important;
                font-weight: 820 !important;
                letter-spacing: 0;
                transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease, background 140ms ease;
            }

            .stButton > button:hover:enabled {
                transform: translateY(-1px);
            }

            .stButton > button:disabled,
            button:disabled {
                width: 100%;
                min-height: 48px;
                border: 1px solid rgba(148, 163, 184, 0.18) !important;
                border-radius: 999px !important;
                background: #2a2a2a !important;
                color: #94a3b8 !important;
                box-shadow: none !important;
                opacity: 1 !important;
            }

            .stButton > button:disabled *,
            button:disabled * {
                color: #94a3b8 !important;
            }

            .st-key-compare_button button,
            div[data-testid="stButton"] button[kind="primary"] {
                width: 100%;
                border: 0 !important;
                background: linear-gradient(135deg, var(--accent), var(--accent-strong)) !important;
                color: white !important;
                box-shadow: 0 16px 34px rgba(255, 145, 0, 0.34);
            }

            .st-key-compare_button button:disabled,
            div[data-testid="stButton"] button[kind="primary"]:disabled {
                background: #2a2a2a !important;
                color: #94a3b8 !important;
                box-shadow: none !important;
            }

            .st-key-reset_button button {
                border: 1px solid rgba(148, 163, 184, 0.34) !important;
                background: transparent !important;
                color: #f3f4f6 !important;
            }

            .st-key-reset_button button:hover {
                border-color: var(--accent-strong) !important;
                color: white !important;
                background: rgba(255, 145, 0, 0.12) !important;
            }

            .st-key-action_row {
                margin-top: 14px;
            }

            .st-key-action_row [data-testid="stHorizontalBlock"] {
                display: flex !important;
                gap: 12px !important;
                align-items: center !important;
                width: 100% !important;
                max-width: 100% !important;
            }

            .st-key-action_row [data-testid="stHorizontalBlock"]:has(.st-key-compare_button):has(.st-key-reset_button) {
                display: flex !important;
                gap: 12px !important;
                align-items: center !important;
                width: 100% !important;
                max-width: 100% !important;
            }

            .st-key-action_row [data-testid="stHorizontalBlock"]:has(.st-key-compare_button):has(.st-key-reset_button) > [data-testid="stColumn"]:first-child {
                flex: 0 0 calc(80% - 6px) !important;
                width: calc(80% - 6px) !important;
                min-width: 0 !important;
                max-width: calc(80% - 6px) !important;
            }

            .st-key-action_row [data-testid="stHorizontalBlock"]:has(.st-key-compare_button):has(.st-key-reset_button) > [data-testid="stColumn"]:last-child {
                flex: 0 0 calc(20% - 6px) !important;
                width: calc(20% - 6px) !important;
                min-width: 0 !important;
                max-width: calc(20% - 6px) !important;
            }

            .st-key-action_row [data-testid="stButton"],
            .st-key-action_row .st-key-compare_button button,
            .st-key-action_row .st-key-reset_button button {
                width: 100% !important;
            }

            .st-key-action_row [data-testid="stButton"] {
                min-height: 48px !important;
                display: flex !important;
                align-items: center !important;
            }

            .st-key-action_row .stButton > button {
                min-height: 48px !important;
                height: 48px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                padding: 0 18px !important;
                box-sizing: border-box !important;
            }

            .custom-alert {
                display: flex;
                align-items: flex-start;
                gap: 12px;
                margin-top: 18px;
                padding: 14px 16px;
                border: 1px solid rgba(255, 145, 0, 0.24);
                border-radius: 16px;
                background: rgba(255, 145, 0, 0.10);
                color: #ffe0b3;
                font-size: 14px;
                line-height: 1.55;
            }

            .alert-icon {
                width: 24px;
                height: 24px;
                display: grid;
                place-items: center;
                flex: 0 0 auto;
                border-radius: 50%;
                background: rgba(255, 145, 0, 0.16);
                color: var(--info);
                font-weight: 900;
            }

            @media (max-width: 760px) {
                [data-testid="stAppViewContainer"] > .main .block-container {
                    padding: 24px 16px 48px;
                }

                .app-header {
                    padding: 22px;
                    border-radius: 22px;
                }

                div[role="radiogroup"] {
                    grid-template-columns: 1fr;
                    border-radius: 18px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def detect_main_face(image):
    """Detect the largest face in an image using Haar Cascade"""
    if haar_cascade is None:
        return None, None

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


def decode_image_file(image_file):
    """Decode an uploaded image into RGB format."""
    image_bytes = image_file.getvalue()
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if image_bgr is None:
        return None, "Could not read the selected image. Try another JPG or PNG file.", None

    signature = hashlib.sha256(image_bytes).hexdigest()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, None, signature


def process_face_image(image_rgb):
    """Run the shared face detection and embedding pipeline for any input source."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    face, box = detect_main_face(image_bgr)

    if face is None:
        return None, "No face detected. Try a clearer front-facing image."

    embedding = get_face_embedding(face)

    return {
        "image_bgr": image_bgr,
        "box": box,
        "embedding": embedding,
    }, None


def render_detected_face(processed_face, label):
    """Draw a label on a processed face image and return it as RGB."""
    detected_bgr = draw_face_box(processed_face["image_bgr"], processed_face["box"], label)
    return cv2.cvtColor(detected_bgr, cv2.COLOR_BGR2RGB)


def build_image_signature(prefix, image_rgb):
    """Create a stable signature for an already decoded RGB image."""
    digest = hashlib.sha256()
    digest.update(str(image_rgb.shape).encode("utf-8"))
    digest.update(image_rgb.tobytes())
    return f"{prefix}:{digest.hexdigest()}"


def store_processed_face_image(slot_key, image_rgb, stored_label, signature):
    """Run detection/embedding for any RGB image and store the processed face."""
    stored_face = st.session_state.get(slot_key)
    if stored_face and stored_face.get("signature") == signature:
        return

    processed_face, process_error = process_face_image(image_rgb)
    if process_error:
        st.session_state[slot_key] = None
        st.error(process_error)
        return

    processed_face["signature"] = signature
    processed_face["detected_rgb"] = render_detected_face(processed_face, stored_label)
    st.session_state[slot_key] = processed_face


def process_uploaded_image_input(slot_key, image_file, stored_label):
    """Decode, detect, embed, and store an uploaded face image."""
    if image_file is None:
        return

    image_rgb, decode_error, signature = decode_image_file(image_file)
    if decode_error:
        st.session_state[slot_key] = None
        st.error(decode_error)
        return

    store_processed_face_image(slot_key, image_rgb, stored_label, f"{signature}:upload")


class OrientationCorrectingVideoProcessor(VideoProcessorBase):
    """Correct camera frames before preview display and before capture storage."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_frame_rgb = None

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")

        if CAMERA_FRAME_IS_MIRRORED:
            image_bgr = cv2.flip(image_bgr, 1)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._latest_frame_rgb = image_rgb.copy()

        return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")

    def get_latest_frame_rgb(self):
        with self._lock:
            if self._latest_frame_rgb is None:
                return None

            return self._latest_frame_rgb.copy()


def render_webrtc_camera_capture(slot_key, widget_key, stored_label, capture_label):
    """Render the controlled camera preview and store the latest corrected frame."""
    if not STREAMLIT_WEBRTC_AVAILABLE:
        st.error("Camera scan requires streamlit-webrtc and av. Install requirements and restart the app.")
        return

    ctx = webrtc_streamer(
        key=widget_key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=CAMERA_MEDIA_CONSTRAINTS,
        video_processor_factory=OrientationCorrectingVideoProcessor,
        async_processing=True,
        video_receiver_size=1,
        video_html_attrs=CAMERA_VIDEO_HTML_ATTRS,
    )

    if st.button(capture_label, key=f"{widget_key}_capture", use_container_width=True):
        processor = ctx.video_processor
        if processor is None:
            st.error("Camera is still starting. Try again in a moment.")
            return

        image_rgb = processor.get_latest_frame_rgb()
        if image_rgb is None:
            st.error("No camera frame is available yet. Try again in a moment.")
            return

        signature = build_image_signature(f"{widget_key}:webrtc-unmirrored", image_rgb)
        store_processed_face_image(slot_key, image_rgb, stored_label, signature)


def build_result_html(similarity, is_match, result_label, face_a_name):
    """Build the styled result card shown after verification"""
    if is_match:
        verdict = result_label
        accent = "#22c55e"
        badge_bg = "#052e16"
        badge_text = "#86efac"
        status_icon = "MATCH"
        verdict_caption = "Face B matched the reference identity"
    else:
        verdict = "NOT MATCH"
        accent = "#ef4444"
        badge_bg = "#450a0a"
        badge_text = "#fca5a5"
        status_icon = "NO MATCH"
        verdict_caption = "Face B did not match the reference identity"

    display_score = max(0, min(100, similarity * 100))

    return f"""
    <div style="
        max-width:850px;
        margin:28px auto 12px;
        font-family:Inter,Arial,sans-serif;
        color:#f9fafb;
    ">
        <div style="
            background:linear-gradient(180deg, rgba(15,15,15,0.98), rgba(5,5,5,0.98));
            border:1px solid rgba(148,163,184,0.18);
            border-top:5px solid {accent};
            border-radius:28px;
            padding:34px;
            box-shadow:0 24px 60px rgba(0,0,0,0.38);
        ">
            <div style="text-align:center; margin-bottom:30px;">
                <div style="
                    display:inline-block;
                    padding:8px 16px;
                    border-radius:999px;
                    background:{badge_bg};
                    color:{badge_text};
                    font-size:12px;
                    font-weight:700;
                    letter-spacing:1px;
                ">
                    FACE MATCH RESULT
                </div>
                <div style="
                    display:inline-flex;
                    min-width:96px;
                    height:42px;
                    padding:0 18px;
                    margin:22px auto 14px auto;
                    display:flex;
                    justify-content:center;
                    align-items:center;
                    border-radius:999px;
                    background:{badge_bg};
                    border:1px solid {accent};
                    color:{accent};
                    font-size:13px;
                    font-weight:900;
                    letter-spacing:0.08em;
                ">
                    {status_icon}
                </div>
                <h1 style="margin:0; color:{accent}; font-size:clamp(40px, 6vw, 68px); font-weight:900; line-height:1;">
                    {verdict}
                </h1>
                <p style="margin:12px 0 0 0; color:#d1d5db; font-size:15px;">
                    {verdict_caption}
                </p>
            </div>
            <div style="
                display:grid;
                grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));
                gap:14px;
                margin-bottom:18px;
            ">
                <div style="background:#050505; border:1px solid rgba(148,163,184,0.16); border-radius:18px; padding:18px; text-align:center;">
                    <div style="color:#94a3b8; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:0.08em;">Similarity</div>
                    <div style="color:{accent}; font-size:30px; font-weight:900; margin-top:6px;">{similarity:.4f}</div>
                </div>
                <div style="background:#050505; border:1px solid rgba(148,163,184,0.16); border-radius:18px; padding:18px; text-align:center;">
                    <div style="color:#94a3b8; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:0.08em;">Match Threshold</div>
                    <div style="color:#f8fafc; font-size:30px; font-weight:900; margin-top:6px;">{THRESHOLD:.2f}</div>
                </div>
                <div style="background:#050505; border:1px solid rgba(148,163,184,0.16); border-radius:18px; padding:18px; text-align:center;">
                    <div style="color:#94a3b8; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:0.08em;">Reference Name</div>
                    <div style="color:#f8fafc; font-size:30px; font-weight:900; margin-top:6px;">{face_a_name}</div>
                </div>
            </div>
            <div style="
                width:100%;
                height:11px;
                border-radius:999px;
                background:#2a2a2a;
                overflow:hidden;
                margin:2px 0 18px;
            ">
                <div style="
                    width:{display_score:.1f}%;
                    height:100%;
                    border-radius:999px;
                    background:{accent};
                "></div>
            </div>
            <div style="
                background:#050505;
                border-left:4px solid {accent};
                border-radius:16px;
                padding:16px 18px;
            ">
                <div style="color:#f9fafb; font-weight:800; margin-bottom:7px;">Decision Analysis</div>
                <div style="color:#d1d5db; font-size:13px; line-height:1.7;">
                    The cosine similarity score is
                    <strong style="color:#ffffff;">{similarity:.4f}</strong>.
                    This score is
                    <strong style="color:{accent};">{"above" if is_match else "below"}</strong>
                    the current verification threshold of
                    <strong style="color:#ffffff;">{THRESHOLD:.2f}</strong>.
                    Therefore, Face B is classified as
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


def verify_stored_faces(face_a, face_b, face_a_name):
    """Compare already processed face embeddings and prepare display output."""
    similarity = float(np.dot(face_a["embedding"], face_b["embedding"]))
    print(
        f"[face-verification] similarity={similarity:.4f}, "
        f"threshold={THRESHOLD:.2f}, reference={face_a_name}"
    )

    if similarity >= THRESHOLD:
        result_label = face_a_name
    else:
        result_label = "NOT MATCH"

    is_match = similarity >= THRESHOLD
    detected_a_rgb = render_detected_face(face_a, f"Face A: {face_a_name}")
    detected_b_rgb = render_detected_face(face_b, result_label)
    result_html = build_result_html(similarity, is_match, result_label, face_a_name)

    return detected_a_rgb, detected_b_rgb, result_html, None, similarity, is_match


@st.dialog("Face match result", width="large")
def show_result_dialog(similarity, threshold, reference_name, is_match, image_a, image_b, result_html):
    """Render the comparison output in a modal dialog."""
    st.markdown(
        f"""
        <div style="
            margin:0 0 12px;
            color:#94a3b8;
            font-size:13px;
            font-weight:700;
        ">
            Reference: <span style="color:#f8fafc;">{reference_name}</span>
            &nbsp; • &nbsp;
            Similarity: <span style="color:#f8fafc;">{similarity:.4f}</span>
            &nbsp; • &nbsp;
            Threshold: <span style="color:#f8fafc;">{threshold:.2f}</span>
            &nbsp; • &nbsp;
            Decision: <span style="color:{'#22c55e' if is_match else '#ef4444'};">{"MATCH" if is_match else "NOT MATCH"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    out_a, out_b = st.columns(2)
    with out_a:
        st.image(image_a, caption=f"Detected Face A: {reference_name}", use_container_width=True)
    with out_b:
        st.image(image_b, caption="Detected Face B Result", use_container_width=True)

    st.markdown(result_html, unsafe_allow_html=True)


def reset_captured_faces():
    """Clear captured face data and input widgets."""
    for key in [
        "face_a_data",
        "face_b_data",
        "face_a_upload",
        "face_a_camera",
        "face_b_upload",
        "face_b_camera",
        "face_a_name",
    ]:
        st.session_state.pop(key, None)


# ==========================================
# STREAMLIT INTERFACE
# ==========================================

inject_custom_css()

st.markdown(
    """
    <section class="app-header">
        <div class="eyebrow">AI face verification</div>
        <h1 class="app-title">Identity Match Studio</h1>
        <p class="app-subtitle">
            Capture or upload a reference face, compare a second face against it,
            and review the cosine similarity decision with the same ArcFace pipeline.
        </p>
        <div class="model-pills">
            <span class="model-pill">Haar Cascade detection</span>
            <span class="model-pill">ArcFace ResNet100</span>
            <span class="model-pill">OpenVINO CPU inference</span>
            <span class="model-pill">Threshold 0.70</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Image Sources</div>', unsafe_allow_html=True)

with st.container(key="reference_name_panel"):
    raw_face_a_name = st.text_input(
        "Reference identity name",
        value=DEFAULT_REFERENCE_NAME,
        placeholder=DEFAULT_REFERENCE_NAME,
        key="face_a_name",
    )

face_a_name = raw_face_a_name.strip() or DEFAULT_REFERENCE_NAME

col_a, col_b = st.columns([1, 1], gap="medium")
with col_a:
    with st.container(key="face_a_card"):
        st.markdown(
            """
            <span class="input-card-marker"></span>
            <div class="input-card-header">
                <div class="avatar-token">A</div>
                <div class="input-card-heading">
                    <div class="input-card-title">Face A</div>
                    <div class="input-card-caption">Reference identity</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mode_a = st.radio(
            "Face A input method",
            ["Upload Image", "Scan with Camera"],
            horizontal=True,
            key="face_a_mode",
        )

        if mode_a == "Upload Image":
            input_a = st.file_uploader(
                "Image A reference",
                type=["jpg", "jpeg", "png"],
                key="face_a_upload",
            )
            process_uploaded_image_input("face_a_data", input_a, f"Face A: {face_a_name}")
        else:
            render_webrtc_camera_capture(
                "face_a_data",
                "face_a_camera",
                f"Face A: {face_a_name}",
                "Capture Face A",
            )

        face_a_data = st.session_state.get("face_a_data")
        if face_a_data:
            face_a_data["detected_rgb"] = render_detected_face(face_a_data, f"Face A: {face_a_name}")
            st.image(
                face_a_data["detected_rgb"],
                caption=f"Captured Face A: {face_a_name}",
                use_container_width=True,
            )

with col_b:
    with st.container(key="face_b_card"):
        st.markdown(
            """
            <span class="input-card-marker"></span>
            <div class="input-card-header">
                <div class="avatar-token">B</div>
                <div class="input-card-heading">
                    <div class="input-card-title">Face B</div>
                    <div class="input-card-caption">Comparison identity</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mode_b = st.radio(
            "Face B input method",
            ["Upload Image", "Scan with Camera"],
            horizontal=True,
            key="face_b_mode",
        )

        if mode_b == "Upload Image":
            input_b = st.file_uploader(
                "Image B comparison",
                type=["jpg", "jpeg", "png"],
                key="face_b_upload",
            )
            process_uploaded_image_input("face_b_data", input_b, "Face B")
        else:
            render_webrtc_camera_capture(
                "face_b_data",
                "face_b_camera",
                "Face B",
                "Capture Face B",
            )

        face_b_data = st.session_state.get("face_b_data")
        if face_b_data:
            face_b_data["detected_rgb"] = render_detected_face(face_b_data, "Face B")
            st.image(
                face_b_data["detected_rgb"],
                caption="Captured Face B",
                use_container_width=True,
            )

face_a_data = st.session_state.get("face_a_data")
face_b_data = st.session_state.get("face_b_data")
can_compare = face_a_data is not None and face_b_data is not None

with st.container(key="action_row"):
    compare_col, reset_col = st.columns([4, 1], gap="small", vertical_alignment="center")
    with compare_col:
        compare_clicked = st.button(
            "Compare Faces",
            type="primary",
            key="compare_button",
            disabled=not can_compare,
            use_container_width=True,
        )
    with reset_col:
        if st.button("Reset", key="reset_button", use_container_width=True):
            reset_captured_faces()
            st.rerun()

if compare_clicked:
    detected_a, detected_b, result_html, error, similarity, is_match = verify_stored_faces(
        face_a_data,
        face_b_data,
        face_a_name,
    )

    if error:
        st.error(error)
    else:
        show_result_dialog(
            similarity,
            THRESHOLD,
            face_a_name,
            is_match,
            detected_a,
            detected_b,
            result_html,
        )
elif not can_compare:
    st.markdown(
        """
        <div class="custom-alert">
            <div class="alert-icon">i</div>
            <div>Capture or upload both Face A and Face B to enable comparison.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
