import os
import cv2
import numpy as np
import urllib.request
import hashlib
import threading
import base64
import json
import textwrap
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import openvino as ov

SEO_TITLE = "Face Recognition"
SEO_DESCRIPTION = (
    "Compare two face images with Haar Cascade detection, ArcFace embeddings, "
    "and OpenVINO CPU inference in a Streamlit face verification app."
)
SEO_URL = "https://face-recognition-ai.streamlit.app/"
SEO_KEYWORDS = (
    "face recognition, face verification, ArcFace, OpenVINO, Haar Cascade, "
    "AI face comparison, Streamlit"
)

st.set_page_config(page_title=SEO_TITLE, page_icon=Path("face.png"), layout="wide")


def inject_seo_metadata():
    """Add client-side metadata for browsers and social previews."""
    components.html(
        f"""
        <script>
        (() => {{
            const doc = window.parent.document;
            const setMeta = (selector, attrs) => {{
                let element = doc.head.querySelector(selector);
                if (!element) {{
                    element = doc.createElement("meta");
                    doc.head.appendChild(element);
                }}
                Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
            }};
            const setLink = (selector, attrs) => {{
                let element = doc.head.querySelector(selector);
                if (!element) {{
                    element = doc.createElement("link");
                    doc.head.appendChild(element);
                }}
                Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
            }};

            doc.title = {json.dumps(SEO_TITLE)};
            setMeta('meta[name="description"]', {{
                name: "description",
                content: {json.dumps(SEO_DESCRIPTION)}
            }});
            setMeta('meta[name="keywords"]', {{
                name: "keywords",
                content: {json.dumps(SEO_KEYWORDS)}
            }});
            setMeta('meta[property="og:title"]', {{
                property: "og:title",
                content: {json.dumps(SEO_TITLE)}
            }});
            setMeta('meta[property="og:description"]', {{
                property: "og:description",
                content: {json.dumps(SEO_DESCRIPTION)}
            }});
            setMeta('meta[property="og:type"]', {{
                property: "og:type",
                content: "website"
            }});
            setMeta('meta[property="og:url"]', {{
                property: "og:url",
                content: {json.dumps(SEO_URL)}
            }});
            setMeta('meta[name="twitter:card"]', {{
                name: "twitter:card",
                content: "summary"
            }});
            setMeta('meta[name="twitter:title"]', {{
                name: "twitter:title",
                content: {json.dumps(SEO_TITLE)}
            }});
            setMeta('meta[name="twitter:description"]', {{
                name: "twitter:description",
                content: {json.dumps(SEO_DESCRIPTION)}
            }});
            setLink('link[rel="canonical"]', {{
                rel: "canonical",
                href: {json.dumps(SEO_URL)}
            }});
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


inject_seo_metadata()

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
    arcface_input = compiled_arcface.input(0)
    arcface_output = compiled_arcface.output(0)
    arcface_infer_lock = threading.Lock()

    return haar_cascade, compiled_arcface, arcface_input, arcface_output, arcface_infer_lock


# ==========================================
# FACE VERIFICATION CONSTANTS
# ==========================================

DEFAULT_REFERENCE_NAME = "Reference"
THRESHOLD = 0.70
CAMERA_FRAME_IS_MIRRORED = True

# ==========================================
# UI STYLING
# ==========================================

def get_theme_palette(theme_mode):
    """Return UI colors for the selected visual theme."""
    if theme_mode == "Light":
        return {
            "bg": "#f8fafc",
            "surface": "#ffffff",
            "surface_strong": "#f1f5f9",
            "surface_soft": "#f8fafc",
            "border": "rgba(15, 23, 42, 0.12)",
            "border_strong": "rgba(255, 145, 0, 0.52)",
            "text": "#0f172a",
            "muted": "#64748b",
            "accent": "#FF9100",
            "accent_strong": "#FFB14A",
            "accent_soft": "rgba(255, 145, 0, 0.14)",
            "accent_border": "rgba(255, 145, 0, 0.40)",
            "danger": "#ef4444",
            "avatar_text": "#9a3412",
            "panel_bg": "#ffffff",
            "card_bg": "rgba(255, 255, 255, 0.98)",
            "pill_bg": "#fff7ed",
            "label_text": "#1e293b",
            "field_bg": "#ffffff",
            "radio_bg": "#f1f5f9",
            "uploader_bg": "#f8fafc",
            "uploader_border": "rgba(100, 116, 139, 0.36)",
            "uploader_button_bg": "#ffffff",
            "uploader_button_hover_bg": "#fff7ed",
            "uploader_button_text": "#0f172a",
            "uploader_button_border": "rgba(100, 116, 139, 0.28)",
            "disabled_bg": "#e2e8f0",
            "disabled_text": "#64748b",
            "reset_text": "#0f172a",
            "alert_bg": "#fff7ed",
            "alert_text": "#7c2d12",
            "alert_border": "rgba(255, 145, 0, 0.28)",
            "alert_icon_bg": "rgba(255, 145, 0, 0.14)",
            "shadow": "0 22px 55px rgba(15, 23, 42, 0.10)",
            "soft_shadow": "0 14px 34px rgba(15, 23, 42, 0.08)",
            "card_shadow": "0 18px 40px rgba(15, 23, 42, 0.10)",
            "hover_shadow": "0 24px 55px rgba(15, 23, 42, 0.14)",
            "image_shadow": "0 16px 34px rgba(15, 23, 42, 0.12)",
            "button_shadow": "0 16px 34px rgba(255, 145, 0, 0.24)",
            "result_bg": "linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98))",
            "result_panel_bg": "#ffffff",
            "progress_bg": "#e2e8f0",
            "divider": "#e2e8f0",
        }

    return {
        "bg": "#000000",
        "surface": "#0b0b0b",
        "surface_strong": "#151515",
        "surface_soft": "#050505",
        "border": "rgba(148, 163, 184, 0.18)",
        "border_strong": "rgba(255, 145, 0, 0.42)",
        "text": "#f8fafc",
        "muted": "#94a3b8",
        "accent": "#FF9100",
        "accent_strong": "#FFB14A",
        "accent_soft": "rgba(255, 145, 0, 0.16)",
        "accent_border": "rgba(255, 145, 0, 0.38)",
        "danger": "#ef4444",
        "avatar_text": "#FFE0B3",
        "panel_bg": "rgba(11, 11, 11, 0.92)",
        "card_bg": "rgba(11, 11, 11, 0.96)",
        "pill_bg": "rgba(5, 5, 5, 0.82)",
        "label_text": "#f3f4f6",
        "field_bg": "rgba(5, 5, 5, 0.90)",
        "radio_bg": "rgba(5, 5, 5, 0.72)",
        "uploader_bg": "rgba(8, 8, 8, 0.82)",
        "uploader_border": "rgba(148, 163, 184, 0.42)",
        "uploader_button_bg": "#111827",
        "uploader_button_hover_bg": "#1f2937",
        "uploader_button_text": "#f8fafc",
        "uploader_button_border": "rgba(148, 163, 184, 0.28)",
        "disabled_bg": "#2a2a2a",
        "disabled_text": "#94a3b8",
        "reset_text": "#f3f4f6",
        "alert_bg": "rgba(255, 145, 0, 0.10)",
        "alert_text": "#ffe0b3",
        "alert_border": "rgba(255, 145, 0, 0.24)",
        "alert_icon_bg": "rgba(255, 145, 0, 0.16)",
        "shadow": "0 22px 55px rgba(0, 0, 0, 0.36)",
        "soft_shadow": "0 14px 34px rgba(0, 0, 0, 0.20)",
        "card_shadow": "0 18px 40px rgba(0, 0, 0, 0.26)",
        "hover_shadow": "0 24px 55px rgba(0, 0, 0, 0.34)",
        "image_shadow": "0 16px 34px rgba(0, 0, 0, 0.28)",
        "button_shadow": "0 16px 34px rgba(255, 145, 0, 0.34)",
        "result_bg": "linear-gradient(180deg, rgba(15,15,15,0.98), rgba(5,5,5,0.98))",
        "result_panel_bg": "#050505",
        "progress_bg": "#2a2a2a",
        "divider": "#374151",
    }


def build_theme_variables(theme_mode):
    """Build CSS variables for the selected theme."""
    palette = get_theme_palette(theme_mode)
    css_names = {
        "bg": "bg",
        "surface": "surface",
        "surface_strong": "surface-strong",
        "surface_soft": "surface-soft",
        "border": "border",
        "border_strong": "border-strong",
        "text": "text",
        "muted": "muted",
        "accent": "accent",
        "accent_strong": "accent-strong",
        "accent_soft": "accent-soft",
        "accent_border": "accent-border",
        "danger": "danger",
        "avatar_text": "avatar-text",
        "panel_bg": "panel-bg",
        "card_bg": "card-bg",
        "pill_bg": "pill-bg",
        "label_text": "label-text",
        "field_bg": "field-bg",
        "radio_bg": "radio-bg",
        "uploader_bg": "uploader-bg",
        "uploader_border": "uploader-border",
        "uploader_button_bg": "uploader-button-bg",
        "uploader_button_hover_bg": "uploader-button-hover-bg",
        "uploader_button_text": "uploader-button-text",
        "uploader_button_border": "uploader-button-border",
        "disabled_bg": "disabled-bg",
        "disabled_text": "disabled-text",
        "reset_text": "reset-text",
        "alert_bg": "alert-bg",
        "alert_text": "alert-text",
        "alert_border": "alert-border",
        "alert_icon_bg": "alert-icon-bg",
        "shadow": "shadow",
        "soft_shadow": "soft-shadow",
        "card_shadow": "card-shadow",
        "hover_shadow": "hover-shadow",
        "image_shadow": "image-shadow",
        "button_shadow": "button-shadow",
    }
    lines = [f"                --{css_name}: {palette[key]};" for key, css_name in css_names.items()]
    lines.extend(
        [
            "                --info: var(--accent);",
            "                --radius: 20px;",
        ]
    )
    return "\n".join(lines)


def inject_custom_css(theme_mode):
    """Apply a polished visual layer over Streamlit widgets."""
    theme_variables = build_theme_variables(theme_mode)
    css = """
        <style>
            :root {
__THEME_VARIABLES__
            }

            *,
            *::before,
            *::after {
                box-sizing: border-box;
            }

            html,
            body,
            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] > .main {
                max-width: 100%;
                overflow-x: hidden;
            }

            .stApp {
                background: var(--bg);
                color: var(--text);
            }

            [data-testid="stHeader"] {
                background: var(--bg);
            }

            [role="dialog"],
            [aria-modal="true"],
            [data-testid="stDialog"],
            div[data-baseweb="modal"] [role="dialog"] {
                background: var(--panel-bg) !important;
                color: var(--text) !important;
                position: relative;
                z-index: 1000000 !important;
            }

            div[data-baseweb="modal"] > div:first-child {
                background: rgba(0, 0, 0, 0.1) !important;
                backdrop-filter: none !important;
                -webkit-backdrop-filter: none !important;
                z-index: 999999 !important;
            }

            [role="dialog"] h1,
            [role="dialog"] h2,
            [role="dialog"] h3,
            [role="dialog"] p,
            [role="dialog"] span,
            [aria-modal="true"] h1,
            [aria-modal="true"] h2,
            [aria-modal="true"] h3,
            [aria-modal="true"] p,
            [aria-modal="true"] span {
                color: inherit;
            }

            [role="dialog"] button,
            [aria-modal="true"] button {
                color: var(--text) !important;
            }

            [data-testid="stAppViewContainer"] > .main .block-container {
                width: 100%;
                max-width: 1180px;
                padding: 36px 28px 64px;
                overflow-x: hidden;
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
                width: 100%;
                max-width: 100%;
                overflow: hidden;
                margin-bottom: 28px;
                padding: 30px;
                border: 1px solid var(--border);
                border-radius: 28px;
                background: var(--panel-bg);
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
                overflow-wrap: break-word;
            }

            .app-subtitle {
                max-width: 760px;
                margin: 16px 0 22px;
                color: var(--label-text);
                font-size: 16px;
                line-height: 1.7;
                overflow-wrap: break-word;
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
                background: var(--pill-bg);
                color: var(--label-text);
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
                width: 100%;
                max-width: 100%;
                overflow: hidden;
                margin: 4px 0 12px;
                padding: 18px 20px;
                border: 1px solid var(--border);
                border-radius: 18px;
                background: var(--panel-bg);
                box-shadow: var(--soft-shadow);
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
                max-width: 100% !important;
                overflow: hidden;
                height: 100%;
                padding: 24px;
                border: 1px solid var(--border);
                border-radius: var(--radius);
                background: var(--card-bg);
                box-shadow: var(--card-shadow);
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                transition: border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
            }

            .st-key-face_a_card:hover,
            .st-key-face_b_card:hover {
                border-color: var(--border-strong);
                transform: translateY(-1px);
                box-shadow: var(--hover-shadow);
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
                border: 1px solid var(--accent-border);
                color: var(--avatar-text);
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
                color: var(--label-text) !important;
                font-weight: 720;
            }

            [data-testid="stTextInput"] input {
                border: 1px solid var(--border);
                border-radius: 14px;
                background: var(--field-bg);
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
                background: var(--radio-bg);
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

            div[role="radiogroup"] label:not(:has(input:checked)) {
                color: var(--muted) !important;
            }

            div[role="radiogroup"] label:not(:has(input:checked)) p,
            div[role="radiogroup"] label:not(:has(input:checked)) [data-testid="stMarkdownContainer"] p {
                color: var(--muted) !important;
                opacity: 1 !important;
            }

            div[role="radiogroup"] label:has(input:checked) p,
            div[role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p {
                color: white !important;
                opacity: 1 !important;
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
                border: 1px dashed var(--uploader-border);
                background: var(--uploader-bg);
                box-sizing: border-box;
                transition: border-color 150ms ease, background 150ms ease;
            }

            [data-testid="stFileUploader"] section > div {
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                flex-wrap: wrap !important;
            }

            [data-testid="stFileUploader"] section p {
                max-width: 100%;
                overflow-wrap: break-word;
            }

            [data-testid="stFileUploader"] button {
                border: 1px solid var(--uploader-button-border) !important;
                border-radius: 12px !important;
                background: var(--uploader-button-bg) !important;
                color: var(--uploader-button-text) !important;
                box-shadow: none !important;
            }

            [data-testid="stFileUploader"] button *,
            [data-testid="stFileUploader"] button svg {
                color: var(--uploader-button-text) !important;
                fill: currentColor !important;
                stroke: currentColor !important;
            }

            [data-testid="stFileUploader"] button:hover {
                border-color: var(--accent-border) !important;
                background: var(--uploader-button-hover-bg) !important;
                color: var(--uploader-button-text) !important;
            }

            [data-testid="stFileUploader"] section:hover {
                border-color: var(--accent-strong);
                background: rgba(255, 145, 0, 0.10);
            }

            [data-testid="stCameraInput"] video,
            [data-testid="stCameraInput"] img {
                transform: scaleX(-1);
                transform-origin: center;
            }

            [data-testid="stCameraInput"] button {
                width: 100% !important;
                min-height: 42px !important;
                border: 1px solid var(--border) !important;
                border-radius: 0 0 10px 10px !important;
                background: var(--field-bg) !important;
                color: var(--text) !important;
                box-shadow: none !important;
                opacity: 1 !important;
            }

            [data-testid="stCameraInput"] button *,
            [data-testid="stCameraInput"] button p,
            [data-testid="stCameraInput"] button span {
                color: var(--text) !important;
                opacity: 1 !important;
                visibility: visible !important;
            }

            [data-testid="stCameraInput"] button:hover {
                border-color: var(--accent-strong) !important;
                background: var(--accent-soft) !important;
                color: var(--text) !important;
            }

            [data-testid="stCameraInput"] button:hover *,
            [data-testid="stCameraInput"] button:hover p,
            [data-testid="stCameraInput"] button:hover span {
                color: var(--text) !important;
            }

            [data-testid="stImage"] img {
                border-radius: 18px;
                border: 1px solid var(--border);
                box-shadow: var(--image-shadow);
            }

            [data-testid="stImageCaption"] {
                color: var(--muted);
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
                background: var(--disabled-bg) !important;
                color: var(--disabled-text) !important;
                box-shadow: none !important;
                opacity: 1 !important;
            }

            .stButton > button:disabled *,
            button:disabled * {
                color: var(--disabled-text) !important;
            }

            .st-key-compare_button button,
            div[data-testid="stButton"] button[kind="primary"] {
                width: 100%;
                border: 0 !important;
                background: linear-gradient(135deg, var(--accent), var(--accent-strong)) !important;
                color: white !important;
                box-shadow: var(--button-shadow);
            }

            .st-key-compare_button button:disabled,
            div[data-testid="stButton"] button[kind="primary"]:disabled {
                background: var(--disabled-bg) !important;
                color: var(--disabled-text) !important;
                box-shadow: none !important;
            }

            .st-key-reset_button button {
                border: 1px solid rgba(148, 163, 184, 0.34) !important;
                background: transparent !important;
                color: var(--reset-text) !important;
            }

            .st-key-reset_button button:hover {
                border-color: var(--accent-strong) !important;
                color: var(--text) !important;
                background: rgba(255, 145, 0, 0.12) !important;
            }

            .st-key-theme_panel {
                width: min(280px, 100%);
                margin: 0 0 18px auto;
            }

            .st-key-theme_panel div[role="radiogroup"] {
                border-radius: 999px;
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
                border: 1px solid var(--alert-border);
                border-radius: 16px;
                background: var(--alert-bg);
                color: var(--alert-text);
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
                background: var(--alert-icon-bg);
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

                .app-title {
                    font-size: 34px;
                    line-height: 1.08;
                }

                .app-subtitle {
                    font-size: 15px;
                    line-height: 1.55;
                }
            }

            @media (max-width: 900px) {
                div[data-testid="stHorizontalBlock"]:has(.st-key-face_a_card):has(.st-key-face_b_card) {
                    grid-template-columns: 1fr !important;
                    gap: 16px !important;
                }

                div[data-testid="stHorizontalBlock"]:has(.st-key-face_a_card):has(.st-key-face_b_card) > div[data-testid="stColumn"] {
                    width: 100% !important;
                    min-width: 0 !important;
                    max-width: 100% !important;
                }

                .st-key-face_a_card,
                .st-key-face_b_card {
                    padding: 18px;
                    border-radius: 18px;
                    height: auto;
                }

                .st-key-face_a_card:hover,
                .st-key-face_b_card:hover {
                    transform: none;
                }

                .input-card-header {
                    grid-template-columns: 48px minmax(0, 1fr);
                    column-gap: 14px;
                    min-height: 0;
                    margin-bottom: 16px;
                    align-items: center;
                }

                .avatar-token {
                    width: 48px;
                    height: 48px;
                    border-radius: 14px;
                    font-size: 16px;
                }

                .input-card-title {
                    font-size: 28px;
                    line-height: 1.12;
                    overflow-wrap: normal;
                    word-break: normal;
                }

                .input-card-caption {
                    margin-top: 4px !important;
                    font-size: 14px;
                }

                .st-key-action_row [data-testid="stHorizontalBlock"]:has(.st-key-compare_button):has(.st-key-reset_button) {
                    flex-direction: column !important;
                    align-items: stretch !important;
                    gap: 10px !important;
                }

                .st-key-action_row [data-testid="stHorizontalBlock"]:has(.st-key-compare_button):has(.st-key-reset_button) > [data-testid="stColumn"]:first-child,
                .st-key-action_row [data-testid="stHorizontalBlock"]:has(.st-key-compare_button):has(.st-key-reset_button) > [data-testid="stColumn"]:last-child {
                    flex: 0 0 auto !important;
                    width: 100% !important;
                    min-width: 0 !important;
                    max-width: 100% !important;
                }
            }

            @media (max-width: 520px) {
                [data-testid="stAppViewContainer"] > .main .block-container {
                    padding: 18px 12px 40px;
                }

                .app-header {
                    margin-bottom: 20px;
                    padding: 18px;
                    border-radius: 18px;
                }

                .app-title {
                    font-size: 30px;
                }

                .model-pills {
                    gap: 8px;
                }

                .model-pill {
                    padding: 7px 10px;
                    font-size: 12px;
                }

                .section-title {
                    margin-top: 18px;
                    font-size: 20px;
                }

                .st-key-reference_name_panel,
                .st-key-face_a_card,
                .st-key-face_b_card {
                    padding: 16px;
                    border-radius: 16px;
                }

                .input-card-header {
                    grid-template-columns: 42px minmax(0, 1fr);
                    column-gap: 12px;
                    margin-bottom: 14px;
                }

                .avatar-token {
                    width: 42px;
                    height: 42px;
                    border-radius: 12px;
                    font-size: 15px;
                }

                .input-card-title {
                    font-size: 24px;
                }

                .input-card-caption {
                    font-size: 13px;
                }

                div[role="radiogroup"] {
                    grid-template-columns: 1fr;
                    border-radius: 18px;
                }

                div[role="radiogroup"] label {
                    height: 40px !important;
                    min-height: 40px;
                    padding: 0 10px;
                }

                div[role="radiogroup"] label p {
                    white-space: normal;
                }

                [data-testid="stFileUploader"] section {
                    min-height: 92px;
                }
            }
        </style>
        """
    st.markdown(css.replace("__THEME_VARIABLES__", theme_variables), unsafe_allow_html=True)


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

    with arcface_infer_lock:
        infer_request = compiled_arcface.create_infer_request()
        result = infer_request.infer({arcface_input: face_tensor})

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

    try:
        embedding = get_face_embedding(face)
    except RuntimeError:
        return None, "Face embedding failed. Try another photo or restart the app if this keeps happening."

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


def process_camera_snapshot_input(slot_key, image_file, stored_label, widget_key):
    """Decode and store a still image captured from Streamlit's browser camera."""
    if image_file is None:
        return

    image_rgb, decode_error, signature = decode_image_file(image_file)
    if decode_error:
        st.session_state[slot_key] = None
        st.error(decode_error)
        return

    if CAMERA_FRAME_IS_MIRRORED:
        image_rgb = cv2.flip(image_rgb, 1)

    corrected_signature = build_image_signature(f"{widget_key}:snapshot-corrected:{signature}", image_rgb)
    store_processed_face_image(slot_key, image_rgb, stored_label, corrected_signature)


def get_input_reset_nonce():
    """Return the current input version used to refresh stateful media widgets."""
    return st.session_state.setdefault("input_reset_nonce", 0)


def build_resettable_widget_key(base_key):
    """Build a widget key that changes whenever the user resets the form."""
    return f"{base_key}_{get_input_reset_nonce()}"


def render_camera_snapshot_capture(slot_key, widget_key, stored_label, capture_label):
    """Render the default browser camera snapshot input and store the corrected frame."""
    snapshot = st.camera_input(
        capture_label.replace("Capture", "Take"),
        key=build_resettable_widget_key(f"{widget_key}_snapshot"),
    )
    process_camera_snapshot_input(slot_key, snapshot, stored_label, widget_key)


def image_rgb_to_data_uri(image_rgb):
    """Encode an RGB image array for inline modal display."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, encoded_image = cv2.imencode(".png", image_bgr)
    if not success:
        return ""

    image_base64 = base64.b64encode(encoded_image.tobytes()).decode("ascii")
    return f"data:image/png;base64,{image_base64}"


def render_html_fragment(html):
    """Render custom HTML without letting Markdown reinterpret nested markup."""
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def build_detected_faces_html(image_a, image_b, reference_name, theme_mode=None):
    """Build themed detected-face image frames for the result modal."""
    palette = get_theme_palette(theme_mode or st.session_state.get("app_theme", "Dark"))
    image_a_uri = image_rgb_to_data_uri(image_a)
    image_b_uri = image_rgb_to_data_uri(image_b)

    return textwrap.dedent(f"""
    <style>
        .result-face-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
            margin: 12px 0 18px;
        }}

        .result-face-frame {{
            position: relative;
            overflow: hidden;
            border-radius: 18px;
            border: 1px solid {palette["border"]};
            background: {palette["result_panel_bg"]};
            box-shadow: {palette["image_shadow"]};
        }}

        .result-face-frame img {{
            display: block;
            width: 100%;
            aspect-ratio: 16 / 9;
            object-fit: cover;
        }}

        .result-face-corner {{
            position: absolute;
            width: 30px;
            height: 30px;
            border-color: {palette["accent"]};
            pointer-events: none;
        }}

        .result-face-corner.top-left {{
            top: 12px;
            left: 12px;
            border-top: 2px solid;
            border-left: 2px solid;
        }}

        .result-face-corner.top-right {{
            top: 12px;
            right: 12px;
            border-top: 2px solid;
            border-right: 2px solid;
        }}

        .result-face-corner.bottom-left {{
            bottom: 12px;
            left: 12px;
            border-bottom: 2px solid;
            border-left: 2px solid;
        }}

        .result-face-corner.bottom-right {{
            right: 12px;
            bottom: 12px;
            border-right: 2px solid;
            border-bottom: 2px solid;
        }}

        .result-face-caption {{
            padding: 10px 12px 12px;
            color: {palette["muted"]};
            font-size: 12px;
            font-weight: 700;
            text-align: center;
        }}

        @media (max-width: 760px) {{
            .result-face-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
    <div class="result-face-grid">
        <figure class="result-face-frame">
            <img src="{image_a_uri}" alt="Detected Face A: {reference_name}">
            <span class="result-face-corner top-left"></span>
            <span class="result-face-corner top-right"></span>
            <span class="result-face-corner bottom-left"></span>
            <span class="result-face-corner bottom-right"></span>
            <figcaption class="result-face-caption">Detected Face A: {reference_name}</figcaption>
        </figure>
        <figure class="result-face-frame">
            <img src="{image_b_uri}" alt="Detected Face B Result">
            <span class="result-face-corner top-left"></span>
            <span class="result-face-corner top-right"></span>
            <span class="result-face-corner bottom-left"></span>
            <span class="result-face-corner bottom-right"></span>
            <figcaption class="result-face-caption">Detected Face B Result</figcaption>
        </figure>
    </div>
    """).strip()


def build_result_html(similarity, is_match, result_label, face_a_name, theme_mode=None):
    """Build the styled result card shown after verification"""
    active_theme = theme_mode or st.session_state.get("app_theme", "Dark")
    palette = get_theme_palette(active_theme)
    if active_theme == "Light":
        match_badge_bg = "#dcfce7"
        match_badge_text = "#166534"
        no_match_badge_bg = "#fee2e2"
        no_match_badge_text = "#991b1b"
    else:
        match_badge_bg = "#052e16"
        match_badge_text = "#86efac"
        no_match_badge_bg = "#450a0a"
        no_match_badge_text = "#fca5a5"

    if is_match:
        verdict = result_label
        accent = "#22c55e"
        badge_bg = match_badge_bg
        badge_text = match_badge_text
        status_icon = "MATCH"
        verdict_caption = "Face B matched the reference identity"
    else:
        verdict = "NOT MATCH"
        accent = "#ef4444"
        badge_bg = no_match_badge_bg
        badge_text = no_match_badge_text
        status_icon = "NO MATCH"
        verdict_caption = "Face B did not match the reference identity"

    display_score = max(0, min(100, similarity * 100))

    return textwrap.dedent(f"""
    <div style="
        max-width:850px;
        margin:28px auto 12px;
        font-family:Inter,Arial,sans-serif;
        color:{palette["text"]};
    ">
        <div style="
            background:{palette["result_bg"]};
            border:1px solid {palette["border"]};
            border-top:5px solid {accent};
            border-radius:28px;
            padding:34px;
            box-shadow:{palette["hover_shadow"]};
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
                <p style="margin:12px 0 0 0; color:{palette["muted"]}; font-size:15px;">
                    {verdict_caption}
                </p>
            </div>
            <div style="
                display:grid;
                grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));
                gap:14px;
                margin-bottom:18px;
            ">
                <div style="background:{palette["result_panel_bg"]}; border:1px solid {palette["border"]}; border-radius:18px; padding:18px; text-align:center;">
                    <div style="color:{palette["muted"]}; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:0.08em;">Similarity</div>
                    <div style="color:{accent}; font-size:30px; font-weight:900; margin-top:6px;">{similarity:.4f}</div>
                </div>
                <div style="background:{palette["result_panel_bg"]}; border:1px solid {palette["border"]}; border-radius:18px; padding:18px; text-align:center;">
                    <div style="color:{palette["muted"]}; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:0.08em;">Match Threshold</div>
                    <div style="color:{palette["text"]}; font-size:30px; font-weight:900; margin-top:6px;">{THRESHOLD:.2f}</div>
                </div>
                <div style="background:{palette["result_panel_bg"]}; border:1px solid {palette["border"]}; border-radius:18px; padding:18px; text-align:center;">
                    <div style="color:{palette["muted"]}; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:0.08em;">Reference Name</div>
                    <div style="color:{palette["text"]}; font-size:30px; font-weight:900; margin-top:6px;">{face_a_name}</div>
                </div>
            </div>
            <div style="
                width:100%;
                height:11px;
                border-radius:999px;
                background:{palette["progress_bg"]};
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
                background:{palette["result_panel_bg"]};
                border-left:4px solid {accent};
                border-radius:16px;
                padding:16px 18px;
            ">
                <div style="color:{palette["text"]}; font-weight:800; margin-bottom:7px;">Decision Analysis</div>
                <div style="color:{palette["muted"]}; font-size:13px; line-height:1.7;">
                    The cosine similarity score is
                    <strong style="color:{palette["text"]};">{similarity:.4f}</strong>.
                    This score is
                    <strong style="color:{accent};">{"above" if is_match else "below"}</strong>
                    the current verification threshold of
                    <strong style="color:{palette["text"]};">{THRESHOLD:.2f}</strong>.
                    Therefore, Face B is classified as
                    <strong style="color:{accent};">{verdict}</strong>.
                </div>
            </div>
            <div style="
                margin-top:16px;
                padding-top:14px;
                border-top:1px solid {palette["divider"]};
                color:{palette["muted"]};
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
    """).strip()


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
    palette = get_theme_palette(st.session_state.get("app_theme", "Dark"))
    decision_color = "#22c55e" if is_match else "#ef4444"
    render_html_fragment(
        textwrap.dedent(f"""
        <div style="
            margin:0 0 12px;
            color:{palette["muted"]};
            font-size:13px;
            font-weight:700;
        ">
            Reference: <span style="color:{palette["text"]};">{reference_name}</span>
            &nbsp; • &nbsp;
            Similarity: <span style="color:{palette["text"]};">{similarity:.4f}</span>
            &nbsp; • &nbsp;
            Threshold: <span style="color:{palette["text"]};">{threshold:.2f}</span>
            &nbsp; • &nbsp;
            Decision: <span style="color:{decision_color};">{"MATCH" if is_match else "NOT MATCH"}</span>
        </div>
        """).strip()
    )

    out_a, out_b = st.columns(2)
    with out_a:
        st.image(image_a, caption=f"Detected Face A: {reference_name}", use_container_width=True)
    with out_b:
        st.image(image_b, caption="Detected Face B Result", use_container_width=True)

    render_html_fragment(result_html)


def reset_captured_faces():
    """Clear captured face data and input widgets."""
    for key in ["face_a_data", "face_b_data"]:
        st.session_state.pop(key, None)

    st.session_state["input_reset_nonce"] = get_input_reset_nonce() + 1


# ==========================================
# STREAMLIT INTERFACE
# ==========================================

st.session_state.setdefault("app_theme", "Dark")
inject_custom_css(st.session_state["app_theme"])

with st.container(key="theme_panel"):
    st.radio(
        "Theme",
        ["Dark", "Light"],
        horizontal=True,
        key="app_theme",
        label_visibility="collapsed",
    )

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

haar_cascade, compiled_arcface, arcface_input, arcface_output, arcface_infer_lock = load_models()

st.markdown('<div class="section-title">Image Sources</div>', unsafe_allow_html=True)

with st.container(key="reference_name_panel"):
    raw_face_a_name = st.text_input(
        "Reference identity name",
        value=DEFAULT_REFERENCE_NAME,
        placeholder=DEFAULT_REFERENCE_NAME,
        key=build_resettable_widget_key("face_a_name"),
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
                key=build_resettable_widget_key("face_a_upload"),
            )
            process_uploaded_image_input("face_a_data", input_a, f"Face A: {face_a_name}")
        else:
            render_camera_snapshot_capture(
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
                key=build_resettable_widget_key("face_b_upload"),
            )
            process_uploaded_image_input("face_b_data", input_b, "Face B")
        else:
            render_camera_snapshot_capture(
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
