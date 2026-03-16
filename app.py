import streamlit as st
import pytesseract
from PIL import Image, ImageDraw
import sys
import os
import pickle

sys.path.insert(0, '.')
from solution import DocFusionSolution

import platform
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(
    page_title="DocFusion",
    page_icon="D",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0F0F0F;
    }
    header[data-testid="stHeader"] {
        background-color: #0F0F0F;
    }
    .block-container {
        padding-top: 3rem;
        max-width: 740px;
    }

    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        padding-top: 0.5rem;
    }
    .app-subtitle {
        font-size: 12px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #888;
        margin: 0 0 6px;
    }
    .app-title {
        font-size: 32px;
        font-weight: 600;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .version-badge {
        background: #1A1A1A;
        border: 1px solid #444;
        border-radius: 100px;
        padding: 5px 14px;
        font-size: 11px;
        color: #999;
    }
    .app-desc {
        color: #777;
        font-size: 15px;
        margin-top: -4px;
        margin-bottom: 1.5rem;
    }

    [data-testid="stFileUploader"] {
        background: #181818;
        border: 1.5px dashed #444;
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploaderDropzone"] span {
        color: #AAA !important;
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploaderDropzone"] small {
        color: #777 !important;
    }
    [data-testid="stFileUploader"] button {
        background: #252525 !important;
        color: #CCC !important;
        border: 1px solid #444 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
    }
    [data-testid="stFileUploader"] svg {
        color: #666 !important;
        fill: #666 !important;
    }

    .field-card {
        background: #1A1A1A;
        border-radius: 0 8px 8px 0;
        padding: 16px;
        margin-bottom: 8px;
    }
    .field-card-vendor { border-left: 3px solid #5DCAA5; }
    .field-card-date { border-left: 3px solid #85B7EB; }
    .field-card-total { border-left: 3px solid #EF9F27; }
    .field-card-missing { border-left: 3px solid #E24B4A; }

    .field-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin: 0 0 6px;
    }
    .label-vendor { color: #5DCAA5; }
    .label-date { color: #85B7EB; }
    .label-total { color: #EF9F27; }
    .label-missing { color: #E24B4A; }

    .field-value {
        font-size: 15px;
        font-weight: 500;
        color: #E8E8E8;
        margin: 0;
        word-wrap: break-word;
    }
    .field-value-missing {
        font-size: 14px;
        color: #666;
        margin: 0;
    }

    .verdict-genuine {
        background: #0D2618;
        border: 1px solid #1D4D33;
        border-radius: 10px;
        padding: 16px;
        margin-top: 12px;
    }
    .verdict-suspicious {
        background: #2A0F0F;
        border: 1px solid #4D1A1A;
        border-radius: 10px;
        padding: 16px;
        margin-top: 12px;
    }
    .verdict-dot-genuine {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #5DCAA5;
        display: inline-block;
        margin-right: 8px;
    }
    .verdict-dot-suspicious {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #E24B4A;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    .verdict-label-genuine {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #5DCAA5;
        font-weight: 600;
        display: inline;
    }
    .verdict-label-suspicious {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #E24B4A;
        font-weight: 600;
        display: inline;
    }
    .verdict-text-genuine {
        font-size: 20px;
        font-weight: 600;
        color: #5DCAA5;
        margin: 8px 0 0;
    }
    .verdict-text-suspicious {
        font-size: 20px;
        font-weight: 600;
        color: #E24B4A;
        margin: 8px 0 0;
    }

    .analysis-card {
        background: #1A1A1A;
        border-radius: 8px;
        padding: 14px 16px;
        margin-top: 8px;
    }
    .analysis-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #888;
        font-weight: 600;
        margin: 0 0 6px;
    }
    .analysis-text {
        font-size: 13px;
        color: #AAA;
        margin: 0;
        line-height: 1.6;
    }

    [data-testid="stExpander"] {
        background: #141414;
        border: 1px solid #2A2A2A;
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary span {
        color: #888 !important;
        font-size: 13px;
    }
    .ocr-text {
        background: #111;
        border: 1px solid #252525;
        border-radius: 8px;
        padding: 14px 16px;
        font-family: monospace;
        font-size: 12px;
        color: #999;
        line-height: 1.7;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 400px;
        overflow-y: auto;
    }

    .receipt-suspicious {
        border: 2px solid #E24B4A;
        border-radius: 8px;
        overflow: hidden;
    }
    .receipt-genuine {
        border: 1px solid #333;
        border-radius: 8px;
        overflow: hidden;
    }

    .stDeployButton, footer, #MainMenu {
        display: none;
    }

    .stMarkdown p, .stText, span, label {
        color: #CCC;
    }

    .stSpinner > div {
        color: #999 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div>
        <p class="app-subtitle">Rihal CodeStacker 2026</p>
        <p class="app-title">DocFusion</p>
    </div>
    <div class="version-badge">v1.0</div>
</div>
<p class="app-desc">Upload a receipt to extract fields and check for tampering.</p>
""", unsafe_allow_html=True)

@st.cache_resource
def load_solution():
    sol = DocFusionSolution()
    model_path = './work_dir'
    if os.path.exists(os.path.join(model_path, 'config.json')):
        sol.model_dir = model_path
    anomaly_path = os.path.join(model_path, 'anomaly_model.pkl')
    if os.path.exists(anomaly_path):
        with open(anomaly_path, 'rb') as f:
            sol.anomaly_model = pickle.load(f)
    return sol

def generate_anomaly_summary(is_forged, vendor, date, total, text):
    """Generate a readable explanation of the anomaly verdict."""
    if is_forged == 0:
        parts = []
        if vendor and date and total:
            parts.append("All three key fields (vendor, date, total) were successfully extracted.")
        elif vendor and total:
            parts.append("Vendor and total were extracted successfully.")
        elif date and total:
            parts.append("Date and total were extracted, though the vendor name could not be identified.")
        if total:
            try:
                val = float(total)
                if val < 200:
                    parts.append(f"The total of RM {total} falls within the typical range for this type of receipt.")
                elif val < 1000:
                    parts.append(f"The total of RM {total} is on the higher end but still within a reasonable range.")
            except:
                pass
        parts.append("No signs of digital tampering were detected in the text patterns.")
        return " ".join(parts)
    else:
        parts = []
        missing = []
        if not vendor:
            missing.append("vendor")
        if not date:
            missing.append("date")
        if not total:
            missing.append("total")
        if missing:
            parts.append(f"The {', '.join(missing)} field{'s' if len(missing) > 1 else ''} could not be extracted, which may indicate removal or alteration.")
        if total:
            try:
                val = float(total)
                if val > 500:
                    parts.append(f"The total amount (RM {total}) is unusually high compared to typical receipts in the training data.")
                elif val < 1:
                    parts.append(f"The total amount (RM {total}) is suspiciously low.")
            except:
                pass
        if len(text.strip()) < 100:
            parts.append("The receipt contains very little readable text, which is unusual for a genuine document.")
        lines = text.strip().split('\n')
        if len(lines) < 5:
            parts.append("The document has very few lines of text.")
        if not parts:
            parts.append("The text patterns in this receipt are inconsistent with genuine documents in the training set.")
        return " ".join(parts)

sol = load_solution()

uploaded_file = st.file_uploader(
    "Drop receipt here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Analyzing..."):
        text = pytesseract.image_to_string(image)
        vendor = sol._extract_vendor(text)
        date = sol._extract_date(text)
        total = sol._extract_total(text)
        is_forged = sol._detect_anomaly(text, vendor, date, total)

    col_img, col_fields = st.columns([1, 1], gap="medium")

    with col_img:
        border_class = "receipt-suspicious" if is_forged == 1 else "receipt-genuine"
        st.markdown(f'<div class="{border_class}">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_fields:
        if vendor:
            st.markdown(f"""
            <div class="field-card field-card-vendor">
                <p class="field-label label-vendor">Vendor</p>
                <p class="field-value">{vendor}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="field-card field-card-missing">
                <p class="field-label label-missing">Vendor</p>
                <p class="field-value-missing">Not found</p>
            </div>""", unsafe_allow_html=True)

        if date:
            st.markdown(f"""
            <div class="field-card field-card-date">
                <p class="field-label label-date">Date</p>
                <p class="field-value">{date}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="field-card field-card-missing">
                <p class="field-label label-missing">Date</p>
                <p class="field-value-missing">Not found</p>
            </div>""", unsafe_allow_html=True)

        if total:
            st.markdown(f"""
            <div class="field-card field-card-total">
                <p class="field-label label-total">Total</p>
                <p class="field-value">RM {total}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="field-card field-card-missing">
                <p class="field-label label-missing">Total</p>
                <p class="field-value-missing">Not found</p>
            </div>""", unsafe_allow_html=True)

        if is_forged == 1:
            st.markdown("""
            <div class="verdict-suspicious">
                <div>
                    <div class="verdict-dot-suspicious"></div>
                    <p class="verdict-label-suspicious">Verdict</p>
                </div>
                <p class="verdict-text-suspicious">Suspicious</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="verdict-genuine">
                <div>
                    <div class="verdict-dot-genuine"></div>
                    <p class="verdict-label-genuine">Verdict</p>
                </div>
                <p class="verdict-text-genuine">Genuine</p>
            </div>""", unsafe_allow_html=True)

        summary = generate_anomaly_summary(is_forged, vendor, date, total, text)
        st.markdown(f"""
        <div class="analysis-card">
            <p class="analysis-label">Analysis</p>
            <p class="analysis-text">{summary}</p>
        </div>""", unsafe_allow_html=True)

    with st.expander("Raw OCR output"):
        st.markdown(f'<div class="ocr-text">{text}</div>', unsafe_allow_html=True)