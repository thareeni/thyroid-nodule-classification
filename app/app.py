"""
app/app.py
----------
Thyroid Nodule Classification — AI-Powered Ultrasound Image Analysis
Streamlit frontend with clean dark medical AI dashboard interface.

Usage:
    streamlit run app/app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

# ─────────────────────────────────────────────
# Page config — MUST be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Thyroid Nodule Classification | Medical AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Constants & Model Options (User-facing: 2 models)
# ─────────────────────────────────────────────
CLASS_NAMES = ["Benign", "Malignant"]
IMG_SIZE    = (224, 224)

MODEL_OPTIONS = {
    "Custom CNN   —  Trained from Scratch": "models/custom_cnn_thyroid_model.h5",
    "MobileNetV2  —  Transfer Learning":   "models/pretrained_mobilenet_thyroid_model.h5",
}

MODEL_DETAILS = {
    "Custom CNN   —  Trained from Scratch": {
        "desc": "Custom multi-block CNN trained from scratch on thyroid dataset. No external pretrained weights used.",
        "badges": [("From Scratch", "#00e896"), ("Custom CNN", "#ff4b6e"), ("4 Conv Blocks", "#00a8ff")]
    },
    "MobileNetV2  —  Transfer Learning": {
        "desc": "Pretrained on ImageNet and fine-tuned on thyroid ultrasound images. Lightweight and high inference speed.",
        "badges": [("Transfer Learning", "#00e5c3"), ("ImageNet", "#7c6af7"), ("MobileNetV2", "#00a8ff")]
    }
}

# ─────────────────────────────────────────────
# Design System CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-deep:       #050d1a;
    --bg-card:       #0a1628;
    --bg-card2:      #0d1e35;
    --border:        #1a3050;
    --border-glow:   #1e4a7a;
    --accent-blue:   #00a8ff;
    --accent-teal:   #00e5c3;
    --accent-violet: #7c6af7;
    --benign:        #00e896;
    --malignant:     #ff4b6e;
    --text-primary:  #e8f0fe;
    --text-secondary:#7a9bb8;
    --text-muted:    #3a5a7a;
}

* {
    box-sizing: border-box !important;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: var(--bg-deep) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
    overflow-x: hidden !important;
}

/* Main block spacing */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1280px !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #070f1e 0%, #0a1628 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span {
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.86rem !important;
}
[data-testid="stSidebar"] strong { color: var(--text-primary) !important; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"]  { display: none; }

::-webkit-scrollbar       { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 3px; }

h1,h2,h3,h4 { font-family: 'Sora', sans-serif !important; letter-spacing: -0.02em; }

/* Streamlit Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
}
[data-testid="stSelectbox"] label {
    color: var(--text-secondary) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.4rem !important;
}

/* Streamlit File Uploader — Scoped Styling (No text overlap / duplication) */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border-glow) !important;
    border-radius: 14px !important;
    background: var(--bg-card) !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    margin-top: 0.4rem !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-teal) !important;
    box-shadow: 0 0 22px rgba(0, 229, 195, 0.15) !important;
}

[data-testid="stFileUploader"] label {
    color: var(--text-secondary) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.4rem !important;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
    padding: 0.4rem !important;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #0d1e35, #102a4a) !important;
    border: 1px solid var(--border-glow) !important;
    color: var(--accent-teal) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.4rem 0.9rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"] button:hover {
    border-color: var(--accent-teal) !important;
    color: #ffffff !important;
    box-shadow: 0 0 12px rgba(0,229,195,0.2) !important;
}

/* Image preview styling */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid var(--border-glow) !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.45) !important;
    max-width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--accent-teal) !important; }

/* Responsive layout adjustments */
@media (max-width: 850px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}
</style>
"""

# ─────────────────────────────────────────────
# HTML Components
# ─────────────────────────────────────────────

HEADER_HTML = """
<style>
.app-header {
    position: relative;
    padding: 1rem 0 1.8rem;
    text-align: center;
    overflow: hidden;
}
.header-bg {
    position: absolute; inset: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 70% 60% at 20% 50%, rgba(0,168,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 70% at 80% 30%, rgba(0,229,195,0.07) 0%, transparent 60%);
}
.header-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(0,229,195,0.08);
    border: 1px solid rgba(0,229,195,0.25);
    border-radius: 99px;
    padding: 5px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #00e5c3;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.pulse-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #00e5c3;
    box-shadow: 0 0 8px #00e5c3;
    animation: pdot 2s infinite;
}
@keyframes pdot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.4; transform:scale(1.4); }
}
.header-title {
    font-family: 'Sora', sans-serif;
    font-size: clamp(2.1rem, 4.5vw, 3.2rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0 0 0.4rem;
    background: linear-gradient(135deg, #e8f0fe 0%, #00a8ff 50%, #00e5c3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.header-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 400;
    color: #7a9bb8;
    max-width: 580px;
    margin: 0 auto;
    line-height: 1.5;
}
.header-divider {
    width: 80px; height: 3px;
    background: linear-gradient(90deg, #00a8ff, #00e5c3);
    border-radius: 99px;
    margin: 1.2rem auto 0;
}
</style>
<div class="app-header">
    <div class="header-bg"></div>
    <div class="header-badge">
        <span class="pulse-dot"></span>
        Deep Learning Medical Assistant
    </div>
    <h1 class="header-title">Thyroid Nodule Classification</h1>
    <div class="header-subtitle">AI-Powered Ultrasound Image Analysis</div>
    <div class="header-divider"></div>
</div>
"""

SIDEBAR_HTML = """
<style>
.sb-head { display:flex; align-items:center; gap:12px; padding:0.8rem 0 0.5rem; }
.sb-icon {
    width:40px; height:40px; border-radius:10px;
    background:linear-gradient(135deg,#0a1e38,#0d2a50);
    border:1px solid #1e4a7a;
    display:flex; align-items:center; justify-content:center;
    font-size:1.3rem;
    box-shadow:0 0 20px rgba(0,168,255,0.15);
}
.sb-name {
    font-family:'Sora',sans-serif; font-size:1.15rem; font-weight:700;
    letter-spacing:-0.02em;
    background:linear-gradient(135deg,#e8f0fe,#00a8ff);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.sb-ver {
    font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#3a5a7a;
}
.sb-div {
    height:1px;
    background:linear-gradient(90deg,transparent,#1a3050,transparent);
    margin:0.8rem 0 1rem;
}
.sb-sec {
    font-family:'Sora',sans-serif; font-size:0.7rem; font-weight:600;
    letter-spacing:0.12em; text-transform:uppercase; color:#3a5a7a !important;
    display:block; margin-bottom:0.6rem;
}
.sg { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:1rem; }
.sc {
    background:linear-gradient(135deg,#0a1628,#0d1e35);
    border:1px solid #1a3050; border-radius:10px;
    padding:10px 10px; text-align:center;
}
.sv { font-family:'Sora',sans-serif; font-size:1.1rem; font-weight:700; color:#00a8ff; display:block; }
.sl { font-family:'DM Sans',sans-serif; font-size:0.68rem; color:#5a7a9a; display:block; margin-top:2px; }
.ir { display:flex; align-items:flex-start; gap:9px; padding:7px 0; border-bottom:1px solid #0a1628; }
.ir:last-child { border-bottom:none; }
.id { width:6px; height:6px; border-radius:50%; margin-top:6px; flex-shrink:0; }
.it { font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#6a8aa8; line-height:1.5; }
.wbox {
    background:rgba(255,75,110,0.05);
    border:1px solid rgba(255,75,110,0.18);
    border-radius:10px; padding:10px 12px; margin-top:1.2rem;
}
.wt { font-family:'Sora',sans-serif; font-size:0.7rem; font-weight:600;
      color:#ff4b6e; letter-spacing:0.06em; text-transform:uppercase;
      display:block; margin-bottom:4px; }
.ww { font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#b05060; line-height:1.5; }
</style>
<div class="sb-head">
    <div class="sb-icon">🔬</div>
    <div>
        <div class="sb-name">ThyroScan AI</div>
        <div class="sb-ver">v2.0 · Medical AI Suite</div>
    </div>
</div>
<div class="sb-div"></div>
<span class="sb-sec">Dataset Information</span>
<div class="sg">
    <div class="sc"><span class="sv">5,000</span><span class="sl">Ultrasounds</span></div>
    <div class="sc"><span class="sv">2</span><span class="sl">Classes</span></div>
    <div class="sc"><span class="sv">224px</span><span class="sl">Input Res</span></div>
    <div class="sc"><span class="sv">2</span><span class="sl">AI Models</span></div>
</div>
<div class="sb-div"></div>
<span class="sb-sec">Workflow</span>
<div>
    <div class="ir"><div class="id" style="background:#00a8ff"></div>
        <div class="it">Select deep learning model architecture</div></div>
    <div class="ir"><div class="id" style="background:#00e5c3"></div>
        <div class="it">Upload B-Mode ultrasound scan image</div></div>
    <div class="ir"><div class="id" style="background:#7c6af7"></div>
        <div class="it">Automated neural network classification</div></div>
    <div class="ir"><div class="id" style="background:#00e896"></div>
        <div class="it">Evaluate confidence score & probability</div></div>
</div>
<div class="wbox">
    <span class="wt">⚠ Clinical Disclaimer</span>
    <span class="ww">For research use only. This tool is intended to assist medical professionals and does not replace official clinical diagnosis.</span>
</div>
"""

def model_info_html(model_name):
    info = MODEL_DETAILS.get(model_name, MODEL_DETAILS["MobileNetV2  —  Transfer Learning"])
    badge_html = "".join([
        f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:0.65rem; padding:3px 9px; border-radius:6px; border:1px solid {c}44; background:{c}10; color:{c};">{label}</span>'
        for label, c in info["badges"]
    ])
    return f"""
<div style="background:var(--bg-card2); border:1px solid var(--border-glow); border-radius:10px; padding:10px 12px; margin-top:0.8rem; margin-bottom:1.2rem;">
    <div style="font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#7a9bb8; line-height:1.5;">
        {info["desc"]}
    </div>
    <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:8px;">
        {badge_html}
    </div>
</div>
"""

def img_meta_html(w, h):
    return f"""
<div style="display:flex; align-items:center; justify-content:space-between; margin-top:1.2rem; margin-bottom:0.6rem;">
    <span style="font-family:'Sora',sans-serif; font-size:0.82rem; font-weight:600; color:#7a9bb8; letter-spacing:0.04em; text-transform:uppercase;">Uploaded Image Preview</span>
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#00e5c3; background:rgba(0,229,195,0.08); border:1px solid rgba(0,229,195,0.25); padding:2px 8px; border-radius:6px;">{w} × {h} px</span>
</div>
"""

def result_card_html(label, conf, idx):
    mal   = (idx == 1)
    col   = "#ff4b6e" if mal else "#00e896"
    bg    = "rgba(255,75,110,0.06)"  if mal else "rgba(0,232,150,0.06)"
    bdr   = "rgba(255,75,110,0.28)"  if mal else "rgba(0,232,150,0.28)"
    glow  = "rgba(255,75,110,0.12)"  if mal else "rgba(0,232,150,0.12)"
    icon  = "⚠️" if mal else "✅"
    risk  = "HIGH RISK · MALIGNANT FEATURES" if mal else "LOW RISK · BENIGN FEATURES"
    sub   = "Immediate specialist evaluation recommended." if mal else "Standard monitoring and routine follow-up suggested."
    w     = int(conf * 100)
    return f"""
<style>
.rc {{
    background:{bg}; border:1px solid {bdr};
    border-radius:16px; padding:1.4rem 1.5rem;
    text-align:center; position:relative; overflow:hidden;
    box-shadow:0 0 40px {glow}; margin-bottom:1.2rem;
}}
.rc-icon {{ font-size:2.4rem; display:block; margin-bottom:.3rem; filter:drop-shadow(0 0 12px {col}); }}
.rc-risk {{
    font-family:'JetBrains Mono',monospace; font-size:0.65rem;
    font-weight:600; letter-spacing:0.12em; color:{col};
    background:rgba(5,13,26,0.6); border:1px solid {bdr};
    border-radius:99px; padding:4px 14px; display:inline-block; margin-bottom:.6rem;
}}
.rc-label {{
    font-family:'Sora',sans-serif; font-size:2rem; font-weight:800;
    color:{col}; letter-spacing:-0.03em; display:block; margin-bottom:.2rem;
}}
.rc-sub {{
    font-family:'DM Sans',sans-serif; font-size:0.85rem; color:#7a9bb8;
}}
.cf-wrap {{ margin-top:1.2rem; text-align:left; }}
.cf-row {{
    display:flex; justify-content:space-between;
    align-items:center; margin-bottom:6px;
}}
.cf-lbl {{ font-family:'Sora',sans-serif; font-size:0.75rem;
    color:#7a9bb8; letter-spacing:.05em; text-transform:uppercase; font-weight:600; }}
.cf-pct {{ font-family:'JetBrains Mono',monospace; font-size:0.95rem;
    font-weight:700; color:{col}; }}
.cf-track {{ width:100%; height:8px; background:rgba(255,255,255,.05);
    border-radius:99px; overflow:hidden; }}
.cf-fill {{ height:100%; width:{w}%;
    background:linear-gradient(90deg,{col}88,{col});
    border-radius:99px; box-shadow:0 0 12px {col}55; }}
</style>
<div class="rc">
    <span class="rc-icon">{icon}</span>
    <span class="rc-risk">{risk}</span>
    <span class="rc-label">{label} Nodule</span>
    <span class="rc-sub">{sub}</span>
    <div class="cf-wrap">
        <div class="cf-row">
            <span class="cf-lbl">Classification Confidence</span>
            <span class="cf-pct">{conf*100:.1f}%</span>
        </div>
        <div class="cf-track"><div class="cf-fill"></div></div>
    </div>
</div>"""

def prob_bars_html(probs):
    colors = ["#00e896", "#ff4b6e"]
    icons  = ["✅", "⚠️"]
    out = """<div style="background:#0a1628; border:1px solid #1a3050; border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:1.2rem;">
<div style="font-family:'Sora',sans-serif; font-size:0.78rem; font-weight:700;
color:#7a9bb8; letter-spacing:.08em; text-transform:uppercase; margin-bottom:1rem;">
Class Probabilities Breakdown</div>"""
    for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs)):
        c = colors[i]
        w = int(p * 100)
        out += f"""
<div style="margin-bottom:14px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <span style="font-family:'DM Sans',sans-serif; font-size:0.88rem; color:#e8f0fe; font-weight:500;">
            {icons[i]} {cls} Nodule</span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.85rem;
            font-weight:600; color:{c};">{p*100:.1f}%</span>
    </div>
    <div style="width:100%; height:7px; background:rgba(255,255,255,.05); border-radius:99px; overflow:hidden;">
        <div style="height:100%; width:{w}%;
            background:linear-gradient(90deg,{c}77,{c});
            border-radius:99px; box-shadow:0 0 8px {c}44;"></div>
    </div>
</div>"""
    out += "</div>"
    return out

INTERP_HTML = """
<style>
.ibox {
    background:#0a1628;
    border:1px solid #1a3050;
    border-radius:14px; padding:1.2rem 1.4rem;
}
.ititle {
    font-family:'Sora',sans-serif; font-size:0.78rem; font-weight:700;
    color:#00a8ff; letter-spacing:.08em; text-transform:uppercase;
    margin-bottom:.8rem; display:flex; align-items:center; gap:8px;
}
.irow { display:flex; align-items:flex-start; gap:10px; padding:7px 0;
    border-bottom:1px solid rgba(26,48,80,0.5); }
.irow:last-child { border-bottom:none; }
.idot { width:7px; height:7px; border-radius:50%; margin-top:6px; flex-shrink:0; }
.itxt { font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#7a9bb8; line-height:1.55; }
.itxt strong { color:#e8f0fe; }
</style>
<div class="ibox">
    <div class="ititle">ℹ Clinical Interpretation Guide</div>
    <div class="irow">
        <div class="idot" style="background:#00e896; box-shadow:0 0 6px #00e89666;"></div>
        <div class="itxt"><strong>Benign</strong> — Smooth margins, well-defined boundaries. Standard follow-up imaging recommended.</div>
    </div>
    <div class="irow">
        <div class="idot" style="background:#ff4b6e; box-shadow:0 0 6px #ff4b6e66;"></div>
        <div class="itxt"><strong>Malignant</strong> — Irregular borders, microcalcifications, or hypoechoic texture. Expert biopsy assessment advised.</div>
    </div>
    <div class="irow">
        <div class="idot" style="background:#7c6af7; box-shadow:0 0 6px #7c6af766;"></div>
        <div class="itxt"><strong>Confidence Thresholds</strong> — Confidence >90% indicates high certainty. Scores below 75% warrant secondary validation.</div>
    </div>
</div>
"""

EMPTY_STATE_RIGHT_HTML = """
<div style="background:#0a1628; border:1px dashed #1a3050; border-radius:16px; padding:3rem 2rem; text-align:center; height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center;">
    <div style="font-size:3.5rem; margin-bottom:1rem; opacity:0.4;">🔬</div>
    <div style="font-family:'Sora',sans-serif; font-size:1.1rem; font-weight:700; color:#e8f0fe; margin-bottom:0.5rem;">
        Awaiting Image Upload
    </div>
    <div style="font-family:'DM Sans',sans-serif; font-size:0.88rem; color:#7a9bb8; max-width:320px; line-height:1.6; margin-bottom:1.2rem;">
        Upload a thyroid ultrasound scan on the left panel to execute real-time deep learning inference.
    </div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#00e5c3; background:rgba(0,229,195,0.08); border:1px solid rgba(0,229,195,0.2); padding:5px 14px; border-radius:99px;">
        Supported: JPG · JPEG · PNG
    </div>
</div>
"""

NOT_FOUND_HTML = lambda p: f"""
<div style="background:rgba(255,75,110,0.06); border:1px solid rgba(255,75,110,0.25);
border-radius:14px; padding:1.2rem 1.4rem; margin:1rem 0;">
    <div style="font-family:'Sora',sans-serif; font-weight:700; color:#ff4b6e;
    font-size:0.9rem; margin-bottom:0.4rem;">⚠ Model File Not Found</div>
    <div style="font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#b05060; line-height:1.6;">
        Model path <code style="background:rgba(0,0,0,.3); padding:2px 8px; border-radius:5px;
        color:#ff7a92; font-family:'JetBrains Mono',monospace; font-size:0.76rem;">{p}</code> does not exist on disk.<br>
        Please check model file availability in <code style="background:rgba(0,0,0,.3); padding:2px 8px; border-radius:5px;
        color:#ff7a92; font-family:'JetBrains Mono',monospace; font-size:0.76rem;">models/</code> folder.
    </div>
</div>
"""

# ─────────────────────────────────────────────
# Model loading & Preprocessing
# ─────────────────────────────────────────────
def loss_fn(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return tf.keras.models.load_model(
        path, custom_objects={"loss_fn": loss_fn}
    )

def preprocess(pil_image: Image.Image) -> np.ndarray:
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.markdown(SIDEBAR_HTML, unsafe_allow_html=True)

    # ── Header ───────────────────────────────
    st.markdown(HEADER_HTML, unsafe_allow_html=True)

    # ── Main Two-Column Layout ────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    # ── LEFT SECTION ──────────────────────────
    with col1:
        # 1. Model Selection Card
        model_name = st.selectbox(
            "⚙️ Select AI Model",
            list(MODEL_OPTIONS.keys()),
            key="model_selector"
        )
        st.markdown(model_info_html(model_name), unsafe_allow_html=True)

        # 2. Image Upload Card
        uploaded = st.file_uploader(
            "📤 Upload Ultrasound Image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="ultrasound_uploader"
        )

        # 3. Selected Image Preview
        if uploaded is not None:
            pil_img = Image.open(uploaded)
            st.markdown(img_meta_html(pil_img.size[0], pil_img.size[1]), unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)

    # ── RIGHT SECTION ─────────────────────────
    with col2:
        model_path = MODEL_OPTIONS[model_name]

        if not os.path.exists(model_path):
            st.markdown(NOT_FOUND_HTML(model_path), unsafe_allow_html=True)
        elif uploaded is None:
            st.markdown(EMPTY_STATE_RIGHT_HTML, unsafe_allow_html=True)
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown(INTERP_HTML, unsafe_allow_html=True)
        else:
            with st.spinner("⚡ Running TensorFlow Deep Learning Inference..."):
                model = load_model(model_path)
                x     = preprocess(pil_img)
                probs = model.predict(x, verbose=0)[0]
                if len(probs) == 1:
                    pm    = float(probs[0])
                    probs = np.array([1.0 - pm, pm])
                idx   = int(np.argmax(probs))
                label = CLASS_NAMES[idx]
                conf  = float(probs[idx])

            # Prediction Card
            st.markdown(result_card_html(label, conf, idx), unsafe_allow_html=True)

            # Class Probabilities Breakdown
            st.markdown(prob_bars_html(probs), unsafe_allow_html=True)

            # Interpretation Guide
            st.markdown(INTERP_HTML, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
