import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import random
import time
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io

# ============================================================
# SETTING
# ============================================================
MODEL_PATH = "best_faceshape_MobileNetV2_FINAL.keras"
HAIR_PATH  = "Rambut_Labeled"
HIJAB_PATH = "Hijab_Labeled"

st.set_page_config(
    page_title="FaceStyle AI",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-sizing: border-box;
}

[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
#MainMenu                        { display: none !important; }
footer                           { display: none !important; }
[data-testid="stHeader"]         { background: transparent !important; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"], .stApp, .main {
    background: #F7F5FF !important;
}
.main .block-container {
    padding: 0 0 4rem !important;
    max-width: 860px !important;
    margin: 0 auto !important;
}

h1,h2,h3,h4,h5,h6 { color: #1A1035 !important; }
p, span, label, div { color: #2D2D2D; }

/* ---- HERO ---- */
.hero {
    background: linear-gradient(135deg, #5B4FCF 0%, #8B7FF5 60%, #B8AFFF 100%);
    padding: 3rem 2rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    border-radius: 0 0 32px 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:280px; height:280px; border-radius:50%;
    background:rgba(255,255,255,0.07);
}
.hero-badge {
    display:inline-block;
    background:rgba(255,255,255,0.2);
    border:1px solid rgba(255,255,255,0.35);
    border-radius:50px; padding:4px 18px;
    font-size:0.72rem; font-weight:700;
    color:#fff !important; letter-spacing:1.5px;
    text-transform:uppercase; margin-bottom:0.8rem;
}
.hero-title {
    font-size:2.8rem; font-weight:800;
    color:#fff !important; margin:0 0 0.5rem;
    letter-spacing:-1px; line-height:1.1;
}
.hero-sub {
    font-size:1rem; color:rgba(255,255,255,0.88) !important;
    font-weight:400; max-width:500px;
    margin:0 auto; line-height:1.65;
    text-align:center !important;
    display:block !important;
}

/* ---- STEP CARD ---- */
.step-header {
    display:flex; align-items:center; gap:0.75rem;
    margin-bottom:1.2rem;
}
.step-num {
    width:36px; height:36px; border-radius:50%;
    background:linear-gradient(135deg,#5B4FCF,#8B7FF5);
    color:#fff !important; font-weight:800; font-size:0.9rem;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0; box-shadow:0 4px 12px rgba(91,79,207,0.3);
}
.step-title {
    font-size:1.1rem; font-weight:800; color:#1A1035 !important;
}
.step-desc {
    font-size:0.82rem; color:#8882B0 !important; margin-top:1px;
}

/* ---- CONTAINER OVERRIDE ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background:#FFFFFF !important;
    border:1px solid #E8E4FF !important;
    border-radius:24px !important;
    box-shadow:0 4px 28px rgba(91,79,207,0.07) !important;
    padding:1.8rem !important;
    margin-bottom:1.2rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding:0 !important; gap:0.8rem !important;
}

/* ---- RADIO PILL ---- */
div[role="radiogroup"] {
    gap:8px !important; display:flex !important; flex-wrap:wrap !important;
}
div[role="radiogroup"] label {
    background:#F0EEFF !important;
    border:2px solid #D8D0F5 !important;
    border-radius:50px !important;
    padding:7px 20px !important; margin:0 !important;
    cursor:pointer !important; transition:all 0.2s !important;
}
div[role="radiogroup"] label span,
div[role="radiogroup"] label p,
div[role="radiogroup"] label > div > p {
    color:#4A3B8B !important; font-weight:600 !important; font-size:0.88rem !important;
}
div[role="radiogroup"] label:has(input:checked) {
    background:linear-gradient(135deg,#5B4FCF,#8B7FF5) !important;
    border-color:#5B4FCF !important;
}
div[role="radiogroup"] label:has(input:checked) span,
div[role="radiogroup"] label:has(input:checked) p,
div[role="radiogroup"] label:has(input:checked) > div > p {
    color:#fff !important;
}
div[role="radiogroup"] label input[type="radio"] { display:none !important; }
div[role="radiogroup"] label > div:first-child   { display:none !important; }

/* ---- SELECTBOX ---- */
[data-testid="stSelectbox"] > label {
    color:#4A3B8B !important; font-weight:700 !important; font-size:0.85rem !important;
}
[data-testid="stSelectbox"] > div > div {
    background:#F0EEFF !important; border:2px solid #D8D0F5 !important;
    border-radius:12px !important; color:#2D2D2D !important;
}
[data-baseweb="popover"] { background:#fff !important; }
[data-baseweb="popover"] [role="listbox"] {
    background:#fff !important; border:2px solid #D8D0F5 !important;
    border-radius:14px !important; box-shadow:0 12px 40px rgba(91,79,207,0.15) !important;
}
[data-baseweb="popover"] [role="option"] {
    background:#fff !important; color:#2D2D2D !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background:#F0EEFF !important; color:#4A3B8B !important;
}
[data-baseweb="popover"] [aria-selected="true"] {
    background:#E8E4FF !important; color:#4A3B8B !important; font-weight:700 !important;
}

/* ---- FILE UPLOADER ---- */
[data-testid="stFileUploader"]         { background:transparent !important; }
[data-testid="stFileUploader"] section {
    background:#F7F5FF !important;
    border:2px dashed #B8AFFF !important;
    border-radius:16px !important; padding:1.5rem !important;
}
[data-testid="stFileUploader"] section * { color:#7B6FD4 !important; }
[data-testid="stFileUploader"] section button {
    background:#fff !important; border:2px solid #D8D0F5 !important;
    border-radius:50px !important; color:#5B4FCF !important;
    font-weight:700 !important; padding:6px 20px !important;
}

/* ---- CAMERA — override semua elemen hitam/merah ---- */
[data-testid="stCameraInput"] {
    background: transparent !important;
}
[data-testid="stCameraInput"] > label {
    color: #4A3B8B !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
}
/* Box container kamera */
[data-testid="stCameraInput"] section,
[data-testid="stCameraInput"] > div {
    background: #F0EEFF !important;
    border: 2px solid #D8D0F5 !important;
    border-radius: 20px !important;
    overflow: hidden !important;
    padding: 0 !important;
}
/* Video live */
[data-testid="stCameraInput"] video {
    border-radius: 18px !important;
    width: 100% !important;
    transform: scaleX(-1) !important;
}
/* Foto hasil tangkap */
[data-testid="stCameraInput"] img {
    border-radius: 18px !important;
    width: 100% !important;
    transform: scaleX(-1) !important;
}
/* Tombol Take Photo — ganti merah jadi ungu */
[data-testid="stCameraInput"] button,
[data-testid="stCameraInput"] [data-testid="cameraCaptureButton"],
[data-testid="stCameraInput"] [kind="secondary"] {
    background: linear-gradient(135deg, #5B4FCF, #8B7FF5) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 50px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 16px rgba(91,79,207,0.3) !important;
    margin-top: 0.5rem !important;
}
[data-testid="stCameraInput"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(91,79,207,0.45) !important;
}
/* Sembunyikan icon kamera bawaan yang aneh */
[data-testid="stCameraInput"] svg { display: none !important; }

/* ---- CHECKBOX ---- */
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label span,
[data-testid="stCheckbox"] label p {
    color:#2D2D2D !important; font-size:0.85rem !important; font-weight:500 !important;
}

/* ---- BUTTONS ---- */
.stButton > button {
    background:linear-gradient(135deg,#5B4FCF 0%,#8B7FF5 100%) !important;
    color:#fff !important; border:none !important; border-radius:50px !important;
    padding:0.75rem 2rem !important; font-weight:700 !important;
    font-size:0.95rem !important; width:100% !important;
    transition:all 0.25s !important;
    box-shadow:0 4px 20px rgba(91,79,207,0.3) !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 30px rgba(91,79,207,0.45) !important;
}
.stButton > button:disabled {
    background:#E0DCF8 !important; color:#B8AFFF !important;
    box-shadow:none !important; transform:none !important;
}
.stDownloadButton > button {
    background:linear-gradient(135deg,#00B894,#55EFC4) !important;
    color:#fff !important; border:none !important; border-radius:50px !important;
    font-weight:700 !important; width:100% !important;
    box-shadow:0 4px 16px rgba(0,184,148,0.25) !important;
}
.stDownloadButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 24px rgba(0,184,148,0.4) !important;
}

/* ---- ALERTS ---- */
.stInfo > div    { background:#F0EEFF !important; color:#4A3B8B !important; border-radius:14px !important; border:none !important; }
.stSuccess > div { background:#EDFFF8 !important; color:#00695C !important; border-radius:14px !important; border:none !important; }
.stError > div   { background:#FFF2F2 !important; border-radius:14px !important; border:none !important; }

/* ---- PHOTO BADGES ---- */
.photo-label { text-align:center; margin-top:0.6rem; }
.label-name  { font-size:0.82rem; font-weight:700; color:#1A1035 !important; text-transform:uppercase; letter-spacing:0.6px; }
.badge-rec {
    display:inline-block;
    background:linear-gradient(135deg,#00B894,#55EFC4);
    color:#fff !important; font-size:0.68rem; font-weight:700;
    padding:2px 10px; border-radius:20px; margin-top:4px;
}
.badge-no {
    display:inline-block; background:#F0F0F0; color:#999 !important;
    font-size:0.68rem; font-weight:600; padding:2px 10px; border-radius:20px; margin-top:4px;
}

/* ---- FACE BANNER ---- */
.face-banner {
    background:linear-gradient(135deg,#5B4FCF 0%,#8B7FF5 100%);
    border-radius:18px; padding:1.3rem 1.8rem; margin-bottom:1.5rem;
    display:flex; align-items:center; justify-content:space-between;
    box-shadow:0 6px 20px rgba(91,79,207,0.22);
}
.face-banner h2 { color:#fff !important; font-size:1.5rem !important; margin:0 !important; font-weight:800 !important; }
.face-banner p  { color:rgba(255,255,255,0.8) !important; margin:0.2rem 0 0 !important; font-size:0.88rem !important; }
.face-conf {
    background:rgba(255,255,255,0.2); border-radius:50px;
    padding:0.4rem 1.1rem; color:#fff !important; font-weight:800; font-size:1rem;
}

/* ---- TIPS ---- */
.tips-box {
    background:#F0EEFF; border-radius:16px;
    padding:1.1rem 1.4rem; border:1px solid #D8D0F5; margin-top:1rem;
}
.tips-box .t { font-weight:800; color:#4A3B8B !important; font-size:0.82rem; margin-bottom:0.4rem; }
.tips-box ul  { margin:0; padding-left:1.2rem; }
.tips-box li  { color:#5B4FCF !important; font-size:0.8rem; margin-bottom:2px; font-weight:500; }

/* ---- HISTORY CHIPS ---- */
.hist-chip {
    display:inline-block; background:#F0EEFF; border:1px solid #D8D0F5;
    border-radius:50px; padding:4px 14px; font-size:0.75rem; font-weight:600;
    color:#5B4FCF !important; margin:3px;
}

/* ---- EMPTY STATE ---- */
.empty { text-align:center; padding:3rem 2rem; }
.empty .ico { font-size:2.8rem; display:block; margin-bottom:0.8rem; }
.empty p { color:#B8AFFF !important; font-size:0.92rem; line-height:1.7; }

/* ---- DIVIDER ---- */
.divider { height:1px; background:#EAE6FF; margin:1.2rem 0; }

/* ---- FOOTER ---- */
.footer {
    text-align:center; padding:2rem 0 1rem;
    border-top:1px solid #E8E4FF; margin-top:2rem;
}
.footer p { color:#B8AFFF !important; font-size:0.78rem; margin:0.2rem 0; }

img { border-radius:12px !important; }
</style>

<script>
(function(){
    function fix(){
        /* Dropdown fix */
        document.querySelectorAll('[data-baseweb="popover"],[role="listbox"]').forEach(function(el){
            el.style.setProperty('background','#FFFFFF','important');
            el.style.setProperty('border','2px solid #D8D0F5','important');
            el.style.setProperty('border-radius','14px','important');
        });
        document.querySelectorAll('[role="option"]').forEach(function(el){
            el.style.setProperty('background','#FFFFFF','important');
            el.style.setProperty('color','#2D2D2D','important');
        });
        /* Camera button fix — ganti merah ke ungu via JS sebagai backup CSS */
        document.querySelectorAll(
            '[data-testid="stCameraInput"] button'
        ).forEach(function(btn){
            btn.style.setProperty('background','linear-gradient(135deg,#5B4FCF,#8B7FF5)','important');
            btn.style.setProperty('color','#FFFFFF','important');
            btn.style.setProperty('border','none','important');
            btn.style.setProperty('border-radius','50px','important');
            btn.style.setProperty('font-weight','700','important');
            btn.style.setProperty('padding','0.6rem 2rem','important');
            btn.style.setProperty('width','100%','important');
            btn.style.setProperty('box-shadow','0 4px 16px rgba(91,79,207,0.3)','important');
        });
        /* File uploader bg fix */
        document.querySelectorAll(
            '[data-testid="stFileUploader"] section *:not(button)'
        ).forEach(function(el){
            el.style.setProperty('background','transparent','important');
            el.style.setProperty('color','#7B6FD4','important');
        });
    }
    new MutationObserver(fix).observe(document.body,{childList:true,subtree:true});
    fix();
})();
</script>
""", unsafe_allow_html=True)

# ============================================================
# LOAD RESOURCES
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

model        = load_model()
face_cascade = load_face_cascade()
class_names  = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# ============================================================
# DATA
# ============================================================
recommendation_rules = {
    'Heart':  {'lurus': ['panjang'], 'gelombang': ['panjang', 'semi'], 'keriting': ['panjang', 'semi']},
    'Oval':   {'lurus': ['panjang', 'semi', 'pendek'], 'gelombang': ['panjang', 'semi', 'pendek'], 'keriting': ['panjang', 'semi', 'pendek']},
    'Round':  {'lurus': ['panjang'], 'gelombang': ['panjang', 'semi'], 'keriting': ['panjang']},
    'Square': {'lurus': ['panjang', 'semi'], 'gelombang': ['semi', 'panjang'], 'keriting': ['semi']},
    'Oblong': {'lurus': ['pendek', 'semi'], 'gelombang': ['pendek', 'semi'], 'keriting': ['pendek', 'semi']},
}
hijab_rules = {
    'Oval':   ['segi_empat', 'pashmina', 'instant'],
    'Round':  ['pashmina', 'segi_empat'],
    'Square': ['pashmina', 'instant'],
    'Oblong': ['segi_empat', 'instant'],
    'Heart':  ['segi_empat', 'pashmina'],
}
face_shape_desc = {
    'Heart':  'Dahi lebar, dagu lancip',
    'Oval':   'Proporsional dan seimbang',
    'Round':  'Lebar dan panjang hampir sama',
    'Square': 'Rahang tegas dan lebar',
    'Oblong': 'Panjang melebihi lebar',
}
hijab_labels = {'instant': 'Instant', 'pashmina': 'Pashmina', 'segi_empat': 'Segi Empat'}

# ============================================================
# HELPERS
# ============================================================
def detect_face_shape(image_array: np.ndarray):
    img   = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    pw, ph = int(w * 0.35), int(h * 0.35)
    y1, y2 = max(0, y-ph), min(img.shape[0], y+h+ph)
    x1, x2 = max(0, x-pw), min(img.shape[1], x+w+pw)
    face    = cv2.resize(img[y1:y2, x1:x2], (224, 224))
    face    = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_arr = np.array(face, dtype=np.float32)
    face_arr = tf.keras.applications.mobilenet_v2.preprocess_input(face_arr)
    tensor   = tf.expand_dims(face_arr, 0)
    preds    = model.predict(tensor, verbose=0)
    return class_names[int(np.argmax(preds))], float(np.max(preds))

def get_photos(folder_path):
    p = Path(folder_path)
    if not p.exists():
        return []
    return list(p.glob('*.jpg')) + list(p.glob('*.jpeg')) + list(p.glob('*.png'))

def save_to_database(image_array, face_shape, sub_type, mode):
    ts = int(time.time())
    if mode == 'rambut':
        length, hair_type = sub_type
        folder = Path(HAIR_PATH) / f"{face_shape}_{length}_{hair_type}"
    else:
        folder = Path(HIJAB_PATH) / f"{face_shape}_{sub_type}"
    folder.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(folder / f"user_{ts}.jpg"), bgr)

def create_result_image(photos, labels, recommended_list, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#F7F5FF')
    for ax, photo, label in zip(axes, photos, labels):
        if photo:
            ax.imshow(mpimg.imread(str(photo)))
        else:
            ax.text(0.5, 0.5, 'Tidak\nTersedia', ha='center', va='center',
                    fontsize=11, color='#AAAAAA', transform=ax.transAxes)
            ax.set_facecolor('#F0EEFF')
        ax.axis('off')
        is_rec = any(r.upper() in label.upper() for r in recommended_list)
        ax.set_title(label, fontsize=10, pad=8,
                     color='#00B894' if is_rec else '#AAAAAA',
                     fontweight='bold' if is_rec else 'normal')
    plt.suptitle(title, fontsize=13, fontweight='bold', color='#4A3B8B', y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close()
    return buf

# ============================================================
# SESSION STATE
# ============================================================
for k, v in [('history', []), ('show_result', False), ('image_array', None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI Beauty Advisor</div>
    <h1 class="hero-title">FaceStyle AI</h1>
    <p class="hero-sub">
        Deteksi bentuk wajahmu dan dapatkan rekomendasi
        gaya rambut &amp; hijab yang paling cocok untukmu
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# STEP 1 — PREFERENSI
# ============================================================
with st.container(border=True):
    st.markdown("""
    <div class="step-header">
        <div class="step-num">1</div>
        <div>
            <div class="step-title">Pilih Mode &amp; Preferensi</div>
            <div class="step-desc">Pilih apakah kamu ingin rekomendasi rambut atau hijab</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Mode Rekomendasi**")
        mode = st.radio("mode", ["Rambut", "Hijab"],
                        horizontal=False, label_visibility="collapsed")

    with c2:
        if mode == "Rambut":
            st.markdown("**Jenis Rambut**")
            hair_type = st.selectbox("jenis", ["lurus", "gelombang", "keriting"],
                                     label_visibility="collapsed")
        else:
            st.markdown("**Jenis Hijab**")
            hijab_type = st.selectbox("jenis", ["instant", "pashmina", "segi_empat"],
                                      label_visibility="collapsed",
                                      format_func=lambda x: hijab_labels.get(x, x))

    with c3:
        if mode == "Rambut":
            st.markdown("**Panjang Rambut Saat Ini**")
            length_user = st.selectbox("panjang", ["pendek", "semi", "panjang"],
                                       label_visibility="collapsed")
        else:
            st.markdown(" ")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    consent = st.checkbox(
        "Saya setuju foto saya disimpan sebagai referensi rekomendasi untuk pengguna lain"
    )

# ============================================================
# STEP 2 — UPLOAD FOTO
# ============================================================
with st.container(border=True):
    st.markdown("""
    <div class="step-header">
        <div class="step-num">2</div>
        <div>
            <div class="step-title">Foto Wajah</div>
            <div class="step-desc">Upload foto atau ambil selfie langsung dari kamera</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    input_method = st.radio(
        "input_method",
        ["Upload dari Galeri", "Selfie dengan Kamera"],
        horizontal=True,
        label_visibility="collapsed"
    )

    image_array = None

    if input_method == "Upload dari Galeri":
        uploaded = st.file_uploader(
            "Pilih foto wajah yang jelas (format JPG atau PNG)",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="visible"
        )
        if uploaded:
            pil_img     = Image.open(uploaded).convert('RGB')
            image_array = np.array(pil_img)
            st.image(pil_img, use_column_width=True, caption="Foto siap dianalisis")
    else:
        st.markdown("""
        <div style='background:#F0EEFF;border-radius:14px;padding:0.8rem 1rem;
                    border:1px solid #D8D0F5;margin-bottom:0.8rem'>
            <p style='color:#5B4FCF !important;font-size:0.82rem;font-weight:600;margin:0'>
                Panduan selfie yang baik: Hadap kamera langsung, pastikan wajah terlihat jelas,
                dan gunakan pencahayaan yang cukup.
            </p>
        </div>
        """, unsafe_allow_html=True)

        cam = st.camera_input(
            "Tekan tombol di bawah untuk mengambil foto",
            label_visibility="visible"
        )
        if cam:
            pil_img     = Image.open(cam).convert('RGB')
            pil_img     = ImageOps.mirror(pil_img)
            image_array = np.array(pil_img)
            st.image(pil_img, use_column_width=True, caption="Foto siap dianalisis")

    if image_array is not None:
        st.session_state.image_array = image_array

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if st.button(
        "Mulai Analisis Wajah",
        disabled=(st.session_state.image_array is None),
        use_container_width=True
    ):
        st.session_state.show_result = True
        st.rerun()

    st.markdown("""
    <div class="tips-box">
        <div class="t">Tips untuk hasil terbaik</div>
        <ul>
            <li>Pastikan wajah terlihat jelas dan tidak tertutup</li>
            <li>Gunakan pencahayaan yang cukup dan merata</li>
            <li>Hadap kamera secara langsung (lurus)</li>
            <li>Hindari penggunaan kacamata atau topi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# STEP 3 — HASIL (hanya muncul setelah tombol ditekan)
# ============================================================
if st.session_state.show_result and st.session_state.image_array is not None:

    with st.container(border=True):
        st.markdown("""
        <div class="step-header">
            <div class="step-num">3</div>
            <div>
                <div class="step-title">Hasil Analisis &amp; Rekomendasi</div>
                <div class="step-desc">Berikut adalah hasil deteksi bentuk wajah dan rekomendasinya</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Sedang menganalisis bentuk wajah..."):
            face_shape, confidence = detect_face_shape(st.session_state.image_array)

        if face_shape is None:
            st.error(
                "Wajah tidak terdeteksi. Coba ulangi dengan foto yang lebih jelas, "
                "wajah menghadap kamera secara langsung, dan pastikan pencahayaan cukup."
            )
            if st.button("Coba Foto Lain", use_container_width=True):
                st.session_state.show_result = False
                st.session_state.image_array = None
                st.rerun()
        else:
            # Face result banner
            st.markdown(f"""
            <div class="face-banner">
                <div>
                    <h2>Wajah {face_shape}</h2>
                    <p>{face_shape_desc.get(face_shape, '')}</p>
                </div>
                <div class="face-conf">{confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # ---- RAMBUT ----
            if mode == "Rambut":
                rec_lengths = recommendation_rules[face_shape][hair_type]
                all_lengths = ['pendek', 'semi', 'panjang']

                st.info(
                    f"Jenis rambut: **{hair_type.capitalize()}**  |  "
                    f"Panjang saat ini: **{length_user.capitalize()}**  |  "
                    f"Cocok untuk: **{', '.join(rec_lengths)}**"
                )

                cols3 = st.columns(3)
                photos_shown, labels_shown = [], []

                for col, length in zip(cols3, all_lengths):
                    folder = Path(HAIR_PATH) / f"{face_shape}_{length}_{hair_type}"
                    photos = get_photos(folder)
                    is_rec = length in rec_lengths

                    with col:
                        if photos:
                            photo = random.choice(photos)
                            st.image(Image.open(str(photo)), use_column_width=True)
                            photos_shown.append(photo)
                        else:
                            st.markdown(
                                "<div style='background:#F0EEFF;border-radius:12px;"
                                "height:150px;display:flex;align-items:center;"
                                "justify-content:center'><p style='color:#B8AFFF;"
                                "font-size:0.78rem;text-align:center;margin:0'>"
                                "Foto belum<br>tersedia</p></div>",
                                unsafe_allow_html=True
                            )
                            photos_shown.append(None)

                        badge     = "badge-rec" if is_rec else "badge-no"
                        badge_txt = "Direkomendasikan" if is_rec else "Kurang cocok"
                        st.markdown(f"""
                        <div class='photo-label'>
                            <span class='label-name'>{length.capitalize()}</span><br>
                            <span class='{badge}'>{badge_txt}</span>
                        </div>""", unsafe_allow_html=True)
                        labels_shown.append(f"{length.upper()} - {badge_txt}")

                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                dl_col, retry_col = st.columns(2)

                with dl_col:
                    buf = create_result_image(
                        photos_shown, labels_shown, rec_lengths,
                        f"Rambut {hair_type.capitalize()} - Wajah {face_shape}"
                    )
                    st.download_button(
                        "Download Hasil",
                        data=buf,
                        file_name=f"facestyle_{face_shape}_{hair_type}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                with retry_col:
                    if st.button("Analisis Foto Lain", use_container_width=True):
                        st.session_state.show_result = False
                        st.session_state.image_array = None
                        st.rerun()

                st.session_state.history.append(f"Rambut {hair_type} - {face_shape}")
                if consent:
                    save_to_database(
                        st.session_state.image_array,
                        face_shape, (length_user, hair_type), 'rambut'
                    )
                    st.success("Foto berhasil disimpan. Terima kasih atas kontribusinya!")

            # ---- HIJAB ----
            else:
                rec_hijab = hijab_rules[face_shape]
                all_hijab = ['instant', 'pashmina', 'segi_empat']

                st.info(
                    f"Jenis hijab: **{hijab_labels.get(hijab_type, hijab_type)}**  |  "
                    f"Cocok untuk: **{', '.join([hijab_labels.get(h, h) for h in rec_hijab])}**"
                )

                cols3 = st.columns(3)
                photos_shown, labels_shown = [], []

                for col, htype in zip(cols3, all_hijab):
                    folder = Path(HIJAB_PATH) / f"{face_shape}_{htype}"
                    photos = get_photos(folder)
                    is_rec = htype in rec_hijab

                    with col:
                        if photos:
                            photo = random.choice(photos)
                            st.image(Image.open(str(photo)), use_column_width=True)
                            photos_shown.append(photo)
                        else:
                            st.markdown(
                                "<div style='background:#F0EEFF;border-radius:12px;"
                                "height:150px;display:flex;align-items:center;"
                                "justify-content:center'><p style='color:#B8AFFF;"
                                "font-size:0.78rem;text-align:center;margin:0'>"
                                "Foto belum<br>tersedia</p></div>",
                                unsafe_allow_html=True
                            )
                            photos_shown.append(None)

                        badge     = "badge-rec" if is_rec else "badge-no"
                        badge_txt = "Direkomendasikan" if is_rec else "Kurang cocok"
                        st.markdown(f"""
                        <div class='photo-label'>
                            <span class='label-name'>{hijab_labels.get(htype, htype)}</span><br>
                            <span class='{badge}'>{badge_txt}</span>
                        </div>""", unsafe_allow_html=True)
                        labels_shown.append(f"{hijab_labels.get(htype, htype)} - {badge_txt}")

                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                dl_col, retry_col = st.columns(2)

                with dl_col:
                    buf = create_result_image(
                        photos_shown, labels_shown, rec_hijab,
                        f"Hijab - Wajah {face_shape}"
                    )
                    st.download_button(
                        "Download Hasil",
                        data=buf,
                        file_name=f"facestyle_{face_shape}_{hijab_type}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                with retry_col:
                    if st.button("Analisis Foto Lain",
                                 use_container_width=True, key="retry_hijab"):
                        st.session_state.show_result = False
                        st.session_state.image_array = None
                        st.rerun()

                st.session_state.history.append(
                    f"Hijab {hijab_labels.get(hijab_type, hijab_type)} - {face_shape}"
                )
                if consent:
                    save_to_database(
                        st.session_state.image_array,
                        face_shape, hijab_type, 'hijab'
                    )
                    st.success("Foto berhasil disimpan. Terima kasih atas kontribusinya!")

        # Riwayat
        if st.session_state.history:
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.8rem;font-weight:700;color:#8882B0'>Riwayat Analisis</p>", unsafe_allow_html=True)
            chips = "".join([
                f"<span class='hist-chip'>{h}</span>"
                for h in reversed(st.session_state.history[-5:])
            ])
            st.markdown(chips, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <p>FaceStyle AI — Powered by MobileNetV2 Transfer Learning</p>
    <p>Akurasi Model 83.54% &nbsp;·&nbsp; 5 Kelas Bentuk Wajah &nbsp;·&nbsp; Semester 4 Project</p>
</div>
""", unsafe_allow_html=True)