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
HAIR_PATH  = "Rambut_Labeled_V2"   # ← folder baru hasil distribute_labeled.py
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
    max-width: 900px !important;
    margin: 0 auto !important;
}

h1,h2,h3,h4,h5,h6 { color: #1A1035 !important; }
p, span, label, div { color: #2D2D2D; }

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
.step-title  { font-size:1.1rem; font-weight:800; color:#1A1035 !important; }
.step-desc   { font-size:0.82rem; color:#8882B0 !important; margin-top:1px; }

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
div[role="radiogroup"] label:has(input:checked) > div > p { color:#fff !important; }
div[role="radiogroup"] label input[type="radio"] { display:none !important; }
div[role="radiogroup"] label > div:first-child   { display:none !important; }

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
[data-baseweb="popover"] [role="option"] { background:#fff !important; color:#2D2D2D !important; }
[data-baseweb="popover"] [role="option"]:hover { background:#F0EEFF !important; color:#4A3B8B !important; }
[data-baseweb="popover"] [aria-selected="true"] {
    background:#E8E4FF !important; color:#4A3B8B !important; font-weight:700 !important;
}

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

[data-testid="stCameraInput"] { background: transparent !important; }
[data-testid="stCameraInput"] > label { color: #4A3B8B !important; font-weight: 700 !important; }
[data-testid="stCameraInput"] section, [data-testid="stCameraInput"] > div {
    background: #F0EEFF !important; border: 2px solid #D8D0F5 !important;
    border-radius: 20px !important; overflow: hidden !important; padding: 0 !important;
}
[data-testid="stCameraInput"] video { border-radius:18px !important; width:100% !important; transform:scaleX(-1) !important; }
[data-testid="stCameraInput"] img   { border-radius:18px !important; width:100% !important; transform:scaleX(-1) !important; }
[data-testid="stCameraInput"] button {
    background: linear-gradient(135deg, #5B4FCF, #8B7FF5) !important;
    color: #FFFFFF !important; border: none !important; border-radius: 50px !important;
    font-weight: 700 !important; width: 100% !important;
    box-shadow: 0 4px 16px rgba(91,79,207,0.3) !important; margin-top: 0.5rem !important;
}
[data-testid="stCameraInput"] svg { display: none !important; }

[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label span,
[data-testid="stCheckbox"] label p {
    color:#2D2D2D !important; font-size:0.85rem !important; font-weight:500 !important;
}

.stButton > button {
    background:linear-gradient(135deg,#5B4FCF 0%,#8B7FF5 100%) !important;
    color:#fff !important; border:none !important; border-radius:50px !important;
    padding:0.75rem 2rem !important; font-weight:700 !important;
    font-size:0.95rem !important; width:100% !important;
    transition:all 0.25s !important;
    box-shadow:0 4px 20px rgba(91,79,207,0.3) !important;
}
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 30px rgba(91,79,207,0.45) !important; }
.stButton > button:disabled { background:#E0DCF8 !important; color:#B8AFFF !important; box-shadow:none !important; }
.stDownloadButton > button {
    background:linear-gradient(135deg,#00B894,#55EFC4) !important;
    color:#fff !important; border:none !important; border-radius:50px !important;
    font-weight:700 !important; width:100% !important;
    box-shadow:0 4px 16px rgba(0,184,148,0.25) !important;
}
.stDownloadButton > button:hover { transform:translateY(-2px) !important; }

.stInfo > div    { background:#F0EEFF !important; color:#4A3B8B !important; border-radius:14px !important; border:none !important; }
.stSuccess > div { background:#EDFFF8 !important; color:#00695C !important; border-radius:14px !important; border:none !important; }
.stError > div   { background:#FFF2F2 !important; border-radius:14px !important; border:none !important; }

.photo-label { text-align:center; margin-top:0.6rem; }
.label-name  { font-size:0.82rem; font-weight:700; color:#1A1035 !important; text-transform:uppercase; letter-spacing:0.6px; }
.badge-rec { display:inline-block; background:linear-gradient(135deg,#00B894,#55EFC4); color:#fff !important; font-size:0.68rem; font-weight:700; padding:2px 10px; border-radius:20px; margin-top:4px; }
.badge-no  { display:inline-block; background:#F0F0F0; color:#999 !important; font-size:0.68rem; font-weight:600; padding:2px 10px; border-radius:20px; margin-top:4px; }

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

.tips-box {
    background:#F0EEFF; border-radius:16px;
    padding:1.1rem 1.4rem; border:1px solid #D8D0F5; margin-top:1rem;
}
.tips-box .t { font-weight:800; color:#4A3B8B !important; font-size:0.82rem; margin-bottom:0.4rem; }
.tips-box ul { margin:0; padding-left:1.2rem; }
.tips-box li { color:#5B4FCF !important; font-size:0.8rem; margin-bottom:2px; font-weight:500; }

.hist-chip {
    display:inline-block; background:#F0EEFF; border:1px solid #D8D0F5;
    border-radius:50px; padding:4px 14px; font-size:0.75rem; font-weight:600;
    color:#5B4FCF !important; margin:3px;
}
.divider { height:1px; background:#EAE6FF; margin:1.2rem 0; }
.footer  { text-align:center; padding:2rem 0 1rem; border-top:1px solid #E8E4FF; margin-top:2rem; }
.footer p { color:#B8AFFF !important; font-size:0.78rem; margin:0.2rem 0; }
img { border-radius:12px !important; }
</style>

<script>
(function(){
    function fix(){
        document.querySelectorAll('[data-baseweb="popover"],[role="listbox"]').forEach(function(el){
            el.style.setProperty('background','#FFFFFF','important');
        });
        document.querySelectorAll('[role="option"]').forEach(function(el){
            el.style.setProperty('background','#FFFFFF','important');
            el.style.setProperty('color','#2D2D2D','important');
        });
        document.querySelectorAll('[data-testid="stCameraInput"] button').forEach(function(btn){
            btn.style.setProperty('background','linear-gradient(135deg,#5B4FCF,#8B7FF5)','important');
            btn.style.setProperty('color','#FFFFFF','important');
            btn.style.setProperty('border','none','important');
            btn.style.setProperty('border-radius','50px','important');
            btn.style.setProperty('font-weight','700','important');
            btn.style.setProperty('width','100%','important');
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
face_shape_desc = {
    'Heart':  'Dahi lebar, dagu lancip',
    'Oval':   'Proporsional dan seimbang',
    'Round':  'Lebar dan panjang hampir sama',
    'Square': 'Rahang tegas dan lebar',
    'Oblong': 'Panjang melebihi lebar',
}

# Panjang yang direkomendasikan per bentuk wajah
length_rules = {
    'Heart':  ['panjang'],
    'Oval':   ['panjang', 'semi', 'pendek'],
    'Round':  ['panjang', 'semi'],
    'Square': ['panjang', 'semi'],
    'Oblong': ['pendek', 'semi'],
}

# Label tampilan nama style
style_labels = {
    'curly':    'Curly',
    'layer':    'Layer',
    'straight': 'Straight',
    'wavy':     'Wavy',
    'wolfcut':  'Wolf Cut',
}

# Aturan poni per bentuk wajah
# Format: {shape: [(jenis_poni, deskripsi_singkat, cocok: True/False), ...]}
poni_rules = {
    'Heart': [
        ('Curtain Bangs',    'Poni belah tengah menyamping',      True),
        ('Wispy Bangs',      'Poni tipis ringan',                  True),
        ('Side-Swept Bangs', 'Poni menyamping ke satu sisi',       True),
        ('Straight Blunt',   'Poni lurus tebal penuh',             False),
        ('Baby Bangs',       'Poni super pendek di atas alis',     False),
    ],
    'Oval': [
        ('Curtain Bangs',    'Poni belah tengah menyamping',      True),
        ('Straight Blunt',   'Poni lurus tebal penuh',             True),
        ('Wispy Bangs',      'Poni tipis ringan',                  True),
        ('Baby Bangs',       'Poni super pendek di atas alis',     True),
        ('Side-Swept Bangs', 'Poni menyamping ke satu sisi',       True),
    ],
    'Round': [
        ('Side-Swept Bangs', 'Poni menyamping ke satu sisi',       True),
        ('Curtain Bangs',    'Poni belah tengah menyamping',       True),
        ('Wispy Bangs',      'Poni tipis ringan',                  True),
        ('Straight Blunt',   'Poni lurus tebal penuh',             False),
        ('Baby Bangs',       'Poni super pendek di atas alis',     False),
    ],
    'Square': [
        ('Curtain Bangs',    'Poni belah tengah menyamping',      True),
        ('Wispy Bangs',      'Poni tipis ringan',                  True),
        ('Side-Swept Bangs', 'Poni menyamping ke satu sisi',       True),
        ('Straight Blunt',   'Poni lurus tebal penuh',             False),
        ('Baby Bangs',       'Poni super pendek di atas alis',     False),
    ],
    'Oblong': [
        ('Straight Blunt',   'Poni lurus tebal penuh',             True),
        ('Wispy Bangs',      'Poni tipis ringan',                  True),
        ('Baby Bangs',       'Poni super pendek di atas alis',     True),
        ('Curtain Bangs',    'Poni belah tengah menyamping',       False),
        ('Side-Swept Bangs', 'Poni menyamping ke satu sisi',       False),
    ],
}

# Hijab
hijab_rules = {
    'Oval':   ['segi_empat', 'pashmina', 'instant'],
    'Round':  ['pashmina', 'segi_empat'],
    'Square': ['pashmina', 'instant'],
    'Oblong': ['segi_empat', 'instant'],
    'Heart':  ['segi_empat', 'pashmina'],
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

    # ✅ float32 tanpa preprocess_input — konsisten dengan cara model dilatih
    face_arr = np.array(face, dtype=np.float32)
    tensor   = tf.expand_dims(face_arr, 0)
    preds    = model.predict(tensor, verbose=0)
    return class_names[int(np.argmax(preds))], float(np.max(preds))


def get_photos(folder_path):
    p = Path(folder_path)
    if not p.exists():
        return []
    return (
        list(p.glob('*.jpg')) +
        list(p.glob('*.jpeg')) +
        list(p.glob('*.png'))
    )


def get_top3_styles(face_shape: str):
    """
    Scan semua folder {face_shape}_{length}_{style} di HAIR_PATH.
    Kembalikan 3 folder terbaik berdasarkan:
      1. Panjang termasuk dalam length_rules (prioritas utama)
      2. Jumlah foto terbanyak (tiebreaker)

    Struktur folder baru: Oval_panjang_layer, Round_semi_wavy, dst.
    Format: {Shape}_{length}_{style}
    """
    base     = Path(HAIR_PATH)
    rec_lens = length_rules.get(face_shape, [])
    candidates = []

    if not base.exists():
        return []

    for folder in base.iterdir():
        if not folder.is_dir():
            continue

        # Nama folder: Shape_length_style  → split maxsplit=2
        parts = folder.name.split('_', 2)
        if len(parts) != 3:
            continue

        shape, length, style = parts
        if shape != face_shape:
            continue

        photos = get_photos(folder)
        if not photos:
            continue

        candidates.append({
            'folder':      folder,
            'length':      length,
            'style':       style,
            'photos':      photos,
            'is_rec':      length in rec_lens,
            'photo_count': len(photos),
        })

    # Recommended dulu, lalu jumlah foto terbanyak
    candidates.sort(key=lambda x: (not x['is_rec'], -x['photo_count']))
    return candidates[:3]


def save_to_database(image_array, face_shape, sub_type, mode):
    """
    mode=rambut: sub_type=(length_user, style_user) simpan ke folder sesuai pilihan user
    mode=hijab:  sub_type=hijab_type string
    """
    ts = int(time.time())
    if mode == 'hijab':
        folder = Path(HIJAB_PATH) / f"{face_shape}_{sub_type}"
    else:
        length_user, style_user = sub_type
        folder = Path(HAIR_PATH) / f"{face_shape}_{length_user}_{style_user}"
    folder.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(folder / f"user_{ts}.jpg"), bgr)


def create_result_image_top3(top3_items, face_shape):
    n    = len(top3_items)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor('#F7F5FF')

    for i, (ax, item) in enumerate(zip(axes, top3_items)):
        photo = random.choice(item['photos'])
        ax.imshow(mpimg.imread(str(photo)))
        ax.axis('off')
        style_txt  = style_labels.get(item['style'], item['style'])
        len_txt    = item['length'].capitalize()
        combo_txt  = f"{len_txt} {style_txt}"
        color      = '#5B4FCF' if item['is_rec'] else '#AAAAAA'
        suffix     = " ✓" if item['is_rec'] else ""
        ax.set_title(
            f"#{i+1}  {combo_txt}{suffix}",
            fontsize=11, pad=8, color=color, fontweight='bold'
        )

    plt.suptitle(
        f"Top 3 Rekomendasi Rambut — Wajah {face_shape}",
        fontsize=13, fontweight='bold', color='#4A3B8B', y=1.03
    )
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
    <p style="font-size:1rem;color:rgba(255,255,255,0.88);font-weight:400;
              max-width:500px;margin:0 auto;line-height:1.65;
              text-align:center;display:block;width:100%;">
        Deteksi bentuk wajahmu dan dapatkan rekomendasi
        gaya rambut &amp; hijab yang paling cocok untukmu
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# STEP 1 — MODE
# ============================================================
with st.container(border=True):
    st.markdown("""
    <div class="step-header">
        <div class="step-num">1</div>
        <div>
            <div class="step-title">Pilih Mode</div>
            <div class="step-desc">Pilih rekomendasi rambut atau hijab</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("mode", ["Rambut", "Hijab"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "Rambut":
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Jenis Rambut Kamu Sekarang**")
            hair_type_user = st.radio(
                "jenis_rambut",
                ["curly", "layer", "straight", "wavy", "wolfcut"],
                format_func=lambda x: style_labels.get(x, x),
                horizontal=True,
                label_visibility="collapsed"
            )
        with col_b:
            st.markdown("**Panjang Rambut Kamu Sekarang**")
            length_user = st.radio(
                "panjang_rambut",
                ["pendek", "semi", "panjang"],
                format_func=lambda x: x.capitalize(),
                horizontal=True,
                label_visibility="collapsed"
            )

    if mode == "Hijab":
        st.markdown("**Jenis Hijab yang kamu pakai saat ini**")
        hijab_type = st.selectbox(
            "jenis_hijab", ["instant", "pashmina", "segi_empat"],
            label_visibility="collapsed",
            format_func=lambda x: hijab_labels.get(x, x)
        )

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
        horizontal=True, label_visibility="collapsed"
    )

    image_array = None

    if input_method == "Upload dari Galeri":
        uploaded = st.file_uploader(
            "Pilih foto wajah yang jelas (JPG / PNG)",
            type=['jpg', 'jpeg', 'png'], label_visibility="visible"
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
                Hadap kamera langsung, pastikan wajah jelas, pencahayaan cukup.
            </p>
        </div>
        """, unsafe_allow_html=True)
        cam = st.camera_input("Tekan tombol untuk mengambil foto", label_visibility="visible")
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
            <li>Wajah terlihat jelas dan tidak tertutup</li>
            <li>Pencahayaan cukup dan merata</li>
            <li>Hadap kamera secara langsung</li>
            <li>Hindari kacamata atau topi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# STEP 3 — HASIL
# ============================================================
if st.session_state.show_result and st.session_state.image_array is not None:

    with st.container(border=True):
        st.markdown("""
        <div class="step-header">
            <div class="step-num">3</div>
            <div>
                <div class="step-title">Hasil Analisis &amp; Rekomendasi</div>
                <div class="step-desc">Top 3 gaya terbaik untuk bentuk wajahmu</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Sedang menganalisis bentuk wajah..."):
            face_shape, confidence = detect_face_shape(st.session_state.image_array)

        if face_shape is None:
            st.error(
                "Wajah tidak terdeteksi. Coba foto lebih jelas, "
                "menghadap kamera langsung, dengan pencahayaan cukup."
            )
            if st.button("Coba Foto Lain", use_container_width=True):
                st.session_state.show_result = False
                st.session_state.image_array = None
                st.rerun()

        else:
            # Banner bentuk wajah
            st.markdown(f"""
            <div class="face-banner">
                <div>
                    <h2>Wajah {face_shape}</h2>
                    <p>{face_shape_desc.get(face_shape, '')}</p>
                </div>
                <div class="face-conf">{confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # ==================================================
            # MODE RAMBUT — Top 3 kartu
            # ==================================================
            if mode == "Rambut":
                top3 = get_top3_styles(face_shape)

                if not top3:
                    st.warning("Belum ada foto referensi untuk bentuk wajah ini.")
                else:
                    rec_lens = length_rules.get(face_shape, [])
                    st.info(
                        f"Bentuk wajah **{face_shape}** cocok dengan panjang rambut: "
                        f"**{', '.join(rec_lens)}**"
                    )

                    st.markdown(
                        "<p style='font-size:0.85rem;font-weight:700;"
                        "color:#8882B0;margin:0.2rem 0 0.8rem'>🏆 Top 3 Gaya Rambut Untukmu</p>",
                        unsafe_allow_html=True
                    )

                    cols = st.columns(3)
                    rank_colors = ["#5B4FCF", "#8B7FF5", "#B8AFFF"]

                    for i, (col, item) in enumerate(zip(cols, top3)):
                        photo      = random.choice(item['photos'])
                        style_name = style_labels.get(item['style'], item['style'])
                        length_cap = item['length'].capitalize()
                        # Label kombinasi: "Panjang Layer", "Semi Wavy", dst.
                        combo_name = f"{length_cap} {style_name}"
                        rec_tag    = (
                            "<span style='display:inline-block;"
                            "background:linear-gradient(135deg,#00B894,#55EFC4);"
                            "color:#fff;font-size:0.65rem;font-weight:700;"
                            "padding:2px 9px;border-radius:20px;margin-top:5px'>"
                            "✓ Direkomendasikan</span>"
                        ) if item['is_rec'] else ""

                        with col:
                            st.image(Image.open(str(photo)), use_column_width=True)
                            st.markdown(f"""
                            <div style='text-align:center;padding:0.5rem 0 0.8rem'>
                                <span style='display:inline-block;
                                    background:{rank_colors[i]};color:#fff;
                                    font-size:0.7rem;font-weight:800;
                                    padding:2px 12px;border-radius:20px;
                                    margin-bottom:6px'>#{i+1}</span><br>
                                <span style='font-size:1rem;font-weight:800;
                                    color:#1A1035'>{combo_name}</span><br>
                                {rec_tag}<br>
                                <span style='font-size:0.68rem;color:#B8AFFF;
                                    margin-top:4px;display:inline-block'>
                                    {item['photo_count']} referensi
                                </span>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                    dl_col, retry_col = st.columns(2)

                    with dl_col:
                        buf = create_result_image_top3(top3, face_shape)
                        st.download_button(
                            "⬇️ Download Hasil",
                            data=buf,
                            file_name=f"facestyle_{face_shape}_top3.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    with retry_col:
                        if st.button("🔄 Analisis Foto Lain", use_container_width=True):
                            st.session_state.show_result = False
                            st.session_state.image_array = None
                            st.rerun()

                    st.session_state.history.append(f"Rambut {face_shape}")
                    if consent:
                        save_to_database(
                            st.session_state.image_array,
                            face_shape,
                            (length_user, hair_type_user),
                            'rambut'
                        )
                        st.success(
                            f"Foto disimpan ke folder **{face_shape}_{length_user}_{hair_type_user}**. "
                            "Terima kasih atas kontribusinya!"
                        )

                    # ---- REKOMENDASI PONI ----
                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                    st.markdown(
                        "<p style='font-size:0.85rem;font-weight:700;"
                        "color:#8882B0;margin:0 0 0.8rem'>✂️ Rekomendasi Poni untuk Wajahmu</p>",
                        unsafe_allow_html=True
                    )

                    poni_list = poni_rules.get(face_shape, [])
                    rec_poni  = [p for p in poni_list if p[2]]
                    not_poni  = [p for p in poni_list if not p[2]]

                    # Cocok
                    poni_rec_html = "".join([
                        f"""<div style='display:flex;align-items:flex-start;gap:10px;
                                margin-bottom:10px;background:#EDFFF8;
                                border:1px solid #B2EFD8;border-radius:14px;
                                padding:10px 14px'>
                            <span style='font-size:1.1rem;margin-top:1px'>✅</span>
                            <div>
                                <span style='font-size:0.88rem;font-weight:800;
                                    color:#00695C'>{p[0]}</span><br>
                                <span style='font-size:0.75rem;color:#4A9E7A'>{p[1]}</span>
                            </div>
                        </div>"""
                        for p in rec_poni
                    ])

                    # Tidak cocok
                    poni_no_html = "".join([
                        f"""<div style='display:flex;align-items:flex-start;gap:10px;
                                margin-bottom:10px;background:#FFF5F5;
                                border:1px solid #FFCDD2;border-radius:14px;
                                padding:10px 14px'>
                            <span style='font-size:1.1rem;margin-top:1px'>❌</span>
                            <div>
                                <span style='font-size:0.88rem;font-weight:800;
                                    color:#C62828'>{p[0]}</span><br>
                                <span style='font-size:0.75rem;color:#E57373'>{p[1]}</span>
                            </div>
                        </div>"""
                        for p in not_poni
                    ])

                    pc1, pc2 = st.columns(2)
                    with pc1:
                        st.markdown(
                            "<p style='font-size:0.78rem;font-weight:700;"
                            "color:#00695C;margin-bottom:6px'>Poni yang Cocok</p>",
                            unsafe_allow_html=True
                        )
                        st.markdown(poni_rec_html, unsafe_allow_html=True)
                    with pc2:
                        st.markdown(
                            "<p style='font-size:0.78rem;font-weight:700;"
                            "color:#C62828;margin-bottom:6px'>Poni yang Kurang Cocok</p>",
                            unsafe_allow_html=True
                        )
                        st.markdown(poni_no_html, unsafe_allow_html=True)

            # ==================================================
            # MODE HIJAB
            # ==================================================
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
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    fig.patch.set_facecolor('#F7F5FF')
                    for ax, photo, label in zip(axes, photos_shown, labels_shown):
                        if photo:
                            ax.imshow(mpimg.imread(str(photo)))
                        else:
                            ax.text(0.5, 0.5, 'Tidak\nTersedia', ha='center', va='center',
                                    fontsize=11, color='#AAAAAA', transform=ax.transAxes)
                            ax.set_facecolor('#F0EEFF')
                        ax.axis('off')
                        is_rec_label = any(r.upper() in label.upper() for r in rec_hijab)
                        ax.set_title(label, fontsize=10, pad=8,
                                     color='#00B894' if is_rec_label else '#AAAAAA',
                                     fontweight='bold' if is_rec_label else 'normal')
                    plt.suptitle(f"Hijab — Wajah {face_shape}", fontsize=13,
                                 fontweight='bold', color='#4A3B8B', y=1.02)
                    plt.tight_layout()
                    buf_h = io.BytesIO()
                    plt.savefig(buf_h, format='png', dpi=150, bbox_inches='tight',
                                facecolor=fig.get_facecolor())
                    buf_h.seek(0)
                    plt.close()

                    st.download_button(
                        "⬇️ Download Hasil",
                        data=buf_h,
                        file_name=f"facestyle_{face_shape}_{hijab_type}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                with retry_col:
                    if st.button("🔄 Analisis Foto Lain",
                                 use_container_width=True, key="retry_hijab"):
                        st.session_state.show_result = False
                        st.session_state.image_array = None
                        st.rerun()

                st.session_state.history.append(
                    f"Hijab {hijab_labels.get(hijab_type, hijab_type)} — {face_shape}"
                )
                if consent:
                    save_to_database(
                        st.session_state.image_array, face_shape, hijab_type, 'hijab'
                    )
                    st.success("Foto berhasil disimpan. Terima kasih atas kontribusinya!")

            # Riwayat
            if st.session_state.history:
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='font-size:0.8rem;font-weight:700;color:#8882B0'>"
                    "Riwayat Analisis</p>", unsafe_allow_html=True
                )
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