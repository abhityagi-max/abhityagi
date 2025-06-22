import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Page Configuration
st.set_page_config(page_title="Bank of America Denoise App", layout="wide")

# Inject custom CSS for styling
st.markdown("""
    <style>
    .title { font-size: 36px; font-weight: bold; color: #012169; }
    .subtitle { font-size: 18px; color: #444; }
    .metric { font-size: 16px; color: #000; }
    .uploaded-image { border: 1px solid #ccc; border-radius: 8px; padding: 4px; }
    </style>
""", unsafe_allow_html=True)

# Load Bank of America Logo
#logo_path = "assets/boa_logo.png"
logo_path=r"C:\Users\mannu tyagi\Downloads\bank_logo.png"  # Ensure this exists in the correct folder
st.image(logo_path, width=250)

st.markdown('<div class="title">Bank of America Denoise App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enhance scanned or noisy images (e.g., ID cards, documents) for better KYC & fraud detection</div>', unsafe_allow_html=True)
st.markdown("")

uploaded_file = st.file_uploader("📤 Upload an image (JPG, PNG, BMP)", type=["jpg", "jpeg", "png", "bmp"])

# Image processing functions
def remove_salt_and_pepper(image, kernel_size=5):
    if len(image.shape) == 2:
        return cv2.medianBlur(image, kernel_size)
    else:
        channels = cv2.split(image)
        filtered = [cv2.medianBlur(ch, kernel_size) for ch in channels]
        return cv2.merge(filtered)

def sharpen_image(image):
    gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

def process_image(image):
    denoised = remove_salt_and_pepper(image)
    sharpened = sharpen_image(denoised)
    return denoised, sharpened

def convert_to_bytes(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# Main logic
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🖼️ Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    with st.spinner("🔄 Processing..."):
        start = time.time()
        denoised, sharpened = process_image(img)
        end = time.time()
        psnr_val = psnr(to_gray(img), to_gray(sharpened))
        ssim_val = ssim(to_gray(img), to_gray(sharpened))

    with col2:
        st.markdown("### 🧽 Denoised")
        st.image(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    with col3:
        st.markdown("### ✨ Sharpened")
        st.image(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    st.markdown("---")
    st.markdown("### 📊 Quality Metrics")
    col_psnr, col_ssim, col_time = st.columns(3)
    col_psnr.metric("🔍 PSNR", f"{psnr_val:.2f}")
    col_ssim.metric("🧠 SSIM", f"{ssim_val:.3f}")
    col_time.metric("⏱️ Time Taken", f"{end - start:.2f} sec")

    st.download_button(
        label="⬇️ Download Processed Image",
        data=convert_to_bytes(sharpened),
        file_name="denoised_sharp.png",
        mime="image/png"
    )
