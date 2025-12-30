import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import cv2
import tempfile
import sys
import os
import time

# إعداد المسارات
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.smoke_net import SmokeTransUNet
import config

st.set_page_config(page_title="ForestGuard AI", page_icon="🌲", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .metric-card {background-color: #262730; padding: 10px; border-radius: 5px;}
    </style>
    """, unsafe_allow_html=True)

# 1. تحميل الموديل الجديد (3 كلاسات)
@st.cache_resource
def load_model():
    # لاحظ: n_classes=3 هنا
    model = SmokeTransUNet(n_channels=3, n_classes=3).to(config.DEVICE)
    # اسم الملف الجديد
    weights_path = "weights/smoke_fire_model.pth" 
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
    else:
        st.error(f"Model weights not found at {weights_path}")
        return None

model = load_model()

# 2. دالة التوقع المعدلة (Softmax + Argmax)
def predict_frame(model, image_pil):
    img_tensor = TF.resize(image_pil, [config.IMAGE_SIZE, config.IMAGE_SIZE], interpolation=Image.BILINEAR)
    img_tensor = TF.to_tensor(img_tensor).unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        output = model(img_tensor) # Shape: [1, 3, H, W]
        # بناخد أعلى احتمال في الـ 3 قنوات
        # 0=Background, 1=Smoke, 2=Fire
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
    return mask

# 3. دالة تلوين الماسك
def colorize_mask(mask):
    # إنشاء صورة فارغة (RGBA)
    h, w = mask.shape
    color_mask = np.zeros((h, w, 4), dtype=np.uint8)

    # تلوين الدخان (رقم 1) -> رمادي
    # R=128, G=128, B=128, Alpha=150
    color_mask[mask == 1] = [128, 128, 128, 150]

    # تلوين النار (رقم 2) -> أحمر فاقع
    # R=255, G=50, B=0, Alpha=180
    color_mask[mask == 2] = [255, 50, 0, 180] 

    return color_mask

# --- الواجهة ---
st.title("🌲 ForestGuard: Advanced Fire & Smoke Segmentation")
st.markdown("### Multi-Class Detection System (Smoke 🌫️ | Fire 🔥)")
st.divider()

st.sidebar.header("Control Panel")
app_mode = st.sidebar.radio("Input Source:", ["🖼️ Upload Image", "🎥 Upload Video"])

if app_mode == "🖼️ Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and model:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        # التوقع
        mask = predict_frame(model, image)
        
        # الحسابات
        smoke_ratio = (np.sum(mask == 1) / mask.size) * 100
        fire_ratio = (np.sum(mask == 2) / mask.size) * 100
        
        # التلوين والعرض
        mask_colored = colorize_mask(mask)
        # تكبير الماسك لحجم الصورة الأصلية للعرض
        mask_colored_pil = Image.fromarray(mask_colored).resize(image.size, resample=Image.NEAREST)
        
        with col2:
            # دمج الصورة الأصلية مع الماسك
            final_overlay = Image.alpha_composite(image.convert("RGBA"), mask_colored_pil)
            st.image(final_overlay, caption="AI Analysis Result", use_column_width=True)
            
        # العدادات
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("🔥 Fire Detected", f"{fire_ratio:.2f}%", delta_color="inverse")
        m2.metric("🌫️ Smoke Detected", f"{smoke_ratio:.2f}%")
        
        if fire_ratio > 0.1:
            st.error("🚨 CRITICAL ALERT: FIRE DETECTED!")
        elif smoke_ratio > 1.0:
            st.warning("⚠️ WARNING: SMOKE DETECTED!")
        else:
            st.success("✅ Area Secure")

elif app_mode == "🎥 Upload Video":
    uploaded_video = st.file_uploader("Upload Video (mp4)", type=["mp4"])
    
    if uploaded_video and model:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        vf = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        kpi_placeholder = st.empty()
        
        stop_btn = st.button("Stop Processing")
        
        while vf.isOpened() and not stop_btn:
            ret, frame = vf.read()
            if not ret:
                break
            
            # تجهيز الفريم
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # التوقع
            mask = predict_frame(model, pil_img)
            
            # تلوين الماسك
            mask_colored = colorize_mask(mask) # RGBA numpy
            mask_resized = cv2.resize(mask_colored, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # دمج بـ OpenCV (AddWeighted)
            # محتاجين نفصل القنوات عشان OpenCV بيتعامل مع BGR
            overlay_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_RGBA2BGR)
            
            # الدمج اليدوي عشان الشفافية
            # (طريقة سريعة: لو البيكسل ملون في الماسك، ناخده، لو لأ، نسيب الأصلي)
            # بس للأسهل هنستخدم addWeighted على الكل
            alpha = 0.6
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # خدع بسيطة للعرض: نلون بس الحتت اللي فيها ماسك
            # Fire Mask Area
            fire_indices = np.where(mask_resized[:, :, 0] == 255) # Red channel
            frame_rgb[fire_indices] = [255, 50, 0] # Color original frame red pixels

            # Smoke Mask Area
            smoke_indices = np.where(mask_resized[:, :, 0] == 128)
            frame_rgb[smoke_indices] = [150, 150, 150]

            stframe.image(frame_rgb, caption="Real-time Analysis", use_column_width=True)
            
            # تحديث العدادات
            f_ratio = (np.sum(mask == 2) / mask.size) * 100
            s_ratio = (np.sum(mask == 1) / mask.size) * 100
            
            status_html = ""
            if f_ratio > 0.1:
                status_html = f"<h2 style='color:red;'>🚨 FIRE: {f_ratio:.1f}%</h2>"
            elif s_ratio > 1.0:
                status_html = f"<h2 style='color:orange;'>⚠️ SMOKE: {s_ratio:.1f}%</h2>"
            else:
                status_html = "<h2 style='color:green;'>✅ SECURE</h2>"
                
            kpi_placeholder.markdown(status_html, unsafe_allow_html=True)
            
        vf.release()