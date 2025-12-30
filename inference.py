import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import torchvision.transforms.functional as TF

from src.models.smoke_net import SmokeTransUNet
import config

def load_checkpoint(checkpoint_file, model):
    print("=> Loading checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval() # وضع الاختبار (بيوقف التحديث)

def predict_image(model, image_path, save_path="result.png"):
    # 1. قراءة الصورة
    original_img = Image.open(image_path).convert("RGB")
    
    # 2. تجهيز الصورة للموديل (نفس اللي عملناه في التدريب)
    x = TF.resize(original_img, [config.IMAGE_SIZE, config.IMAGE_SIZE])
    x = TF.to_tensor(x)
    x = x.unsqueeze(0).to(config.DEVICE) # إضافة بعد الـ Batch

    # 3. التوقع
    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float() # تحويل لـ 0 و 1
    
    # 4. تحويل النتيجة لصورة عشان نشوفها
    # بنشيل البعد الزيادة ونحولها لـ Numpy
    mask = preds.squeeze().cpu().numpy()
    
    # 5. الرسم (Matplotlib)
    # هنعرض الصورة الأصلية جنب الماسك اللي الموديل توقعه
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(original_img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))
    ax1.set_title("Original Image")
    ax1.axis("off")
    
    ax2.imshow(mask, cmap="gray")
    ax2.set_title("AI Prediction (Smoke Mask)")
    ax2.axis("off")
    
    plt.savefig(save_path)
    print(f"✅ Result saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # 1. بناء الموديل
    model = SmokeTransUNet(n_channels=3, n_classes=1).to(config.DEVICE)
    
    # 2. تحميل الأوزان (اللي دربناها)
    weights_path = config.MODEL_SAVE_PATH
    if os.path.exists(weights_path):
        load_checkpoint(weights_path, model)
    else:
        print(f"❌ Error: No weights found at {weights_path}")
        exit()

    # 3. اختيار صورة عشوائية للتجربة
    images_dir = config.DATA_DIR + "/raw/images"
    all_images = os.listdir(images_dir)
    random_image = random.choice(all_images)
    img_path = os.path.join(images_dir, random_image)
    
    print(f"Testing on image: {random_image}")
    predict_image(model, img_path)