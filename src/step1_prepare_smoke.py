import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# --- مسارات ---
# المصدر (القديم)
RAW_IMG_DIR = "data/raw/images"
RAW_MASK_DIR = "data/raw/masks"

# الهدف (الجديد)
DEST_IMG_DIR = "data/processed/images"
DEST_MASK_DIR = "data/processed/masks"

# التأكد من وجود الفولدرات
os.makedirs(DEST_IMG_DIR, exist_ok=True)
os.makedirs(DEST_MASK_DIR, exist_ok=True)

def main():
    print("🚀 Step 1: Processing Original Smoke Data...")
    
    images = os.listdir(RAW_IMG_DIR)
    
    for img_name in tqdm(images):
        # 1. تحديد المسارات
        src_img_path = os.path.join(RAW_IMG_DIR, img_name)
        
        # الماسك غالباً نفس الاسم بس png
        mask_name = os.path.splitext(img_name)[0] + ".png"
        src_mask_path = os.path.join(RAW_MASK_DIR, mask_name)
        
        # لو الماسك مش موجود (مشكلة في التسمية مثلاً)، تخطاه
        if not os.path.exists(src_mask_path):
            # محاولة أخيرة: ممكن يكون jpg في فولدر الماسك؟ (نادراً)
            src_mask_path = os.path.join(RAW_MASK_DIR, img_name)
            if not os.path.exists(src_mask_path):
                continue

        # 2. نقل الصورة كما هي
        shutil.copy(src_img_path, os.path.join(DEST_IMG_DIR, img_name))
        
        # 3. معالجة الماسك (أهم خطوة)
        # بنقرأ الماسك، ونخلي أي حاجة بيضا (255) تبقى قيمتها (1)
        mask = cv2.imread(src_mask_path, 0) # Grayscale
        
        new_mask = np.zeros_like(mask)
        new_mask[mask > 100] = 1  # Class 1 = Smoke
        
        # حفظ الماسك الجديد
        cv2.imwrite(os.path.join(DEST_MASK_DIR, mask_name), new_mask)

    print(f"✅ Step 1 Done. Moved {len(os.listdir(DEST_IMG_DIR))} smoke images.")

if __name__ == "__main__":
    main()