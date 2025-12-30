import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

# المصدر (D-Fire)
DFIRE_ROOT = "data/D-Fire"

# الهدف (processed)
OUTPUT_IMG_DIR = "data/processed/images"
OUTPUT_MASK_DIR = "data/processed/masks"

def create_fire_mask_from_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # مدى ألوان النار (أصفر لبرتقالي لأحمر)
    lower1 = np.array([18, 50, 50])
    upper1 = np.array([35, 255, 255])
    lower2 = np.array([0, 100, 100])
    upper2 = np.array([15, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    fire_mask = cv2.bitwise_or(mask1, mask2)
    
    # تنظيف
    kernel = np.ones((3,3), np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fire_mask = cv2.dilate(fire_mask, kernel, iterations=1)
    
    return fire_mask

def process_subset(subset_name):
    # بيمشي في train و test
    search_path = os.path.join(DFIRE_ROOT, subset_name, "images", "*.jpg")
    image_files = glob.glob(search_path)
    
    print(f"   -> Scanning {subset_name}: found {len(image_files)} images.")
    
    saved_count = 0
    for img_path in tqdm(image_files):
        img = cv2.imread(img_path)
        if img is None: continue
        
        # استخراج النار
        fire_mask = create_fire_mask_from_color(img)
        
        # لو مفيش نار (أقل من 50 بيكسل)، تجاهل الصورة
        if np.sum(fire_mask) < 50: 
            continue

        # تجهيز الماسك بقيمة 2
        final_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        final_mask[fire_mask > 0] = 2  # Class 2 = Fire

        # الحفظ
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        new_name = f"dfire_{subset_name}_{filename}"
        new_mask_name = f"dfire_{subset_name}_{name_no_ext}.png"
        
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, new_name), img)
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, new_mask_name), final_mask)
        saved_count += 1
        
    return saved_count

def main():
    print("🚀 Step 3: Extracting Fire from D-Fire...")
    
    if not os.path.exists(DFIRE_ROOT):
        print("❌ Error: D-Fire folder not found!")
        return

    c1 = process_subset("train")
    c2 = process_subset("test")
    
    print(f"✅ Step 3 Done. Added {c1 + c2} Fire images.")
    print(f"🔥 FINAL DATASET SIZE: {len(os.listdir(OUTPUT_IMG_DIR))} Images.")

if __name__ == "__main__":
    main()