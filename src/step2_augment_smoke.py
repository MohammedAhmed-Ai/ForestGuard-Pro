import os
import cv2
import numpy as np
from tqdm import tqdm

# بنشتغل على الداتا اللي بقت في processed خلاص
IMAGES_DIR = "data/processed/images"
MASKS_DIR = "data/processed/masks"

def main():
    print("🚀 Step 2: Augmenting Smoke Data (Balancing)...")
    
    # قائمة الصور الحالية
    image_files = os.listdir(IMAGES_DIR)
    
    count = 0
    for img_name in tqdm(image_files):
        mask_name = os.path.splitext(img_name)[0] + ".png"
        
        img_path = os.path.join(IMAGES_DIR, img_name)
        mask_path = os.path.join(MASKS_DIR, mask_name)
        
        if not os.path.exists(mask_path): continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0) # قراءة القيم (0 و 1)

        # 1. Flip Horizontal
        img_h = cv2.flip(img, 1)
        mask_h = cv2.flip(mask, 1)
        
        cv2.imwrite(os.path.join(IMAGES_DIR, f"aug_h_{img_name}"), img_h)
        cv2.imwrite(os.path.join(MASKS_DIR, f"aug_h_{mask_name}"), mask_h)
        count += 1

        # 2. Flip Vertical
        img_v = cv2.flip(img, 0)
        mask_v = cv2.flip(mask, 0)
        
        cv2.imwrite(os.path.join(IMAGES_DIR, f"aug_v_{img_name}"), img_v)
        cv2.imwrite(os.path.join(MASKS_DIR, f"aug_v_{mask_name}"), mask_v)
        count += 1
        
        # 3. Rotate 90
        img_r = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask_r = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        
        cv2.imwrite(os.path.join(IMAGES_DIR, f"aug_r_{img_name}"), img_r)
        cv2.imwrite(os.path.join(MASKS_DIR, f"aug_r_{mask_name}"), mask_r)
        count += 1

    print(f"✅ Step 2 Done. Created {count} extra smoke images.")
    print(f"Total Images now: {len(os.listdir(IMAGES_DIR))}")

if __name__ == "__main__":
    main()