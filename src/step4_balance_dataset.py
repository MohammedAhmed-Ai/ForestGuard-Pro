import os
import random
import glob
from tqdm import tqdm

# المسارات
IMG_DIR = "data/processed/images"
MASK_DIR = "data/processed/masks"

def main():
    print("⚖️ Step 4: Balancing Dataset (Downsampling Fire)...")

    # 1. حصر الملفات
    all_images = os.listdir(IMG_DIR)
    
    smoke_imgs = [f for f in all_images if not f.startswith("dfire_")]
    fire_imgs = [f for f in all_images if f.startswith("dfire_")]

    n_smoke = len(smoke_imgs)
    n_fire = len(fire_imgs)

    print(f"   -> Found Smoke: {n_smoke}")
    print(f"   -> Found Fire:  {n_fire}")

    # إحنا عايزين النار تكون قد الدخان تقريباً (أو أكتر سنة بسيطة)
    # هنخلي التارجت 1500 صورة نار (عشان يبقى المجموع حوالي 2800)
    target_fire = 1500

    if n_fire > target_fire:
        print(f"   -> Too many fire images! Deleting {n_fire - target_fire} images...")
        
        # ترتيب عشوائي عشان نمسح عشوائي ونحافظ على التنوع
        random.shuffle(fire_imgs)
        
        # الصور اللي هنمسحها (من بعد الـ 1500)
        imgs_to_delete = fire_imgs[target_fire:]
        
        for img_name in tqdm(imgs_to_delete):
            # مسار الصورة والماسك
            img_path = os.path.join(IMG_DIR, img_name)
            
            mask_name = os.path.splitext(img_name)[0] + ".png"
            mask_path = os.path.join(MASK_DIR, mask_name)
            
            # حذف
            if os.path.exists(img_path): os.remove(img_path)
            if os.path.exists(mask_path): os.remove(mask_path)
            
        print(f"✅ Deleted extra fire images. Remaining Fire: {target_fire}")
    else:
        print("✅ Fire count is acceptable. No deletion needed.")

    # الإحصائية النهائية
    final_total = len(os.listdir(IMG_DIR))
    print(f"📊 FINAL DATASET READY: {final_total} Images.")
    print("   Ready for Multi-Class Training!")

if __name__ == "__main__":
    main()