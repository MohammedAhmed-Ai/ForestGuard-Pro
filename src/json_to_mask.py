import os
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# ==========================================
# (1) حط هنا مسار الفولدر "الملخبط" اللي فيه الصور وملفات الـ json
# مثال: r"C:\Users\Ahmed\Downloads\smoke_images_with_annot\raw"
SOURCE_FOLDER = r"D:\smoke_images_with_annot\\raw" 
# ==========================================

# المسارات اللي هننقل ليها (النظيفة)
DEST_IMAGES = "./data/raw/images"
DEST_MASKS = "./data/raw/masks"

def create_masks_from_json():
    # إنشاء الفولدرات
    os.makedirs(DEST_IMAGES, exist_ok=True)
    os.makedirs(DEST_MASKS, exist_ok=True)

    files = os.listdir(SOURCE_FOLDER)
    # ناخد بس ملفات الـ json عشان هي اللي تهمن
    json_files = [f for f in files if f.endswith(".json")]
    
    print(f"Found {len(json_files)} JSON files. Converting to masks...")

    success_count = 0

    for json_file in tqdm(json_files):
        try:
            # 1. قراءة ملف الـ JSON
            json_path = os.path.join(SOURCE_FOLDER, json_file)
            with open(json_path, "r") as f:
                data = json.load(f)

            # 2. تحديد اسم الصورة المرتبطة بالـ JSON
            # عادة بتكون موجودة جوه الـ json في حقل "imagePath"
            # أو بنستنتجها من اسم الملف
            image_filename = data.get("imagePath")
            
            # لو الاسم مش موجود جوه، نستخدم نفس اسم الـ json بس jpg
            if not image_filename:
                image_filename = json_file.replace(".json", ".jpg")
            
            # تصحيح بسيط لو الاسم فيه مسارات غريبة
            image_filename = os.path.basename(image_filename)

            src_img_path = os.path.join(SOURCE_FOLDER, image_filename)
            
            # التأكد إن الصورة الأصلية موجودة
            if not os.path.exists(src_img_path):
                # نجرب ندور عليها بنفس اسم الـ json
                alt_name = json_file.replace(".json", ".jpg")
                src_img_path = os.path.join(SOURCE_FOLDER, alt_name)
                if not os.path.exists(src_img_path):
                    # print(f"Skipping {json_file}, image not found.")
                    continue

            # 3. تجهيز أبعاد الماسك (نفس أبعاد الصورة الأصلية)
            img_height = data.get("imageHeight")
            img_width = data.get("imageWidth")
            
            # لو الأبعاد مش مكتوبة، نفتح الصورة نجيب أبعادها
            if not img_height or not img_width:
                with Image.open(src_img_path) as temp_img:
                    img_width, img_height = temp_img.size

            # 4. رسم الماسك
            # بنعمل صورة سوداء تماماً
            mask = Image.new("L", (img_width, img_height), 0)
            draw = ImageDraw.Draw(mask)

            # نلف على كل الأشكال (shapes) المرسومة جوه الـ json
            for shape in data["shapes"]:
                label = shape["label"]
                points = shape["points"] # الإحداثيات
                
                # تحويل النقاط لشكل (tuples) عشان الرسم
                polygon = [tuple(point) for point in points]
                
                # رسم الشكل باللون الأبيض (255)
                draw.polygon(polygon, outline=1, fill=255)

            # 5. الحفظ والنقل
            # حفظ الماسك بصيغة PNG
            mask_filename = os.path.splitext(image_filename)[0] + ".png"
            mask.save(os.path.join(DEST_MASKS, mask_filename))

            # نسخ الصورة الأصلية
            shutil.copy(src_img_path, os.path.join(DEST_IMAGES, image_filename))

            success_count += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print(f"✅ تمت العملية! تم استخراج {success_count} صورة وماسك.")
    print(f"Images: {DEST_IMAGES}")
    print(f"Masks: {DEST_MASKS}")

if __name__ == "__main__":
    create_masks_from_json()