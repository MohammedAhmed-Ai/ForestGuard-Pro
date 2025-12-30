import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class SmokeDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.images_dir, img_name)
        
        # الماسك نفس الاسم بس png
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        # قراءة الصورة
        image = Image.open(img_path).convert("RGB")
        
        # قراءة الماسك (Grayscale)
        # القيم جواه هي فعلياً 0 و 1 و 2 لأننا حفظناها كده
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask