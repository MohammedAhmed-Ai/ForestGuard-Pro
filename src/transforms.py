import torch
import torchvision.transforms.functional as TF
import random
import numpy as np
import config
from PIL import Image

class SmokeTransforms:
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.size = config.IMAGE_SIZE

    def __call__(self, image, mask):
        # image: PIL Image
        # mask: PIL Image (Mode L) -> قيمها 0, 1, 2

        # 1. Resize (Nearest للماسك عشان القيم متسيحش)
        image = TF.resize(image, [self.size, self.size], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size], interpolation=Image.NEAREST)

        # 2. Augmentation
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # 3. التحويل لـ Tensor
        # الصورة بتتقسم على 255 وتبقى Float
        image = TF.to_tensor(image)
        
        # الماسك بيفضل زي ما هو أرقام صحيحة (Long) وميتقسمش على 255
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        
        return image, mask