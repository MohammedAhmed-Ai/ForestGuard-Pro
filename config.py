import torch

# المسارات
DATA_DIR = "data"
MODEL_SAVE_PATH = "weights/smoke_fire_model.pth" # غيرنا الاسم

# إعدادات التدريب
LEARNING_RATE = 1e-4
BATCH_SIZE = 2      # خليها 2 عشان الرامات تستحمل
NUM_EPOCHS = 15     # 15 كفاية جداً للداتا دي
NUM_WORKERS = 0
IMAGE_SIZE = 256
PIN_MEMORY = True
LOAD_MODEL = False

# الإعدادات الجديدة
NUM_CLASSES = 3  # (0=Background, 1=Smoke, 2=Fire)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"