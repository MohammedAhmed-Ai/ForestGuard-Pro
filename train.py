import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.dataset import SmokeDataset
from src.transforms import SmokeTransforms
from src.models.smoke_net import SmokeTransUNet
# شيلنا DiceLoss لأننا هنستخدم CrossEntropy الجاهزة بتاعت بايثون
import config

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(config.DEVICE)
        # Targets لازم تكون Long (أعداد صحيحة) وشيلنا unsqueeze
        targets = targets.to(config.DEVICE)

        # Forward
        predictions = model(data) # Shape: (Batch, 3, H, W)
        
        # Loss calculation
        loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            preds = model(x) # (Batch, 3, H, W)
            # بناخد أعلى قيمة في الـ 3 قنوات (argmax)
            preds = torch.argmax(preds, dim=1) 
            
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    model.train()

def main():
    if not os.path.exists("weights"):
        os.makedirs("weights")

    transforms = SmokeTransforms(is_train=True)

    train_dataset = SmokeDataset(
        images_dir=config.DATA_DIR + "/processed/images",
        masks_dir=config.DATA_DIR + "/processed/masks",
        transform=transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # الموديل بـ 3 كلاسات
    model = SmokeTransUNet(n_channels=3, n_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    # دالة الخسارة للمالتي كلاس
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"--- Starting Multi-Class Training (Smoke & Fire) on {config.DEVICE} ---")

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
        train_fn(train_loader, model, optimizer, loss_fn)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        
        if (epoch+1) % 1 == 0: # Check every epoch because dataset is large
            check_accuracy(train_loader, model, device=config.DEVICE)
            torch.save(checkpoint, config.MODEL_SAVE_PATH)
            print(f"Model saved!")

if __name__ == "__main__":
    main()