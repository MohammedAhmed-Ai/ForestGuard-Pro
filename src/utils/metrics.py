import torch

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval() # وضع الاختبار

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            
            # الموديل يتوقع
            preds = torch.sigmoid(model(x))
            
            # تحويل الاحتمالات لـ 0 أو 1 (أبيض وأسود)
            preds = (preds > 0.5).float()
            
            # حساب الدقة
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            
            # حساب Dice Score (قريب من IoU)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice Score: {dice_score/len(loader)}")
    
    model.train() # نرجع لوضع التدريب
    return dice_score/len(loader)