import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # تفعيل Sigmoid عشان نحول الأرقام لاحتمالات (بين 0 و 1)
        inputs = torch.sigmoid(inputs)       
        
        # فرد الصور لصف واحد (Flatten) عشان نعرف نحسب المعادلات
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # حساب التقاطع (Intersection)
        intersection = (inputs * targets).sum()                            
        
        # معادلة Dice Score
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        # الـ Loss هو عكس الـ Score (يعني 1 - الدقة)
        return 1 - dice