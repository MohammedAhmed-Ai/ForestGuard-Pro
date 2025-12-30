import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models
from .unet_parts import *
from .attention import CBAM

# --- 1. كلاس الترانسفورمر (كان ناقص) ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # طبقة الانكودر القياسية
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
        # Positional Embedding (عشان يعرف مكان كل بيكسل)
        # 4096 دي تكفي صورة بحجم صغير بعد الضغط (64x64)
        self.pos_embedding = nn.Parameter(torch.randn(1, 4096, embed_dim)) 

    def forward(self, x):
        B, C, H, W = x.shape
        # تحويل الشكل لـ Sequence عشان الترانسفورمر يفهمه
        x = x.flatten(2).transpose(1, 2)
        seq_len = x.shape[1]
        
        # إضافة الـ Positional Embedding (بناخد على قد حجم الصورة)
        if seq_len <= self.pos_embedding.shape[1]:
             x = x + self.pos_embedding[:, :seq_len, :]
        else:
             x = x + self.pos_embedding[:, :self.pos_embedding.shape[1], :]
             
        # الدخول للترانسفورمر
        x = self.transformer_encoder(x)
        
        # إرجاع الشكل لأصله (صورة)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

# --- 2. كلاس الموديل الرئيسي (المعدل لـ 3 كلاسات) ---
class SmokeTransUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(SmokeTransUNet, self).__init__()
        
        # 1. Backbone (ResNet34)
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder1 = resnet.layer1 
        self.encoder2 = resnet.layer2 
        self.encoder3 = resnet.layer3 
        self.encoder4 = resnet.layer4 

        # 2. Transformer Bottleneck
        self.transformer = TransformerBlock(embed_dim=512, num_heads=8)

        # 3. Attention Modules (CBAM)
        self.att1 = CBAM(64)
        self.att2 = CBAM(128)
        self.att3 = CBAM(256)

        # 4. Decoder (Upsampling)
        self.up1 = Up(768, 256) # 512 + 256
        self.up2 = Up(384, 128) # 256 + 128
        self.up3 = Up(192, 64)  # 128 + 64
        self.up4 = Up(128, 64)  # 64 + 64

        # طبقة الخروج (بتطلع 3 قنوات: خلفية، دخان، نار)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        input_size = x.size()[2:] 

        # Encoder Path
        x0 = self.inc(x)      
        x1 = self.encoder1(x0) 
        x2 = self.encoder2(x1) 
        x3 = self.encoder3(x2) 
        x4 = self.encoder4(x3) 

        # Transformer Path
        x4 = self.transformer(x4)

        # Decoder Path with Attention
        x = self.up1(x4, self.att3(x3))
        x = self.up2(x, self.att2(x2))
        x = self.up3(x, self.att1(x1))
        x = self.up4(x, x0)
        
        logits = self.outc(x)
        
        # تكبير الصورة لنفس حجم الدخل الأصلي
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        return logits