## 应用model去预测picture的label

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms

import os
import numpy as np
from PIL import Image


# BuildingParamNet 是您之前定义的模型类
class BuildingParamNet(nn.Module):
    def __init__(self, output_dim=4):
        super(BuildingParamNet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 移除最后的全连接层
        
        self.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)  # 输出维度为 4
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BuildingParamNet(output_dim=4)
model.load_state_dict(torch.load('/home/sjtu_dzn/Project/model/cnn.pth', map_location=device))
model.to(device).eval()  # 设置为评估模式并移动到正确的设备

# 定义预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

# 加载图像
image_path = '/home/sjtu_dzn/Project/data/easy/test_dataset/1.jpg'  # 替换为你的图片路径
image = Image.open(image_path).convert('RGB')  # 确保是RGB格式

# 应用预处理
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0).to(device)  # 创建一个批次并移动到正确的设备

# 进行预测
with torch.no_grad():
    output = model(input_batch)
    predictions = output.squeeze()  # 移除批次维度

# 将预测结果转换为CPU张量并转换为列表
predicted_values = predictions.cpu().tolist()

# 打印预测结果
print("Predicted 4x1 vector:", predicted_values)

# 将预测结果转化为可以用UE三维重建的label
label = [round(x) for x in predicted_values]
label[1] = f"B{label[1]}"
label[2] = f"D{label[2]}"
label[3] = f"W{label[3]}"

print("label:", label)
