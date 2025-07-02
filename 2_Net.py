import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class JankenMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        base = mobilenet_v2(pretrained=True)  # 事前学習モデルを使用
        self.features = base.features         # 特徴抽出部分（CNN）

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 特徴を1×1に平均化

        # 出力層：1280次元の特徴ベクトルを共有
        self.classifier = nn.Linear(1280, 4)   # 4クラス分類（グー・チョキ・パー・その他）
        self.regressor = nn.Linear(1280, 1)    # 回帰：指の本数（0〜5）

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)  # (B, 1280)
        class_out = self.classifier(x)        # 分類出力
        reg_out = self.regressor(x).squeeze() # 回帰出力（float）

        return class_out, reg_out
