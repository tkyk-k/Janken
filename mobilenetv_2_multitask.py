# --- MobileNetV2ベースのマルチタスクモデル（PyTorch） ---
# 分類（4クラス：グー・チョキ・パー・それ以外） + 回帰（指の本数）

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# --- 1. モデル定義 --- #
class JankenMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        # MobileNetV2の特徴抽出部を使用（学習済み）
        base = mobilenet_v2(pretrained=True)
        self.features = base.features  # 特徴抽出部（畳み込み層）
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 出力特徴量を1x1に平均プーリング

        # 出力層：1280次元特徴を入力として
        self.classifier = nn.Linear(1280, 4)  # 4クラス分類（グー・チョキ・パー・それ以外）
        self.regressor = nn.Linear(1280, 1)   # 回帰（指の本数を予測）

    def forward(self, x):
        x = self.features(x)                      # 特徴抽出
        x = self.pool(x).view(x.size(0), -1)      # プーリング後ベクトル化 (B, 1280)
        class_out = self.classifier(x)            # 分類出力
        reg_out = self.regressor(x).squeeze()     # 回帰出力（1次元）
        return class_out, reg_out


# --- 2. 損失関数（分類＋回帰） --- #
def multitask_loss_fn(class_logits, reg_output, class_labels, finger_counts, alpha=1.0, beta=1.0):
    # クロスエントロピー損失（分類）
    loss_cls = F.cross_entropy(class_logits, class_labels)
    # 平均二乗誤差損失（回帰）
    loss_reg = F.mse_loss(reg_output, finger_counts.float())
    # 損失の合計（重み係数つき）
    return alpha * loss_cls + beta * loss_reg, loss_cls, loss_reg


# --- 3. ダミーデータセット定義 --- #
class DummyJankenDataset(Dataset):
    def __init__(self, length=100):
        self.length = length
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.randn(3, 128, 128)                 # ランダムなダミー画像
        class_label = torch.randint(0, 4, (1,)).item()   # クラスラベル（0〜3）
        finger_count = torch.randint(0, 6, (1,)).item()  # 指の本数（0〜5）
        return image, class_label, finger_count


# --- 4. 学習関数 --- #
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_cls, total_reg = 0, 0, 0

    for images, class_labels, finger_counts in dataloader:
        images = images.to(device)
        class_labels = class_labels.to(device)
        finger_counts = finger_counts.to(device).float()

        optimizer.zero_grad()
        class_logits, reg_output = model(images)  # 順伝播
        loss, loss_cls, loss_reg = multitask_loss_fn(class_logits, reg_output, class_labels, finger_counts)

        loss.backward()       # 誤差逆伝播
        optimizer.step()      # パラメータ更新

        # 損失の合計
        total_loss += loss.item()
        total_cls += loss_cls.item()
        total_reg += loss_reg.item()

    # 各損失の平均を返す
    return total_loss / len(dataloader), total_cls / len(dataloader), total_reg / len(dataloader)


# --- 5. メイン実行部（テスト用） --- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JankenMobileNetV2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ダミーデータセットとデータローダー作成
    dataset = DummyJankenDataset(length=200)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 簡単な学習ループ（5エポック）
    for epoch in range(5):
        loss, loss_cls, loss_reg = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}: Loss={loss:.4f} | Cls={loss_cls:.4f} | Reg={loss_reg:.4f}")
