import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import os

# --- 損失関数（分類 + 回帰） --- #
def multitask_loss_fn(class_logits, reg_output, class_labels, finger_counts, alpha=1.0, beta=1.0):
    loss_cls = F.cross_entropy(class_logits, class_labels)
    loss_reg = F.mse_loss(reg_output, finger_counts.float())
    return alpha * loss_cls + beta * loss_reg, loss_cls, loss_reg

# --- 学習ループ --- #
def train_model(model, train_dataset, num_epochs=10, batch_size=32, lr=1e-4, save_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_cls, total_reg = 0, 0, 0

        for images, class_labels, finger_counts in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            class_labels = class_labels.to(device)
            finger_counts = finger_counts.to(device).float()

            optimizer.zero_grad()
            class_logits, reg_output = model(images)

            loss, loss_cls, loss_reg = multitask_loss_fn(class_logits, reg_output, class_labels, finger_counts)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_reg += loss_reg.item()

        avg_loss = total_loss / len(dataloader)
        avg_cls = total_cls / len(dataloader)
        avg_reg = total_reg / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Cls: {avg_cls:.4f} | Reg: {avg_reg:.4f}")

    # モデル保存
    torch.save(model.state_dict(), save_path)
    print(f"✅ モデルを保存しました: {save_path}")

from dataset import JankenDataset
from model import JankenMobileNetV2

if __name__ == "__main__":
    dataset = JankenDataset("janken_labels.csv")  # CSVを指定
    model = JankenMobileNetV2()
    train_model(model, dataset, num_epochs=10)

