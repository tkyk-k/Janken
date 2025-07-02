import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T

class JankenDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # CSVを読み込む（image_path, class_label, finger_count）
        self.data = pd.read_csv(csv_file)
        self.transform = transform or T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        class_label = int(row['class_label'])       # 0〜3
        finger_count = float(row['finger_count'])   # 0〜5

        image = Image.open(img_path).convert('RGB')  # 画像読み込み

        if self.transform:
            image = self.transform(image)

        return image, class_label, finger_count