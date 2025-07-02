import torch
from PIL import Image
import torchvision.transforms as T
from model import JankenMobileNetV2  # モデル定義ファイル
import sys

# --- 推論用の前処理 --- #
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
])

# --- クラス名 --- #
class_names = ['グー', 'チョキ', 'パー', 'それ以外']

# --- 推論関数 --- #
def predict(image_path, model_path="model.pth"):
    # モデル準備
    model = JankenMobileNetV2()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 画像読み込み・前処理
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # (1, C, H, W)

    # 推論
    with torch.no_grad():
        class_logits, reg_output = model(input_tensor)
        class_pred = torch.argmax(class_logits, dim=1).item()
        reg_pred = reg_output.item()

    # 結果表示
    print(f"🧠 予測結果：{class_names[class_pred]}")
    print(f"✌️ 指の本数（回帰）：{reg_pred:.2f}")

# --- 実行例 --- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使い方: python predict.py 画像ファイルパス")
    else:
        predict(sys.argv[1])
