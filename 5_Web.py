import cv2
import torch
from model import JankenMobileNetV2
import torchvision.transforms as T
import numpy as np

# モデル読み込み（CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = JankenMobileNetV2()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# 前処理
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor(),
])

class_names = ['グー', 'チョキ', 'パー', 'それ以外']

cap = cv2.VideoCapture(0)  # カメラ開始

if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    # 画面中央128x128を切り出し（四角を描画）
    x1, y1 = cx - 64, cy - 64
    x2, y2 = cx + 64, cy + 64
    roi = frame[y1:y2, x1:x2]

    # 推論用前処理
    input_tensor = transform(roi).unsqueeze(0).to(device)

    with torch.no_grad():
        class_logits, reg_output = model(input_tensor)
        class_id = torch.argmax(class_logits, dim=1).item()
        finger_count = reg_output.item()

    # 結果テキスト描画
    text1 = f"判定: {class_names[class_id]}"
    text2 = f"指の数: {finger_count:.2f}"
    cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # 四角を描画
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 画面表示
    cv2.imshow("Janken Real-time", frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
