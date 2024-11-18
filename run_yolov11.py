from gymnasium import spaces
import torch
from mss import mss
from PIL import Image
import numpy as np
import time
import cv2
import torchvision.transforms as transforms

from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("runs/detect/train44/weights/best.pt").to(device)
print(device)

name_map = [ 
 'Deepslate Diamond Ore', 'Deepslate Gold Ore', 'Deepslate Iron Ore', 'Deepslate RedstoneOre', 'Iron Ore' 
]


mon = {'left': 160, 'top': 160, 'width': 700, 'height': 700}

with mss() as sct:
  while True:
    time.sleep(1) 
    screenshot = sct.grab(mon) 
    img = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)

    # show=True displays a window with the annotated image
    predicted = model.predict(img, device=device, show=True)
    ores = np.zeros(len(name_map))

    for detection in predicted:
      boxes = detection.boxes
      for box in boxes:
        ores[int(box.cls[0])] = 1


