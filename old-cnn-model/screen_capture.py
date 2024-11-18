from model import ImageNetwork
import torch
from mss import mss
from PIL import Image
import numpy as np
import time
import cv2
import torchvision.transforms as transforms

from ultralytics import YOLO

model = YOLO("runs/detect/train16/weights/best.onnx")

name_map = [
  "coal",
  "diamond",
  "emerald",
  "gold",
  "iron",
  "lapis",
  "nether_gold_ore",
  "redstone_ore"
]

device = torch.device("cpu")
print(device)
# Set up the image model
# model = ImageNetwork()
# model.load_state_dict(torch.load('model.pth'))
# model.eval()
# model.to(device)
# model.to(device)


transform = transforms.Compose([
      transforms.Resize((350, 350)),
      transforms.ToTensor(),
  ])

mon = {'left': 160, 'top': 160, 'width': 700, 'height': 700}

with mss() as sct:
  while True:
    # time.sleep(0.05) 
    screenshot = sct.grab(mon) 
    img = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)

    # Run the image through the model
    image = transform(img)  
    image = image.unsqueeze(0).to(device)

    predicted = model.predict(img, device=device, show=True)
    # img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # cv2.imshow('test', img_bgr)
    # if cv2.waitKey(33) & 0xFF in (
    #     ord('q'), 
    #     27, 
    # ):
    #     break

    # with torch.no_grad():
    #   output = model(image)
    #   _, predicted = torch.max(output, 1)


      
    # print(f'Predicted class: {name_map[predicted.item()]}')
