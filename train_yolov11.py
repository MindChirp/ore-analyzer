if __name__ == "__main__":
  from ultralytics import YOLO
  from roboflow import Roboflow
  import torch

  print("CUDA available:", torch.cuda.is_available())
  print("CUDA device count:", torch.cuda.device_count())

  from roboflow import Roboflow
  rf = Roboflow(api_key="xmgiE08gLRF5WUcKt1vn")
  project = rf.workspace("oblig10").project("minecraft-ore-detection-20pzg-xejw6")
  version = project.version(2)
  dataset = version.download("yolov11")

  model = YOLO('yolo11n.pt')
  results = model.train(data=f"{ dataset.location }/data.yaml", epochs=10, workers=0)
  # test = model("/test/diamond_ore/diamant.png")