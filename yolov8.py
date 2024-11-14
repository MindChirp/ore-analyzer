from roboflow import Roboflow
import os

rf = Roboflow(api_key="xmgiE08gLRF5WUcKt1vn")
project = rf.workspace("main-data").project("minecraft-blocks")
version = project.version(1)
dataset = version.download("yolov8")


# Change the permissions of the directory
# os.chmod('c:\\Users\\frikk\\Documents\\maskinlaering\\oblig10\\Minecraft-Blocks-1', 0o777)

from ultralytics import YOLO

model = YOLO("yolov8s.yaml")

results = model.train(data=f"{ dataset.location }/data.yaml", epochs=5)

results = model.val()

results = model("https://ultralytics.com/images/bus.jpg")

success = model.export(format="onnx")