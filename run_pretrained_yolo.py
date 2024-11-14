from inference import get_model
import numpy as np
from mss import mss
import supervision as sv
import cv2
import time
from PIL import Image

mon = {'left': 160, 'top': 160, 'width': 1000, 'height': 1080}

model = get_model(model_id="minecraft-blocks/1", api_key="xmgiE08gLRF5WUcKt1vn")



# create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# load the results into the supervision Detections api

with mss() as sct:
  while True:
    # time.sleep(0.05) 
    screenshot = sct.grab(mon) 
    img = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)

    results = model.infer(img)[0]
    detections = sv.Detections.from_inference(results)

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
      scene=img, detections=detections)
    annotated_image = label_annotator.annotate(
      scene=annotated_image, detections=detections)
    
    img_bgr = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
    cv2.imshow('test', img_bgr)
    if cv2.waitKey(33) & 0xFF in (
        ord('q'), 
        27, 
    ):
        break
    # sv.plot_image(annotated_image)






