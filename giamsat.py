import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
import cv2
model = YOLO("best.pt")
image_count = 0


box_annotator = sv.BoundingBoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    global image_count
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    
    
    labels = [
    model.model.names[class_id]
    for class_id in detections.class_id
]
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    
    
        
    for xyxy in detections.xyxy:
        cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
            
        image_name = f"crop_{image_count}.png"
        image_path = os.path.join('out', image_name)
        cv2.imwrite(image_path, cropped_image)
        image_count += 1
    return frame

sv.process_video(
    source_path="input/hardhat.mp4",
    target_path="out/out3.mp4",
    callback=callback
)