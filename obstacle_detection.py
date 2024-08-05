import cv2
import numpy as np
import torch

from models import (
    DEPTH_ESTIMATION_MODEL,
    DEVICE,
    OBSTACLE_DETECTION_MODEL,
    TRANSFORMER
)

CONFIDENCE_THRESHOLD=0.7
DEPTH_THRESHOLD = 5 


def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    transformed_frame = TRANSFORMER(frame).to(DEVICE)
    boxes, classes, names, confidences = detect_obstacles(transformed_frame)
    depth_map = estimate_depth(frame, transformed_frame)

    objects_distances_list = []

    for box, cls, confidence in zip(boxes, classes, confidences):
        x_min, y_min, x_max, y_max = box
        object_depth = calculate_object_depth(confidence, depth_map, x_min, y_min, x_max, y_max)
        if object_depth > DEPTH_THRESHOLD:
            objects_distances_list.append({
                "name": names[cls],
                "distance": object_depth
            })

    return objects_distances_list
            

def detect_obstacles(frame):
    results =  OBSTACLE_DETECTION_MODEL(frame,imgsz=640)
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    return boxes, classes, names, confidences


def estimate_depth(frame, transformed_frame):
    with torch.no_grad():
        prediction = DEPTH_ESTIMATION_MODEL(transformed_frame)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    return depth_map
    

def calculate_object_depth(confidence, depth_map, x_min, y_min, x_max, y_max):
    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)

    if confidence >= CONFIDENCE_THRESHOLD:
        depth_values = depth_map[y_min:y_max, x_min:x_max]  

        object_depth = np.mean(depth_values) if depth_values.size > 0 else 0  

        return object_depth
    else:
        return 0