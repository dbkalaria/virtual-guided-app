import torch
from ultralytics import YOLO

DEPTH_ESTIMATION_MODEL_LIST = {
    "s": "MiDaS_small",
    "m": "DPT_Hybrid",
    "l": "DPT_Large"
}

DEPTH_ESTIMATION_MODEL_TYPE = DEPTH_ESTIMATION_MODEL_LIST["m"] 

OBSTACLE_DETECTION_MODEL = YOLO('yolov8l.pt')
DEPTH_ESTIMATION_MODEL = torch.hub.load("intel-isl/MiDaS", DEPTH_ESTIMATION_MODEL_TYPE)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DEPTH_ESTIMATION_MODEL.to(DEVICE)
DEPTH_ESTIMATION_MODEL.eval()

DEPTH_ESTIMATION_TRANSFORMER = torch.hub.load("intel-isl/MiDaS", "transforms")

TRANSFORMER = DEPTH_ESTIMATION_TRANSFORMER.small_transform \
    if DEPTH_ESTIMATION_MODEL_TYPE == DEPTH_ESTIMATION_MODEL_LIST['s'] \
    else DEPTH_ESTIMATION_TRANSFORMER.dpt_transform
    
