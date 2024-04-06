import numpy as np
import torch
#from torchvision import models


def load_model(model_repo: str, model_weights_path: str, conf: float, iou: float): #-> models.common.AutoShape:
    model = torch.hub.load(
        repo_or_dir=model_repo,
        model='custom',
        path=model_weights_path,
        source='local',
        force_reload=True
    )
    model.conf = conf  # NMS confidence threshold
    model.iou = iou  # NMS IoU threshold
    model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image
    return model


def predict(model, img: np.ndarray, size: int, augment: bool = False):
    results = model(img, size=size, augment=augment)
    predictions = results.pandas().xyxy[0]
    bboxes = predictions[['xmin', 'ymin', 'xmax', 'ymax']].values

    if len(bboxes):
        confs = predictions.confidence.values
        return bboxes, confs
    else:
        return [], []


def convert_coco_labels_2_yolo_labels(box: list, label: int, image_height: int, image_width: int) -> list:
    """
    Конвертирует метки в формате coco в формат yolo
    :param box: вектор координат в формате coco
    :param label: метка класса
    :param image_height: высота картинки
    :param image_width: ширина картинки
    :return: преобразованный вектор координат в формате yolo
    """
    # [xmin, ymin, w, h]  => [xc, yc, w, h] (normalized)
    yolo_bbox = [
        label,
        (box[0] + box[2] / 2) / image_width,
        (box[1] + box[3] / 2) / image_height,
        box[2] / image_width,
        box[3] / image_height,
    ]
    yolo_format = [min(bbox_item, 1) for bbox_item in yolo_bbox]
    return yolo_format
