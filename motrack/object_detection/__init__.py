"""
Object detection interface.
"""
from motrack.object_detection.factory import object_detection_inference_factory
from motrack.object_detection.algorithms.base import ObjectDetectionInference
from motrack.object_detection.algorithms.yolox import YOLOXInference
from motrack.object_detection.algorithms.yolov8 import YOLOv8Inference
from motrack.object_detection.manager import DetectionManager
