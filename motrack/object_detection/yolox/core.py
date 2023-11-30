"""
Support for YOLOX inference (only pretrained models)
"""
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch

from motrack.object_detection.yolox.yolox_x import DEFAULT_EXP_PATH, DEFAULT_EXP_NAME


class YOLOXPredictor:
    """
    YOLOX predictor (inference) used for already trained model (loaded from checkpoint).
    """
    def __init__(
        self,
        checkpoint_path: str,
        accelerator: str,
        legacy: bool = True,
        conf_threshold: Optional[float] = None,
        exp_path: str = DEFAULT_EXP_PATH,
        exp_name: str = DEFAULT_EXP_NAME
    ):
        """
        Args:
            checkpoint_path: Checkpoint of the pretrained model.
            accelerator: Accelerator
            legacy: Optionally use legacy preprocess
            exp_path: Experiment path (Path to the python script that defines experiments)
            exp_name: Experiment name (Name of the class in the python script that defines experiments)
        """
        from yolox.data.data_augment import ValTransform
        from yolox.exp import get_exp

        self._accelerator = accelerator

        # Load Exp
        self._exp = get_exp(exp_path, exp_name)
        self._conf_threshold = self._exp.test_conf if conf_threshold is None else conf_threshold

        # Load model
        ckpt = torch.load(checkpoint_path)

        model = self._exp.get_model()
        model.load_state_dict(ckpt['model'])
        model.to(self._accelerator)
        model.eval()
        self._model = model

        # Setup preprocess
        self._preprocess = ValTransform(legacy=legacy)

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Performs inference on a simple image.

        Args:
            img: Raw image

        Returns:
            - Image bboxes in format: (x1, y1, x2, y1, obj_conf * class_conf, class_conf, class_pred)
            - Image info (used for visualization)
        """
        from yolox.utils import postprocess

        img = img.copy()

        # Store image info
        img_info = {'id': 0}
        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        img_info['raw_img'] = img

        ratio = min(self._exp.test_size[0] / img.shape[0], self._exp.test_size[1] / img.shape[1])
        img_info['ratio'] = ratio

        # Preprocess image
        img, _ = self._preprocess(img, None, self._exp.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float()
        img = img.to(self._accelerator)

        # Inference
        outputs = self._model(img)
        outputs = outputs.detach().cpu()

        # Postprocess
        outputs = postprocess(outputs, self._exp.num_classes, self._conf_threshold, self._exp.nmsthre, class_agnostic=True)
        output = outputs[0]
        if output is None:
            # Nothing detected
            return np.empty(shape=(0, 7), dtype=np.float32), img_info

        output = output.numpy()
        output[:, 0:4] /= ratio
        output[:, 4] *= output[:, 5]

        # Output format : (x1, y1, x2, y2, obj_conf, class_conf, class_label)
        return output, img_info

    def visualize(self, img: np.ndarray, output: np.ndarray, cls_conf: float = 0.10) -> np.ndarray:
        """
        Visualizes model output on the same image.

        Args:
            img: Same image as in the inference
            output: Model outputs for the same image
            cls_conf: Class confidence threshold

        Returns:
            Image with visualizations
        """
        from yolox.utils import vis

        bboxes = output[:, 0:4]
        cls = output[:, 6]
        scores = output[:, 4]
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self._exp.class_names)

        return vis_res
