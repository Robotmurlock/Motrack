"""
mmdetection-based YOLOX inference registered to the motrack object detection catalog.

Supports:
  - Standard mmdetection checkpoints (.pth)
  - Legacy ByteTrack / original Megvii YOLOX checkpoints (.pth.tar) via key remapping
"""
from typing import Any, Optional, Tuple

import numpy as np
import torch

from motrack.object_detection.algorithms.base import ObjectDetectionInference
from motrack.object_detection.catalog import OBJECT_DETECTION_CATALOG
from motrack.utils.lookup import LookupTable


def _remap_bytetrack_state_dict(state_dict: dict) -> dict:
    """
    Remap state-dict keys from the original YOLOX (Megvii / ByteTrack) format
    to the mmdetection YOLOX format.

    ByteTrack packs both CSPDarknet and the FPN/PAN neck (YOLOPAFPN) under a
    single ``backbone`` key.  mmdetection separates them into ``backbone``
    (CSPDarknet only) and ``neck`` (YOLOXPAFPN).  The head is renamed from
    ``head`` to ``bbox_head`` and every sub-key gets a ``multi_level_`` prefix.
    CSP block internals are also renamed (conv1→main_conv, etc.).

    Keys with no mmdetection equivalent (``head.stems``) are silently dropped;
    ``neck.out_convs`` has no ByteTrack source and stays randomly-initialised.
    """
    # ByteTrack dark stage index → mmdet stage index
    _DARK_TO_STAGE = {'2': '1', '3': '2', '4': '3', '5': '4'}

    # ByteTrack CSP block internal key → mmdet CSPLayer key
    _CSP_KEY_MAP = {
        'conv1': 'main_conv',
        'conv2': 'short_conv',
        'conv3': 'final_conv',
        'm':     'blocks',
    }

    # ByteTrack neck component → mmdet neck component (preserving C3 blocks)
    _NECK_COMPONENT_MAP = {
        'lateral_conv0': ('reduce_layers.0', False),
        'reduce_conv1':  ('reduce_layers.1', False),
        'C3_p4':         ('top_down_blocks.0', True),
        'C3_p3':         ('top_down_blocks.1', True),
        'bu_conv2':      ('downsamples.0', False),
        'bu_conv1':      ('downsamples.1', False),
        'C3_n3':         ('bottom_up_blocks.0', True),
        'C3_n4':         ('bottom_up_blocks.1', True),
    }

    # ByteTrack head component → mmdet head component
    _HEAD_KEY_MAP = {
        'cls_convs': 'multi_level_cls_convs',
        'reg_convs': 'multi_level_reg_convs',
        'cls_preds': 'multi_level_conv_cls',
        'reg_preds': 'multi_level_conv_reg',
        'obj_preds': 'multi_level_conv_obj',
    }

    def _remap_csp_suffix(suffix: str) -> str:
        parts = suffix.split('.', 1)
        if parts[0] in _CSP_KEY_MAP:
            tail = ('.' + parts[1]) if len(parts) > 1 else ''
            return _CSP_KEY_MAP[parts[0]] + tail
        return suffix

    new_state_dict: dict = {}
    skipped: list = []

    for key, value in state_dict.items():

        # ── backbone.backbone.stem.* → backbone.stem.* ──────────────────────
        if key.startswith('backbone.backbone.stem.'):
            new_key = 'backbone.stem.' + key[len('backbone.backbone.stem.'):]

        # ── backbone.backbone.darkN.* → backbone.stageM.* ──────────────────
        elif key.startswith('backbone.backbone.dark'):
            rest = key[len('backbone.backbone.dark'):]   # e.g. '2.1.conv1.conv.weight'
            dark_idx = rest[0]
            if dark_idx not in _DARK_TO_STAGE:
                skipped.append(key)
                continue
            stage_idx = _DARK_TO_STAGE[dark_idx]
            sub_rest = rest[2:]                          # e.g. '1.conv1.conv.weight'
            block_parts = sub_rest.split('.', 1)
            block_idx = block_parts[0]
            # dark2/3/4: [Conv(0), C3/CSPLayer(1)]
            # dark5:     [Conv(0), SPP(1), C3/CSPLayer(2)]
            # CSP internal renaming applies only to the last (C3) block.
            csp_block_idx = '2' if dark_idx == '5' else '1'
            if block_idx == csp_block_idx and len(block_parts) > 1:
                csp_suffix = _remap_csp_suffix(block_parts[1])
                new_key = f'backbone.stage{stage_idx}.{csp_block_idx}.{csp_suffix}'
            else:
                new_key = f'backbone.stage{stage_idx}.{sub_rest}'

        # ── backbone.NECK_COMP.* → neck.MMDET_COMP.* ────────────────────────
        elif key.startswith('backbone.'):
            rest = key[len('backbone.'):]
            matched = False
            for bt_comp, (mmdet_comp, is_csp) in _NECK_COMPONENT_MAP.items():
                prefix = bt_comp + '.'
                if rest == bt_comp or rest.startswith(prefix):
                    suffix = rest[len(bt_comp):]
                    if suffix.startswith('.'):
                        suffix = suffix[1:]
                    if is_csp and suffix:
                        suffix = _remap_csp_suffix(suffix)
                    new_key = f'neck.{mmdet_comp}' + (f'.{suffix}' if suffix else '')
                    matched = True
                    break
            if not matched:
                skipped.append(key)
                continue

        # ── head.* → bbox_head.multi_level_*.* ──────────────────────────────
        elif key.startswith('head.'):
            rest = key[len('head.'):]
            if rest.startswith('stems.'):
                # head.stems in ByteTrack == neck.out_convs in mmdet
                # (channel reduction before the detection head)
                new_key = 'neck.out_convs.' + rest[len('stems.'):]
                new_state_dict[new_key] = value
                continue
            matched = False
            for bt_comp, mmdet_comp in _HEAD_KEY_MAP.items():
                prefix = bt_comp + '.'
                if rest == bt_comp or rest.startswith(prefix):
                    suffix = rest[len(bt_comp):]   # e.g. '.0.0.conv.weight'
                    new_key = f'bbox_head.{mmdet_comp}{suffix}'
                    matched = True
                    break
            if not matched:
                skipped.append(key)
                continue

        else:
            new_key = key

        new_state_dict[new_key] = value

    if skipped:
        import logging
        logging.getLogger(__name__).debug(
            'ByteTrack keys without mmdet equivalent (skipped): %s', skipped
        )

    return new_state_dict


@OBJECT_DETECTION_CATALOG.register('mmdet_yolox')
class MMDetYOLOXInference(ObjectDetectionInference):
    """
    Object Detection – YOLOX via mmdetection.

    Parameters
    ----------
    model_config : str
        Path to an mmdetection Python config file (e.g.
        ``configs/mmdet/yolox_x_bytetrack.py``).
    model_path : str
        Path to a checkpoint.  Two formats are supported:
        * Standard mmdet checkpoint (``.pth``) – used when
          ``bytetrack_compat=False``.
        * ByteTrack / original YOLOX checkpoint (``.pth.tar``) – used when
          ``bytetrack_compat=True`` (the default).
    accelerator : str
        PyTorch device string, e.g. ``'cuda:0'`` or ``'cpu'``.
    conf : float
        Confidence threshold applied after inference.
    min_bbox_area : int
        Minimum bounding-box area (in pixels²) to keep a detection.
    bytetrack_compat : bool
        When ``True`` (default) the checkpoint is assumed to be in the
        original YOLOX / ByteTrack format and key remapping is applied.
        Set to ``False`` for native mmdetection checkpoints.
    lookup : LookupTable, optional
        Class-index → label string mapping.
    """

    def __init__(
        self,
        model_config: str,
        model_path: str,
        accelerator: str = 'cuda:0',
        conf: float = 0.1,
        min_bbox_area: int = 0,
        bytetrack_compat: bool = True,
        lookup: Optional[LookupTable] = None,
    ) -> None:
        super().__init__(lookup=lookup)

        try:
            from mmdet.apis import init_detector
        except ImportError as exc:
            raise ImportError(
                'mmdetection is required for MMDetYOLOXInference. '
                'Run: uv sync && uv run mim install "mmcv>=2.0.0"'
            ) from exc

        # Initialise architecture without loading weights so we can load them
        # manually (needed for the ByteTrack key-remapping path).
        self._model = init_detector(model_config, checkpoint=None, device=accelerator)

        # Disable score filtering inside mmdet; we apply our own threshold.
        self._model.test_cfg.score_thr = 0.0

        if bytetrack_compat:
            self._load_bytetrack_checkpoint(model_path)
        else:
            from mmengine.runner import load_checkpoint
            load_checkpoint(self._model, model_path, map_location=accelerator)

        self._model.eval()
        self._conf = conf
        self._min_bbox_area = min_bbox_area

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_bytetrack_checkpoint(self, path: str) -> None:
        import logging
        log = logging.getLogger(__name__)

        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        raw_state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        remapped = _remap_bytetrack_state_dict(raw_state_dict)
        missing, unexpected = self._model.load_state_dict(remapped, strict=False)

        # neck.out_convs are present in mmdet YOLOX but not in ByteTrack YOLOX;
        # they are randomly initialised and their absence is expected.
        fatal_missing = [k for k in missing if not k.startswith('neck.out_convs')]
        if fatal_missing:
            raise RuntimeError(
                f'Missing keys when loading ByteTrack checkpoint:\n{fatal_missing}\n'
                'The mmdet config may not match the checkpoint architecture.'
            )
        if missing:
            log.debug('Expected missing keys (randomly initialised): %s', missing)
        if unexpected:
            log.warning(
                'Unexpected keys in ByteTrack checkpoint (ignored): %s', unexpected
            )

    # ------------------------------------------------------------------
    # ObjectDetectionInference interface
    # ------------------------------------------------------------------

    def predict_raw(self, image: np.ndarray) -> Any:
        """Run mmdetection inference and return raw DetDataSample."""
        from mmdet.apis import inference_detector
        with torch.no_grad():
            return inference_detector(self._model, image)

    def postprocess(
        self, image: np.ndarray, raw: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert mmdetection output to the motrack format.

        Returns
        -------
        bboxes_xyxy : np.ndarray, shape (N, 4), float32
            Normalised bounding boxes [x1, y1, x2, y2] in [0, 1].
        classes : np.ndarray, shape (N,), float32
            Class indices.
        confidences : np.ndarray, shape (N,), float32
            Confidence scores.
        """
        pred = raw.pred_instances
        bboxes = pred.bboxes.cpu().numpy().astype(np.float32)    # (N, 4) pixel xyxy
        scores = pred.scores.cpu().numpy().astype(np.float32)    # (N,)
        labels = pred.labels.cpu().numpy().astype(np.float32)    # (N,)

        # Confidence filter
        mask = scores >= self._conf
        bboxes, scores, labels = bboxes[mask], scores[mask], labels[mask]

        # Min-area filter (pixel area before normalisation)
        if self._min_bbox_area > 0:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            mask = areas >= self._min_bbox_area
            bboxes, scores, labels = bboxes[mask], scores[mask], labels[mask]

        # Normalise coordinates to [0, 1]
        h, w = image.shape[:2]
        bboxes[:, [0, 2]] /= w
        bboxes[:, [1, 3]] /= h
        bboxes = np.clip(bboxes, 0.0, 1.0)

        return bboxes, labels, scores
