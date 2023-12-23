"""
BBox implementation
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union

import cv2
import numpy as np

from motrack.library.numpy_utils.bbox import affine_transform


@dataclass
class Point:
    """
    Point in 2D (x, y).
    """
    x: float
    y: float

    def copy(self) -> 'Point':
        """
        Returns: Copies object
        """
        return Point(x=self.x, y=self.y)

    def clip(self) -> None:
        """
        Clips coords to [0, 1] range.
        """
        self.x = max(0.0, min(1.0, self.x))
        self.y = max(0.0, min(1.0, self.y))

    def __lt__(self, other: 'Point') -> bool:
        return self.x < other.x and self.y < other.y

    def __le__(self, other: 'Point') -> bool:
        return self.x <= other.x and self.y <= other.y

    def __eq__(self, other: 'Point') -> bool:
        return all([np.isclose(self.x, other.x), np.isclose(self.y, other.y)])

    def as_numpy_xy(self, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Converts point to numpy array.

        Args:
            dtype: Numpy dtype

        Returns:
            Numpy point
        """
        return np.array([self.x, self.y], dtype=dtype)


@dataclass
class BBox:
    """
    BBox (rectangle) defined by two upper left and bottom right corners.
    """
    upper_left: Point
    bottom_right: Point

    def __post_init__(self):
        """
        Validation
        """
        return self.upper_left <= self.bottom_right

    def clip(self) -> None:
        """
        Clips coords to [0, 1] range.
        """
        self.bottom_right.clip()
        self.upper_left.clip()

    def copy(self) -> 'BBox':
        """
        Returns: Deepcopy of object
        """
        return BBox(
            upper_left=self.upper_left.copy(),
            bottom_right=self.bottom_right.copy()
        )

    @property
    def width(self) -> float:
        """
        Returns: Width
        """
        return self.bottom_right.x - self.upper_left.x

    @property
    def height(self) -> float:
        """
        Returns: Height
        """
        return self.bottom_right.y - self.upper_left.y

    @property
    def area(self) -> float:
        """
        Returns: Area
        """
        return self.height * self.width

    @property
    def center(self) -> Point:
        """
        Returns: Center
        """
        return Point(x=(self.bottom_right.x + self.upper_left.x) / 2, y=(self.bottom_right.y + self.upper_left.y) / 2)

    @property
    def xyxy(self) -> np.ndarray:
        """
        Returns:
            Numpy array in xyxy format
        """
        return np.array([self.upper_left.x, self.upper_left.y, self.bottom_right.x, self.bottom_right.y],
                        dtype=np.float32)

    def __eq__(self, other: 'BBox') -> bool:
        return self.upper_left == other.upper_left and self.bottom_right == other.bottom_right

    def intersection(self, other: 'BBox') -> Optional['BBox']:
        """
        Returns BBox if intersection exists else returns None

        Args:
            other: Other BBox

        Returns:
            Intersection of two bboxes
        """
        upper = max(self.upper_left.y, other.upper_left.y)
        bottom = min(self.bottom_right.y, other.bottom_right.y)
        left = max(self.upper_left.x, other.upper_left.x)
        right = min(self.bottom_right.x, other.bottom_right.x)

        if upper > bottom or left > right:
            return None

        return BBox(
            upper_left=Point(x=upper, y=left),
            bottom_right=Point(x=bottom, y=right)
        )

    def iou(self, other: 'BBox') -> float:
        """
        Intersection Over Union / Jaccard Index

        Args:
            other: Other BBox

        Returns: IOU
        """
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0

        union_area = self.area + other.area - intersection.area
        if union_area == 0.0:
            return 0.0

        return intersection.area / union_area

    def max_iou(self, others: List['BBox']) -> Tuple[float, int]:
        """
        Max iou over all other bboxes

        Args:
            others: List of BBoxes

        Returns: Value, index
        """
        ious = [self.iou(other) for other in others]
        index = int(np.argmax(ious))
        return ious[index], index

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float, clip: bool = False) -> 'BBox':
        """
        Creates BBox from xyxy format

        Args:
            x1: x1
            y1: y1
            x2: x2
            y2: y2
            clip: Clip bbox coordinates to [0, 1] range

        Returns: Bbox
        """
        bbox = cls(
            upper_left=Point(x=x1, y=y1),
            bottom_right=Point(x=x2, y=y2),
        )

        if clip:
            bbox.clip()

        return bbox

    @classmethod
    def from_xywh(cls, x: float, y: float, h: float, w: float, clip: bool = False) -> 'BBox':
        """
        Creates BBox from xyhw format

        Args:
            x: left
            y: up
            h: height
            w: width
            clip: Clip bbox coordinates to [0, 1] range

        Returns: Bbox
        """
        x1, y1, x2, y2 = x, y, x + h, y + w
        return cls.from_xyxy(x1, y1, x2, y2, clip=clip)

    @classmethod
    def from_cxywh(cls, x: float, y: float, w: float, h: float, clip: bool = False) -> 'BBox':
        """
        Creates BBox from cxywh format

        Args:
            x: center x
            y: center y
            w: width
            h: height
            clip: Clip bbox coordinates to [0, 1] range

        Returns: Bbox
        """
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        return cls.from_xyxy(x1, y1, x2, y2, clip=clip)

    def as_numpy_xyxy(self, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Converts Bbox to xyxy numpy array.

        Args:
            dtype: Numpy array dtype (default: np.float32)

        Returns:
            BBox xyxy coords as a numpy array.
        """
        return np.array([self.upper_left.x, self.upper_left.y, self.bottom_right.x, self.bottom_right.y], dtype=dtype)

    def as_numpy_xywh(self, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Converts Bbox to yxwh numpy array.

        Args:
            dtype: Numpy array dtype (default: np.float32)

        Returns:
            BBox yxwh coords as a numpy array.
        """
        return np.array([self.upper_left.x, self.upper_left.y, self.width, self.height], dtype=dtype)

    def as_numpy_cxywh(self, dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Converts Bbox to cxyhw numpy array.

        Args:
            dtype: Numpy array dtype (default: np.float32)

        Returns:
            BBox cxyhw coords as a numpy array.
        """
        return np.array([*self.center, self.width, self.height], dtype=dtype)

    @staticmethod
    def clip_coords(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
        """
        Clip coordinates to [0, 1] range such that x1 <= x2 and y1 <= y2.

        Args:
            x1: x1
            y1: y1
            x2: x2
            y2: y2

        Returns:
            Clipped coordinates
        """
        x1, y1, x2, y2 = [min(1.0, max(v, 0.0)) for v in [x1, y1, x2, y2]]
        x2 = max(x1, x2)  # assumption: x1 <= x2
        y2 = max(y1, y2)  # assumption: y1 <= y2

        return x1, y1, x2, y2

    def crop(self, image: np.ndarray) -> np.ndarray:
        """
        Crops image

        Args:
            image: Raw image

        Returns: Crop
        """
        x1, y1, x2, y2 = self.scaled_xyxy_from_image(image)
        return image[y1:y2, x1:x2, :].copy()

    def scaled_xyxy_from_image(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Scales coordinates to given image

        Args:
            image: Image

        Returns: coord in xyxy format (scaled)
        """
        h, w = image.shape[:2]
        return round(self.upper_left.x * w), round(self.upper_left.y * h), \
            round(self.bottom_right.x * w), round(self.bottom_right.y * h)

    def draw(self, image: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
        """
        Draws bbox on image

        Args:
            image: Image
            color: BBox color
            thickness: BBox thickness

        Returns: Image with drawn bbox
        """
        y1, x1, y2, x2 = self.scaled_xyxy_from_image(image)
        # noinspection PyUnresolvedReferences
        return cv2.rectangle(image, (y1, x1), (y2, x2), color, thickness)

    def affine_transform(self, warp: np.ndarray) -> 'BBox':
        """
        Performs affine transformation on the bbox (creates new bbox)

        Args:
            warp: Warp matrix 2x3

        Returns:
            New transformed bbox
        """
        points = self.as_numpy_xyxy()
        t_points = affine_transform(warp, points)
        return BBox.from_xyxy(*t_points)



LabelType = Union[int, str]


@dataclass
class PredBBox(BBox):
    """
    BBox with class label and detection confidence (optional).
    """
    label: LabelType = -1
    conf: Optional[float] = field(default=None)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def conf_annot(self) -> str:
        """
        Returns:
            Confidence annotation (as string).
        """
        return f'{100*self.conf:.1f}%' if self.conf is not None else 'GT'

    @property
    def compact_repr(self) -> str:
        """
        Returns: Compact PredBBox repr
        """
        coords_str = f'[{self.upper_left.x}, {self.upper_left.y}, {self.bottom_right.x}, {self.bottom_right.y}]'
        return f'PredBBox({coords_str} {self.label} ({self.conf_annot}))'

    @classmethod
    def create(cls, bbox: BBox, label: LabelType, conf: Optional[float] = None):
        """
        Creates PredBbox from regular BBox

        Args:
            bbox: BBox
            label: Label
            conf: Confidence

        Returns:
            Prediction BBox
        """
        return cls(
            upper_left=bbox.upper_left.copy(),
            bottom_right=bbox.bottom_right.copy(),
            label=label,
            conf=conf
        )

    def draw(self, image: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
        # Draw bbox
        super().draw(image, color=color, thickness=thickness)

        # Draw bbox annotation
        y1, x1, _, _ = self.scaled_xyxy_from_image(image)
        annot = f'[{self.label}] {self.conf_annot}'
        # noinspection PyUnresolvedReferences
        image = cv2.putText(image, annot, (y1 + 2, x1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        # noinspection PyUnresolvedReferences
        image = cv2.putText(image, annot, (y1 + 2, x1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return image
