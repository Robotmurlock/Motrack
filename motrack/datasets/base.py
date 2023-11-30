"""
Dataset interface (base class)
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple

import cv2
import numpy as np


@dataclass
class BasicSceneInfo:
    """
    Basic scene information
    """
    name: str
    category: str
    seqlength: Union[int, str]  # Can be parsed from `str`
    imheight: Union[int, str]
    imwidth: Union[int, str]

    def __post_init__(self):
        """
        Convert to proper type.
        """
        self.seqlength = int(self.seqlength)
        self.imheight = int(self.imheight)
        self.imwidth = int(self.imwidth)


@dataclass
class ObjectFrameData:
    frame_id: int
    bbox: Union[List[float], np.ndarray]
    image_path: str
    scene: Union[str, int]
    category: Union[str, int]
    occ: bool
    oov: bool


class BaseDataset(ABC):
    """
    Defines dataset interface.
    """
    def __init__(
        self,
        test: bool = False,
        sequence_list: Optional[List[str]] = None,
        image_shape: Union[None, List[int], Tuple[int, int]] = None,
        image_bgr_to_rgb: bool = True
    ):
        self._test = test
        self._sequence_list = sequence_list

        if image_shape is not None:
            assert isinstance(image_shape, (list, tuple)), \
                f'Invalid image shape type "{type(image_shape)}".'
            image_shape = tuple(image_shape)
            assert len(image_shape) == 2, f'Invalid image shape length "{len(image_shape)}"'

        self._image_shape = image_shape
        self._image_bgr_to_rgb = image_bgr_to_rgb

    def load_image(self, path: str) -> np.ndarray:
        """
        Loads image using cv2.

        Args:
            path: Image source path

        Returns:
            Loaded image
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'Failed to load image! File not found "{path}".')
        image = cv2.imread(path)
        assert image is not None, 'Invalid Program State!'

        if self._image_bgr_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self._image_shape is not None:
            image = cv2.resize(image, self._image_shape, interpolation=cv2.INTER_NEAREST)
        return image

    @abstractmethod
    def load_scene_image_by_frame_index(self, scene_name: str, frame_index: int) -> np.ndarray:
        """
        Loads scene image chosen by frame

        Args:
            scene_name: Scene name
            frame_index: Frame index

        Returns:
            Loaded image
        """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def scenes(self) -> List[str]:
        """
        Returns:
            List of scenes in dataset.
        """

    @abstractmethod
    def parse_object_id(self, object_id: str) -> Tuple[str, str]:
        """
        Parses and validates object id.

        Object id convention is `{scene_name}_{scene_object_id}` and is unique over all scenes.

        For MOT {scene_name} represents one video sequence.
        For SOT {scene_name} does not need to be unique for the sequence
        but `{scene_name}_{scene_object_id}` is always unique

        Args:
            object_id: Object id

        Returns:
            scene name, scene object id
        """

    @abstractmethod
    def get_object_category(self, object_id: str) -> str:
        """
        Gets category for object.

        Args:
            object_id: Object id

        Returns:
            Object category
        """

    @abstractmethod
    def get_scene_object_ids(self, scene_name: str) -> List[str]:
        """
        Gets object ids for given scene name

        Args:
            scene_name: Scene name

        Returns:
            Scene objects
        """

    def get_scene_number_of_object_ids(self, scene_name: str) -> int:
        """
        Gets number of unique objects in the scene.

        Args:
            scene_name: Scene name

        Returns:
            Number of objects in the scene
        """
        return len(self.get_scene_object_ids(scene_name))

    @abstractmethod
    def get_object_data_length(self, object_id: str) -> int:
        """
        Gets total number of data points for given `object_id` for .

        Args:
            object_id: Object id

        Returns:
            Number of data points
        """

    @abstractmethod
    def get_object_data(self, object_id: str, index: int, relative_bbox_coords: bool = True) -> ObjectFrameData:
        """
        Get object data point index.

        Args:
            object_id: Object id
            index: Index
            relative_bbox_coords: Scale bbox coords to [0, 1]

        Returns:
            Data point.
        """

    @abstractmethod
    def get_object_data_by_frame_index(
        self,
        object_id: str,
        frame_index: int,
        relative_bbox_coords: bool = True
    ) -> Optional[ObjectFrameData]:
        """
        Like `get_object_data_label` but data is relative to given frame_index.
        If object does not exist in given frame index then None is returned.

        Args:
            object_id: Object id
            frame_index: Frame Index
            relative_bbox_coords: Scale bbox coords to [0, 1]

        Returns:
            Data point.
        """

    @abstractmethod
    def get_scene_info(self, scene_name: str) -> BasicSceneInfo:
        """
        Get scene metadata by name.

        Args:
            scene_name: Scene name

        Returns:
            Scene metadata
        """

    @abstractmethod
    def get_scene_image_path(self, scene_name: str, frame_index: int) -> str:
        """
        Get image (frame) path for given scene and frame id.

        Args:
            scene_name: scene name
            frame_index: frame index

        Returns:
            Frame path
        """
