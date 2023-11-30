"""
MOT Challenge Dataset support. Supports: MOT17, MOT20, DanceTrack.
"""
import configparser
import copy
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Union, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from motrack.datasets.base import BaseDataset, BasicSceneInfo, ObjectFrameData
from motrack.datasets.catalog import DATASET_CATALOG
from motrack.utils import file_system

CATEGORY = 'pedestrian'
N_IMG_DIGITS = 6


@dataclass
class SceneInfo(BasicSceneInfo):
    """
    MOT Scene metadata (name, frame shape, ...)
    """
    dirpath: str
    gt_path: str
    framerate: Union[str, int]
    imdir: str
    imext: str

    def __post_init__(self):
        """
        Convert to proper type
        """
        super().__post_init__()
        self.framerate = int(self.framerate)


SceneInfoIndex = Dict[str, SceneInfo]


logger = logging.getLogger('MOTDataset')


@DATASET_CATALOG.register('mot')
class MOTDataset(BaseDataset):
    """
    Parses MOT dataset in given format
    """
    def __init__(
        self,
        path: str,
        sequence_list: Optional[List[str]] = None,
        image_shape: Union[None, List[int], Tuple[int, int]] = None,
        image_bgr_to_rgb: bool = True,
        test: bool = False
    ) -> None:
        """
        Args:
            path: Path to dataset
            sequence_list: Sequence filter by defined list
            image_shape: Resize images (optional - default: no resize)
            image_bgr_to_rgb: Convert BGR to RGB
        """
        super().__init__(
            test=test,
            sequence_list=sequence_list,
            image_shape=image_shape,
            image_bgr_to_rgb=image_bgr_to_rgb
        )

        self._path = path

        self._scene_info_index, self._n_digits = self._index_dataset(path, sequence_list, test=test)
        self._data_labels, self._n_labels, self._frame_to_data_index_lookup = self._parse_labels(self._scene_info_index, test=test)

    @property
    def scenes(self) -> List[str]:
        return list(self._scene_info_index.keys())

    def parse_object_id(self, object_id: str) -> Tuple[str, str]:
        assert object_id in self._data_labels, f'Unknown object id "{object_id}".'
        scene_name, scene_object_id = object_id.split('_')
        return scene_name, scene_object_id

    def get_object_category(self, object_id: str) -> str:
        return CATEGORY

    def get_scene_object_ids(self, scene_name: str) -> List[str]:
        assert scene_name in self.scenes, f'Unknown scene "{scene_name}". Dataset scenes: {self.scenes}.'
        return [d for d in self._data_labels if d.startswith(scene_name)]

    def get_scene_number_of_object_ids(self, scene_name: str) -> int:
        return len(self.get_scene_object_ids(scene_name))

    def get_object_data_length(self, object_id: str) -> int:
        return len(self._data_labels[object_id])

    def get_object_data(self, object_id: str, index: int, relative_bbox_coords: bool = True) -> Optional[ObjectFrameData]:
        data = copy.deepcopy(self._data_labels[object_id][index])
        if data is None:
            return None

        if relative_bbox_coords:
            scene_name, _ = self.parse_object_id(object_id)
            scene_info = self._scene_info_index[scene_name]
            data.bbox[0] = data.bbox[0] / scene_info.imwidth
            data.bbox[1] = data.bbox[1] / scene_info.imheight
            data.bbox[2] = data.bbox[2] / scene_info.imwidth
            data.bbox[3] = data.bbox[3] / scene_info.imheight

        return data

    def load_scene_image_by_frame_index(self, scene_name: str, frame_index: int) -> np.ndarray:
        # noinspection PyTypeChecker
        scene_info: SceneInfo = self.get_scene_info(scene_name)
        image_path = self._get_image_path(scene_info, frame_index + 1)
        return self.load_image(image_path)

    def get_object_data_by_frame_index(
        self,
        object_id: str,
        frame_index: int,
        relative_bbox_coords: bool = True
    ) -> Optional[dict]:
        return self.get_object_data(object_id, frame_index, relative_bbox_coords=relative_bbox_coords)

    def get_scene_info(self, scene_name: str) -> BasicSceneInfo:
        return self._scene_info_index[scene_name]

    def _get_image_path(self, scene_info: SceneInfo, frame_id: int) -> str:
        """
        Get frame path for given scene and frame id

        Args:
            scene_info: scene metadata
            frame_id: frame number

        Returns:
            Path to image (frame)
        """
        return os.path.join(scene_info.dirpath, scene_info.imdir, f'{frame_id:0{self._n_digits}d}{scene_info.imext}')

    def get_scene_image_path(self, scene_name: str, frame_index: int) -> str:
        scene_info = self._scene_info_index[scene_name]
        return self._get_image_path(scene_info, frame_index + 1)

    @staticmethod
    def _get_data_cache_path(path: str, data_name: str) -> str:
        """
        Get cache path for data object path.

        Args:
            path: Path

        Returns:
            Path where data object is or will be stored.
        """
        filename = Path(path).stem
        cache_filename = f'.{filename}.{data_name}.json'
        dirpath = str(Path(path).parent)
        return os.path.join(dirpath, cache_filename)

    @staticmethod
    def parse_scene_ini_file(scene_directory: str, label_type_name) -> SceneInfo:
        gt_path = os.path.join(scene_directory, label_type_name, f'{label_type_name}.txt')
        scene_info_path = os.path.join(scene_directory, 'seqinfo.ini')
        raw_info = configparser.ConfigParser()
        raw_info.read(scene_info_path)
        raw_info = dict(raw_info['Sequence'])
        raw_info['gt_path'] = gt_path
        raw_info['dirpath'] = scene_directory

        return SceneInfo(**raw_info, category=CATEGORY)

    @staticmethod
    def _index_dataset(
        path: str,
        sequence_list: Optional[List[str]],
        test: bool = False
    ) -> Tuple[SceneInfoIndex, int]:
        """
        Index dataset content. Format: { {scene_name}: {scene_labels_path} }

        Args:
            path: Path to dataset
            sequence_list: Filter scenes
            test: Is it test split

        Returns:
            Index to scenes, number of digits used in images name convention (may vary between datasets)
        """
        scene_names = [file for file in file_system.listdir(path) if not file.startswith('.')]
        logger.debug(f'Found {len(scene_names)} scenes. Names: {scene_names}.')
        n_digits = N_IMG_DIGITS

        scene_info_index: SceneInfoIndex = {}
        n_skipped: int = 0

        for scene_name in scene_names:
            if sequence_list is not None and scene_name not in sequence_list:
                continue

            scene_directory = os.path.join(path, scene_name)
            scene_files = file_system.listdir(scene_directory)

            # Scene content validation
            skip_scene = False
            files_to_check = ['gt', 'seqinfo.ini'] if not test else ['seqinfo.ini']
            for filename in files_to_check:
                if filename not in scene_files:
                    msg = f'Ground truth file "{filename}" not found on path "{scene_directory}". Contents: {scene_files}'
                    raise FileNotFoundError(msg)

            if 'img1' in scene_files:
                # Check number of digits used in image name (e.g. DanceTrack and MOT20 have different convention)
                img1_path = os.path.join(scene_directory, 'img1')
                image_names = file_system.listdir(img1_path)
                assert len(image_names) > 0, 'Image folder exists but it is empty!'
                image_name = Path(image_names[0]).stem
                n_digits = len(image_name)

            if skip_scene:
                n_skipped += 1
                continue

            scene_info = MOTDataset.parse_scene_ini_file(scene_directory, 'gt')
            scene_info_index[scene_name] = scene_info
            logger.debug(f'Scene info {scene_info}.')

        if n_digits != N_IMG_DIGITS:
            logger.warning(f'This dataset does not have default number of digits in image name. Got {n_digits} where default is {N_IMG_DIGITS}.')

        logger.info(f'Total number of parsed scenes is {len(scene_info_index)}. Number of skipped scenes is {n_skipped}.')

        return scene_info_index, n_digits

    def _parse_labels(self, scene_infos: SceneInfoIndex, test: bool = False) -> Tuple[Dict[str, List[ObjectFrameData]], int, Dict[str, Dict[int, int]]]:
        """
        Loads all labels dictionary with format:
        {
            {scene_name}_{object_id}: {
                {frame_id}: [ymin, xmin, w, h]
            }
        }

        Args:
            scene_infos: Scene Metadata
            test: If test then no parsing is performed

        Returns:
            Labels dictionary
        """
        data: Dict[str, List[Optional[ObjectFrameData]]] = {}
        frame_to_data_index_lookup: Dict[str, Dict[int, int]] = defaultdict(dict)
        n_labels = 0
        if test:
            # Return empty labels
            return data, n_labels, frame_to_data_index_lookup

        for scene_name, scene_info in scene_infos.items():
            seqlength = self._scene_info_index[scene_name].seqlength

            df = pd.read_csv(scene_info.gt_path, header=None)
            df = df[df[7] == 1]  # Ignoring values that are not evaluated

            df = df.iloc[:, :6]
            df.columns = ['frame_id', 'object_id', 'xmin', 'ymin', 'w', 'h']  # format: yxwh
            df['object_global_id'] = \
                scene_name + '_' + df['object_id'].astype(str)  # object id is not unique over all scenes
            df = df.drop(columns='object_id', axis=1)
            df = df.sort_values(by=['object_global_id', 'frame_id'])
            n_labels += df.shape[0]

            object_groups = df.groupby('object_global_id')
            for object_global_id, df_grp in tqdm(object_groups, desc=f'Parsing {scene_name}', unit='pedestrian'):
                df_grp = df_grp.drop(columns='object_global_id', axis=1).set_index('frame_id')

                data[object_global_id] = [None for _ in range(seqlength)]
                for frame_id, row in df_grp.iterrows():
                    data[object_global_id][int(frame_id) - 1] = ObjectFrameData(
                        frame_id=frame_id,
                        bbox=row.values.tolist(),
                        image_path=self._get_image_path(scene_info, frame_id),
                        occ=False,
                        oov=False,
                        scene=scene_name,
                        category=CATEGORY
                    )
                    frame_to_data_index_lookup[object_global_id][frame_id] = len(data[object_global_id]) - 1

        logger.debug(f'Parsed labels. Dataset size is {n_labels}.')
        data = dict(data)  # Disposing unwanted defaultdict side-effects
        return data, n_labels, frame_to_data_index_lookup

    def __len__(self) -> int:
        return self._n_labels
