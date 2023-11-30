"""
MP4Reader.
"""
import os
from typing import Tuple, Union, Generator

import cv2
import numpy as np


class MP4Reader:
    """
    RAII mp4 reader
    """
    def __init__(self, path: str):
        """
        Args:
            path: Path to mp4 file

        Raises:
            FileNotFoundError: if video or sync file (meta file) does not exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'Mp4 file {path} does not exist')

        self._path = path

        # State
        self._mp4_reader = None

    def next(self) -> Union[np.ndarray, None]:
        if not self._mp4_reader.isOpened():
            raise StopIteration
        ret, frame = self._mp4_reader.read()
        if not ret:
            return None

        return frame

    def open(self) -> 'MP4Reader':
        # noinspection PyUnresolvedReferences
        self._mp4_reader = cv2.VideoCapture(self._path)
        return self

    def close(self) -> None:
        if self._mp4_reader is not None:
            self._mp4_reader.release()

    def __enter__(self) -> 'MP4Reader':
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def iter(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Returns:
            Frames generator (reading from video)
        """
        step_index = 0

        self.open()

        while True:
            try:
                frame = self.next()
                if frame is None:
                    break

                yield (step_index + 1), frame
            except StopIteration:
                break

            step_index += 1

        self.close()
