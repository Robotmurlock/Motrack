"""
RAII wrapper of MP4Writer.
"""
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np


class MP4WriterResolutionMismatchException(Exception):
    """
    Frame resolution does not match with the previously used one
    """
    pass


class MP4WriterNotOpen(Exception):
    """
    MP4Writer is not open yet
    """
    pass


class MP4Writer:
    """
    RAII MP4 writer
    """
    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        path: str,
        fps: int,
        shape: Optional[Tuple[int, int]] = None,
        resize: bool = False,
        resize_interpolation: int = cv2.INTER_NEAREST
    ):
        """
        Args:
            path: Output video path
            fps: Output video fps
            shape: Set video output shape (optional) - Format (w, h)
                - if not set then first video frame resolution will be taken as shape
            resize: Resize video to some shape (requires shape to be set)
        """
        self._fps = fps
        self._path = path
        self._resize = resize
        self._resize_interpolation = resize_interpolation

        # State
        self._writer = None  # Stored on first frame
        self._shape: Optional[Tuple[int, int]] = shape  # Stored on first frame if not set

        # Check
        if self._resize and not self._shape:
            raise AttributeError('Shape has to be set with the "resize" option!')

    @property
    def fps(self) -> int:
        """
        Gets video fps.

        Returns:
            fps
        """
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        """
        Returns:
            Video resolution
        """
        if self._shape is None:
            raise MP4WriterNotOpen('Shape is unknown. Please use write() at least once.')
        return self._shape

    def write(self, frame: np.ndarray) -> None:
        """
        Appends new frame to mp4 video.

        Args:
            frame: Frame to write to video
        """
        h, w, _ = frame.shape
        shape = (w, h)
        if self._writer is None:
            if self._shape is None:
                self._shape = shape

            # noinspection PyUnresolvedReferences
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            # noinspection PyUnresolvedReferences
            self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, self._shape)

        if shape != self._shape:
            if not self._resize:
                raise MP4WriterResolutionMismatchException(f'Resolution mismatch! Expected {self._shape} but got {shape}.')
            else:
                # noinspection PyUnresolvedReferences
                frame = cv2.resize(frame, self._shape, interpolation=self._resize_interpolation)

        self._writer.write(frame)

    def close(self) -> None:
        """
        Closes mp4 writer.
        """
        self._writer.release()

    def __enter__(self) -> 'MP4Writer':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
