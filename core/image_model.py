from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class ImageInfo:
    filename: str = ""
    width: int = 0
    height: int = 0


class ImageModel:
    """
    Encapsulates the application's image state + history (Undo/Redo).
    Demonstrates:
      - Encapsulation: internal image + stacks hidden behind methods
      - Constructor: __init__
      - Methods: load/set/undo/redo/etc.
      - Class interaction: used by GUI + Processor
    """

    def __init__(self) -> None:
        self._image: Optional[np.ndarray] = None            # current image (BGR)
        self._original: Optional[np.ndarray] = None         # original loaded image (BGR)
        self._undo_stack: List[np.ndarray] = []
        self._redo_stack: List[np.ndarray] = []
        self._info: ImageInfo = ImageInfo()

    @property
    def info(self) -> ImageInfo:
        return self._info

    def has_image(self) -> bool:
        return self._image is not None

    def get_image_copy(self) -> Optional[np.ndarray]:
        if self._image is None:
            return None
        return self._image.copy()

    def load_image(self, bgr_img: np.ndarray, filename: str) -> None:
        self._original = bgr_img.copy()
        self._image = bgr_img.copy()
        self._undo_stack.clear()
        self._redo_stack.clear()

        h, w = bgr_img.shape[:2]
        self._info = ImageInfo(filename=filename, width=w, height=h)

    def set_image(self, bgr_img: np.ndarray, push_undo: bool = True) -> None:
        if self._image is None:
            self.load_image(bgr_img, self._info.filename)
            return

        if push_undo:
            self._undo_stack.append(self._image.copy())
            self._redo_stack.clear()

        self._image = bgr_img.copy()
        h, w = self._image.shape[:2]
        self._info.width = w
        self._info.height = h

    def undo(self) -> bool:
        if not self._undo_stack or self._image is None:
            return False
        self._redo_stack.append(self._image.copy())
        self._image = self._undo_stack.pop()
        h, w = self._image.shape[:2]
        self._info.width = w
        self._info.height = h
        return True

    def redo(self) -> bool:
        if not self._redo_stack or self._image is None:
            return False
        self._undo_stack.append(self._image.copy())
        self._image = self._redo_stack.pop()
        h, w = self._image.shape[:2]
        self._info.width = w
        self._info.height = h
        return True

    def reset_to_original(self) -> bool:
        if self._original is None:
            return False
        self.set_image(self._original.copy(), push_undo=True)
        return True
