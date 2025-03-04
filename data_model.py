import numpy as np
from queue import Queue
from tifffile import imread

from PIL import Image, ImageDraw

from typing import TypeAlias
from gui_elements.constants import Message
from dataclasses import dataclass

Point: TypeAlias = tuple[float, float]


@dataclass
class Label:
    x0: int
    y0: int
    bbox: tuple[int, int, int, int]
    diff: np.ndarray


def draw_points_get_arr(
    points: list[Point], box_h: int, box_w: int, label_val: int, brush_width: int
) -> np.ndarray:
    temp_img = Image.new("L", (box_w, box_h), 0)
    ImageDraw.Draw(temp_img).line(points, fill=label_val, width=int(brush_width))
    return np.array(temp_img, dtype=np.int16)


def label_from_points(
    points: list[Point],
    seg_arr: np.ndarray,
    label_val: int,
    brush_width: int,
    update_seg_arr: bool = True,
) -> Label:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, y0 = int(min(xs)), int(min(ys))
    x1, y1 = int(max(xs)), int(max(ys))

    h, w = (y1 - y0), (x1 - x0)
    bbox = (x0, y0, x1, y1)
    # TODO: consider how erasing works
    new_label = draw_points_get_arr(points, h, w, label_val, brush_width)

    prev_state = seg_arr[y0:y1, x0:x1]

    diff = new_label - prev_state
    if update_seg_arr:
        seg_arr[y0:y1, x0:x1] += diff
    return Label(x0, y0, bbox, diff)


@dataclass
class Piece:
    """
    Piece.

    Fundamental unit of program. Holds the data associated with the image in $arr, a PIL image of the arr in $img.
    Labels is a list of label objects that belong to this piece. __post_init__ adds some mutable objects like
    segmentations and grid_points that are useful later.
    """

    img: Image.Image
    img_arr: np.ndarray

    labels: list[Label]
    labelled: bool = False
    segmented: bool = False

    def __post_init__(self) -> None:
        """Set these here because dataclasses don't like mutable objects being assigned in __init__."""
        shape: tuple[int, ...] = self.img_arr.shape
        self.h: int = shape[0]
        self.w: int = shape[1]

        # integer arr where 0 = not labelled and N > 0 indicates a label for class N at that pixel
        self.labels_arr: np.ndarray = np.zeros(shape, dtype=np.uint8)
        # integer arr where value N at pixel P indicates the classifier thinks P is class N
        self.seg_arr: np.ndarray = np.zeros(shape, dtype=np.uint8)

        # boolean arr where 1 = show this pixel in the overlay and 0 means hide. Used for hiding/showing labels later.
        self.label_alpha_mask = np.ones_like(self.seg_arr, dtype=bool)


class DataModel(object):
    def __init__(self) -> None:
        self.in_queue: Queue[Message] = Queue(maxsize=40)
        self.out_queue: Queue[Message] = Queue(maxsize=40)

        init_msg = Message("NOTIF", "hello world")
        self.out_queue.put(init_msg)

        self.gallery: list[Piece] = []

    def add_image(self, filepath: str) -> Piece:
        extension: str = filepath.split(".")[-1]
        if extension.lower() not in ["png", "jpg", "jpeg", "tif", "bmp", "tiff"]:
            raise Exception(f".{extension} is not a valid image file format")

        if extension.lower() in ["tiff", "tif"]:
            np_array: np.ndarray = imread(filepath)  # type: ignore
            pil_image = Image.fromarray(np_array).convert("RGBA")
        else:
            pil_image = Image.open(filepath)
            np_array = np.array(pil_image)
            pil_image.convert("RGBA")

        return Piece(pil_image, np_array, [], False, False)
