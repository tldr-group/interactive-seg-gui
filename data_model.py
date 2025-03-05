import numpy as np
from queue import Queue
from tifffile import imread

from skimage.draw import ellipse
from PIL import Image

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
    points: list[Point],
    box_h: int,
    box_w: int,
    y0: int,
    x0: int,
    label_val: int,
    brush_width: int,
) -> np.ndarray:
    o = brush_width
    temp_arr = np.zeros((box_h, box_w), dtype=np.int16)
    for p in points:
        rr, cc = ellipse(p[1] - y0, p[0] - x0, brush_width, brush_width)
        temp_arr[rr, cc] = label_val
    return temp_arr


def label_from_points(
    points: list[Point],
    seg_arr: np.ndarray,
    label_val: int,
    brush_width: int,
    update_seg_arr: bool = True,
) -> Label:
    o = brush_width
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, y0 = int(min(xs)) - o, int(min(ys)) - o
    x1, y1 = int(max(xs)) + o, int(max(ys)) + o

    h, w = (y1 - y0), (x1 - x0)
    bbox = (x0, y0, x1, y1)
    # TODO: consider how erasing works
    new_label = draw_points_get_arr(points, h, w, y0, x0, label_val, brush_width)
    prev_state = seg_arr[y0:y1, x0:x1]

    diff = new_label - prev_state
    diff *= new_label > 0
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
        self.labels_arr: np.ndarray = np.zeros(shape, dtype=np.int16)
        # integer arr where value N at pixel P indicates the classifier thinks P is class N
        self.seg_arr: np.ndarray = np.zeros(shape, dtype=np.int16)

        # boolean arr where 1 = show this pixel in the overlay and 0 means hide. Used for hiding/showing labels later.
        self.label_alpha_mask = np.ones_like(self.seg_arr, dtype=bool)


class DataModel(object):
    def __init__(self) -> None:
        self.in_queue: Queue[Message] = Queue(maxsize=40)
        self.out_queue: Queue[Message] = Queue(maxsize=40)

        init_msg = Message("NOTIF", "microSeg v0.01 04/03/25")
        self.out_queue.put(init_msg)

        self.gallery: list[Piece] = []

    def add_image(self, filepath: str, add_to_gallery: bool = True) -> Piece:
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

        new_piece = Piece(pil_image, np_array, [], False, False)
        if add_to_gallery:
            self.gallery.append(new_piece)

        return new_piece

    def create_and_add_labels_from_points(
        self, points: list[Point], piece_idx: int, label_val: int, brush_width: int
    ) -> None:
        piece = self.gallery[piece_idx]
        label = label_from_points(
            points, piece.labels_arr, label_val, brush_width, True
        )
        piece.labels.append(label)
        piece.labelled = True

    def remove_last_label(self, piece_idx: int) -> None:
        piece = self.gallery[piece_idx]
        label = piece.labels.pop(-1)
        x0, y0, x1, y1 = label.bbox
        piece.labels_arr[y0:y1, x0:x1] -= label.diff

        if len(piece.labels) == 0:
            piece.labelled = False
