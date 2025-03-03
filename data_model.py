import numpy as np
from queue import Queue

from PIL import Image, ImageDraw

from typing import Literal, TypedDict, TypeAlias
from constants import MessageTypes
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


class Message(TypedDict):
    category: MessageTypes
    data: object


class DataModel(object):
    def __init__(self) -> None:
        self.in_queue: Queue[Message] = Queue(maxsize=40)
        self.out_queue: Queue[Message] = Queue(maxsize=40)

        self.out_queue.put({"category": "NOTIF", "data": "hello_world"})
