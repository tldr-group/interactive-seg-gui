from typing import Literal
from dataclasses import dataclass

# COLOURS: list[str] = [
#     "#fafafa",
#     "#1f77b4",
#     "#ff7f0e",
#     "#2ca02c",
#     "#d62728",
#     "#9467bd",
#     "#8c564b",
#     "#e377c2",
#     "#7f7f7f",
#     "#bcbd22",
#     "#17becf",
# ]

COLOURS: list[str] = [
    "#fafafa",
    "#003e83",
    "#b5d1cc",
    "#fa2b00",
    "#ffb852",
    "#718600",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


CANVAS_W: int = 1200
CANVAS_H: int = 750
CANVAS_H_GRID: int = 5
CANVAS_W_GRID: int = 1


PAD: int = 10
MENU_BAR_ROW: int = 0
SIDE_BAR_COL: int = 0
SIDE_COL: int = 2
BOTTOM_BAR_IDX = MENU_BAR_ROW + CANVAS_H_GRID + 1

MIN_TIME: float = 0.07
N_PREVIEW_SLICES: int = 10

MessageTypes = Literal["NOTIF", "TRAIN", "SEGMENT", "POINTS", "UNDO", "CLEAR", "CLASSIFIER", "UMAP_POLY_FRAC_POINTS"]
ProgressTypes = Literal["start", "stop", "progress", "N/A"]


@dataclass
class Message:
    category: MessageTypes
    data: object
