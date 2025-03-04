from typing import Literal
from dataclasses import dataclass

COLOURS: list[str] = [
    "#fafafa",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
CANVAS_W: int = 900
CANVAS_H: int = 600
CANVAS_H_GRID: int = 5
CANVAS_W_GRID: int = 1

PAD: int = 10
MENU_BAR_ROW: int = 0
SIDE_BAR_COL: int = 0
SIDE_COL: int = 2

MIN_TIME: float = 0.07

MessageTypes = Literal["NOTIF", "TRAIN", "SEGMENT", "POINTS", "UNDO"]
ProgressTypes = Literal["start", "stop", "progress", "N/A"]


@dataclass
class Message:
    category: MessageTypes
    data: object
