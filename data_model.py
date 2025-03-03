import numpy as np
from queue import Queue

from PIL import Image, ImageDraw

from typing import Literal, TypedDict, TypeAlias
from constants import MessageTypes
from dataclasses import dataclass


class Message(TypedDict):
    category: MessageTypes
    data: object


class DataModel(object):
    def __init__(self) -> None:
        self.in_queue: Queue[Message] = Queue(maxsize=40)
        self.out_queue: Queue[Message] = Queue(maxsize=40)

        self.out_queue.put({"category": "NOTIF", "data": "hello_world"})
