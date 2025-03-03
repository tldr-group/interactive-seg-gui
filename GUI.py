# %% IMPORTS
import tkinter as tk
from tkinter import ttk
from matplotlib.colors import ListedColormap

from constants import COLOURS, MessageTypes
from data_model import DataModel, Message


def _make_frame_contents_expand(frame: tk.Tk | tk.Frame | ttk.LabelFrame, i=5):
    for idx in range(i):
        frame.columnconfigure(index=idx, weight=1)
        frame.rowconfigure(index=idx, weight=1)


class App(ttk.Frame):
    """Parent widget for GUI. Contains event scheduler in listen() method."""

    def __init__(self, root: tk.Tk, data_model: DataModel) -> None:
        """Take $root and assign it to attr .root. Inits other widgets and starts scheduler."""
        ttk.Frame.__init__(self)
        self.root = root
        self.data_model = data_model

        self.overlay_needs_updating: bool = False
        self.seg_overlay_alpha: float = 1.0

        self.brush_width: float = 1.0

        self.cmap = ListedColormap(COLOURS)

        self.root.option_add("*tearOff", False)
        _make_frame_contents_expand(self.root)

        self.event_loop()

    def handle_message(self, message: Message) -> None:
        header = message["category"]
        match header:
            case "NOTIF":
                print(message["data"])
            case _:
                raise Exception(f"Undefined message type {header}")

    def event_loop(self) -> None:
        queue = self.data_model.out_queue
        while queue.empty() is False:
            data_in = queue.get_nowait()
            self.handle_message(data_in)
            self.needs_updating = True

        if self.needs_updating:
            pass
        self.loop = self.root.after(100, self.event_loop)
