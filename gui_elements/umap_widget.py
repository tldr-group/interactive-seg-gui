from queue import Queue
import tkinter as tk

from tkinter import ttk
from typing import Callable

from .constants import COLOURS, Message

from .interactive_canvas import InteractiveCanvas


def _foo() -> None:
    pass


def is_bbox_area_above_threshold(points: list[tuple[float, float]], threshold: float) -> bool:
    """Return True if the bounding box area of the points is above the threshold."""
    if len(points) <= 1:
        return False
    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    area = (max_x - min_x) * (max_y - min_y)
    return area > threshold


class UMAP_window(tk.Toplevel):
    """Window to hold the UMAP plots and have special onclick events."""

    def __init__(
        self, app: tk.Widget, current_class: tk.IntVar, set_label_val: Callable, out_queue: Queue[Message]
    ) -> None:
        """Init all the widgets and function."""
        super(UMAP_window, self).__init__(app)
        self.title("UMAP")
        self.app = app
        self.queue = out_queue

        self.img_frame = ttk.LabelFrame(self, text="UMAP", padding=(35, 35))
        self.img_frame.grid(row=0, column=0, padx=(20, 10), pady=(20, 10), rowspan=3, columnspan=3, sticky="nsew")
        self.img_frame.rowconfigure(0, weight=1, minsize=750)
        self.img_frame.columnconfigure(0, weight=1, minsize=750)

        self.canvas = UMAPCanvas(self.img_frame, out_queue)
        self.canvas.grid(row=0, column=0)
        self.canvas.fill_colour = COLOURS[current_class.get()]  # "#000000"

        def on_class_change():
            """Change the fill colour of the canvas when the class changes."""
            self.canvas.fill_colour = COLOURS[current_class.get()]
            self.canvas.change_class_redraw()

        current_class.trace("w", lambda *_: on_class_change())

        self.spinbox = ttk.Spinbox(self, from_=1, to=60, command=set_label_val, textvariable=current_class)
        self.spinbox.grid(row=3, column=0, pady=(0, 10))

        self.cancel_button = ttk.Button(self, text="Cancel", command=_foo)
        self.cancel_button.grid(row=3, column=1, pady=(0, 10))

        self.done_button = ttk.Button(self, text="Add label!", command=self.confirm_label)
        self.done_button.grid(row=3, column=2, pady=(0, 10))
        # add buttons to RHS of this: a (coloured) spinbox for class selection (also linked to keypad) and a send seg button (also linked to enter/right click). Change colour of done box depening on what class it is

    def confirm_label(self) -> None:
        """Send confirm label message to data_model."""
        msg = Message("UMAP_CONFIRM_LABEL", None)
        self.queue.put(msg)

        self.canvas.canvas.delete("in_progress")
        self.canvas.current_label_frac_points = []

    def cancel_label(self) -> None:
        """Cancel current label."""
        msg = Message("UMAP_CLEAR_LABEL", None)
        self.queue.put(msg)

        self.canvas.canvas.delete("in_progress")


class UMAPCanvas(InteractiveCanvas):
    """Canvas specialised for UMAP labelling."""

    def __init__(self, parent: tk.Widget, out_queue: Queue[Message]):
        super().__init__(parent, out_queue)

        # %% DRAWING

    def _draw(self, frac_points: list[tuple[float, float]]) -> None:
        """Draw polygon from fractional points list."""
        if len(frac_points) == 0:
            return

        canvas_coords = [self._frac_to_canvas_coords(px, py) for px, py in self.current_label_frac_points]

        self.canvas.delete("in_progress")
        self.canvas.create_polygon(
            canvas_coords, fill="", width=2, tags="in_progress", outline=self.fill_colour, smooth=True
        )

    def place_poly_point(self, x: int, y: int, frac_x: float, frac_y: float, r: int) -> None:
        """Draw oval at click. Draw line from prev point to new point. Append fractional coords of new point to list."""
        self.current_label_frac_points.append((frac_x, frac_y))

        self._draw(self.current_label_frac_points)
        return None

    def change_class_redraw(self) -> None:
        """Redraw current polygon in new colour."""
        print("oi")
        self._draw(self.current_label_frac_points)

    def _mouse_motion_draw_cursor(self, x: int, y: int, r: int):
        scaled_w = r * self.imscale
        self.canvas.delete("animated")
        self.canvas.create_oval(
            x - scaled_w,
            y - scaled_w,
            x + scaled_w,
            y + scaled_w,
            outline=self.fill_colour,
            fill="",
            width=2,
            tags="animated",
        )

    def _mouse_motion_poly(self, x: int, y: int) -> None:
        self.canvas.delete("animated")
        prev_point_frac_coords = self.current_label_frac_points[-1]
        x0, y0 = self._frac_to_canvas_coords(*prev_point_frac_coords)
        # self.canvas.create_line(x0, y0, x, y, fill=self.fill_colour, width=2.2, tags="animated")

    def finish_poly(self, _event: tk.Event) -> None:
        """Submit current label to data_model, delete in progress gui stuff."""
        # self.canvas.delete("animated")

        if not is_bbox_area_above_threshold(self.current_label_frac_points, 0.0001):
            print("Bounding box too small, ignoring polygon")
            self.current_label_frac_points = []
            msg = Message("UMAP_CLEAR_LABEL", None)
            self.queue.put(msg)
            return

        msg = Message("UMAP_POLY_FRAC_POINTS", self.current_label_frac_points.copy())
        self.queue.put(msg)

        # self.current_label_frac_points = []
