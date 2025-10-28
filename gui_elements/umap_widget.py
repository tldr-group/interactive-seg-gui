from queue import Queue
import tkinter as tk

from tkinter import ttk

from .constants import Message

from .interactive_canvas import InteractiveCanvas


def _foo() -> None:
    pass


class UMAP_window(tk.Toplevel):
    """Window to hold the UMAP plots and have special onclick events."""

    def __init__(self, app: tk.Widget, n_classes: int, out_queue: Queue[Message]) -> None:
        """Init all the widgets and function."""
        super(UMAP_window, self).__init__(app)
        self.title("UMAP")
        self.app = app
        self.img_frame = ttk.LabelFrame(self, text="UMAP", padding=(35, 35))
        self.img_frame.grid(row=0, column=0, padx=(20, 10), pady=(20, 10), rowspan=3, columnspan=3, sticky="nsew")
        self.img_frame.rowconfigure(0, weight=1, minsize=750)
        self.img_frame.columnconfigure(0, weight=1, minsize=750)

        self.canvas = UMAPCanvas(self.img_frame, out_queue)
        self.canvas.grid(row=0, column=0)
        self.canvas.fill_colour = "#000000"

        self.spinbox_var = tk.IntVar()

        def switch():
            x = self.spinbox_var.get()
            print(x)
            # self.canvas.switch_class(int(x))

        self.spinbox = ttk.Spinbox(self, from_=1, to=n_classes, command=switch, textvariable=self.spinbox_var)
        self.spinbox.grid(row=3, column=0, pady=(0, 10))

        self.cancel_button = ttk.Button(self, text="Cancel", command=_foo)
        self.cancel_button.grid(row=3, column=1, pady=(0, 10))

        self.done_button = ttk.Button(self, text="Apply", command=_foo)
        self.done_button.grid(row=3, column=2, pady=(0, 10))
        # add buttons to RHS of this: a (coloured) spinbox for class selection (also linked to keypad) and a send seg button (also linked to enter/right click). Change colour of done box depening on what class it is


class UMAPCanvas(InteractiveCanvas):
    """Canvas specialised for UMAP labelling."""

    def __init__(self, parent: tk.Widget, out_queue: Queue[Message]):
        super().__init__(parent, out_queue)

        # %% DRAWING

    def place_poly_point(self, x: int, y: int, frac_x: float, frac_y: float, r: int) -> None:
        """Draw oval at click. Draw line from prev point to new point. Append fractional coords of new point to list."""
        self.current_label_frac_points.append((frac_x, frac_y))

        canvas_coords = [self._frac_to_canvas_coords(px, py) for px, py in self.current_label_frac_points]

        self.canvas.delete("in_progress")
        self.canvas.create_polygon(
            canvas_coords, fill="", width=2, tags="in_progress", outline=self.fill_colour, smooth=True
        )
        return None

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
        # self.canvas.delete("in_progress")
        # self.canvas.delete("animated")

        # print(self.current_label_frac_points)

        msg = Message("UMAP_POLY_FRAC_POINTS", self.current_label_frac_points.copy())
        self.queue.put(msg)

        print("mouseup")

        self.current_label_frac_points = []
