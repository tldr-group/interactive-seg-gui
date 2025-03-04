import tkinter as tk
from queue import Queue
from PIL import Image

from gui_elements.zoomable_canvas import CanvasImage
from gui_elements.constants import Message, COLOURS


class InteractiveCanvas(CanvasImage):
    """
    PolygonCanvas.

    Inherits from gui_elements/zoom_scroll_canvas.py. Contains all the methods for drawing onto the
    zooming/scrolling canvas and passing that data to the GUI. Generally the pattern is: draw on
    canvas object then once labe lis confirmed, send to data model, delete drawing on canvas,
    update label overlay with the confirmed data.
    """

    def __init__(
        self,
        parent: tk.Widget,
        out_queue: Queue[Message],
    ):
        """Init the canvas and bind all the keypresses."""
        super(InteractiveCanvas, self).__init__(parent)
        self.parent = parent
        self.queue = out_queue

        self.current_img_hw = (10, 10)
        self.image_available = False

        self.label_val = tk.IntVar(parent, 1)  # 1
        self.brush_width = tk.IntVar(parent, 5)

        self.fill_colour: str = COLOURS[
            self.label_val.get()
        ]  # self.app.class_colours[1]

        self.current_label_frac_points: list[tuple[float, float]] = []

        # self.canvas.bind("<Button-1>", self.left_click)
        # self.canvas.bind("<Button-3>", self.right_click)
        self.canvas.bind("<Motion>", self.mouse_motion)
        self.canvas.bind("<B1-Motion>", self.mouse_motion_while_click)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_release)
        self.canvas.bind("<Control-z>", self.undo)

        self.canvas.bind("<Escape>", self.cancel)
        # self.canvas.bind("<Delete>", self.delete)

        for i in range(10):
            self.canvas.bind(f"{i}", self._num_key_press)

    def set_current_image(self, pil_image, new=False) -> None:
        super().set_current_image(pil_image, new)
        self.image_available = True
        self.current_img_hw = (pil_image.height, pil_image.width)

    def mouse_motion(self, event: tk.Event) -> None:
        result = self._bounds_check_return_coords(event)
        if result is None:
            return None
        else:
            x, y, _, _ = result
            self._mouse_motion_draw_cursor(x, y, int(self.brush_width.get()))

    def mouse_motion_while_click(self, event: tk.Event) -> None:
        """For brush type labelling."""
        result = self._bounds_check_return_coords(event)
        if result is None:
            return None
        else:
            self.place_poly_point(*result, int(self.brush_width.get()))

    def mouse_release(self, event: tk.Event) -> None:
        """For brush/circle/rectangle label types (others covered by different click)."""
        result = self._bounds_check_return_coords(event)
        if result is None:
            return None
        else:
            self.finish_poly(event)

    def cancel(self, _event: tk.Event) -> None:
        """On esc key, cancel current label and delete in progress drawings on canvas."""
        self.canvas.delete("in_progress")
        self.current_label_frac_points = []

    def undo(self, _event: tk.Event) -> None:
        msg = Message("UNDO", None)
        self.queue.put(msg)

    # TODO: delete button that wipes all labels

    def _num_key_press(self, event):
        number = int(event.char)
        print(number)
        self.label_val = number
        self.fill_colour = COLOURS[self.label_val]

    # CONVERSION
    def _canvas_to_frac_coords(
        self, canvas_x: int, canvas_y: int
    ) -> tuple[float, float]:
        bbox = self.canvas.coords(self.container)
        dx, dy = bbox[2] - bbox[0], bbox[3] - bbox[1]
        frac_x, frac_y = (canvas_x - bbox[0]) / dx, (canvas_y - bbox[1]) / dy
        return (frac_x, frac_y)

    def _canvas_to_arr_coords(self, canvas_x: int, canvas_y: int) -> tuple[int, int]:
        h, w = self.current_img_hw
        frac_x, frac_y = self._canvas_to_frac_coords(canvas_x, canvas_y)
        return (int(frac_x * w), int(frac_y * h))

    def _frac_to_canvas_coords(
        self, frac_x: float, frac_y: float
    ) -> tuple[float, float]:
        bbox = self.canvas.coords(self.container)
        dx, dy = bbox[2] - bbox[0], bbox[3] - bbox[1]
        canvas_x, canvas_y = (frac_x * dx) + bbox[0], (frac_y * dy) + bbox[1]
        return (canvas_x, canvas_y)

    def _arr_to_frac_coords(self, arr_x: int, arr_y: int) -> tuple[float, float]:
        h, w = self.current_img_hw
        return arr_x / w, arr_y / h

    def _arr_to_canvas_coords(self, arr_x: int, arr_y: int) -> tuple[float, float]:
        frac_x, frac_y = self._arr_to_frac_coords(arr_x, arr_y)
        canvas_x, canvas_y = self._frac_to_canvas_coords(frac_x, frac_y)
        return (canvas_x, canvas_y)

    # LOGIC
    def _bounds_check_return_coords(
        self, event: tk.Event
    ) -> tuple[int, int, float, float] | None:
        bbox = self.canvas.coords(self.container)
        x, y = int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            frac_x, frac_y = self._canvas_to_frac_coords(x, y)
            return x, y, frac_x, frac_y
        else:
            self.canvas.delete("animated")
            return None

    def place_poly_point(
        self, x: int, y: int, frac_x: float, frac_y: float, r: int
    ) -> None:
        """Draw oval at click. Draw line from prev point to new point. Append fractional coords of new point to list."""
        scaled_w = r * self.imscale
        self.canvas.create_oval(
            x - scaled_w,
            y - scaled_w,
            x + scaled_w,
            y + scaled_w,
            fill=self.fill_colour,
            width=0,
            tags="in_progress",
        )
        self.current_label_frac_points.append((frac_x, frac_y))
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
            fill=None,
            width=2,
            tags="animated",
        )

    def _mouse_motion_poly(self, x: int, y: int) -> None:
        self.canvas.delete("animated")
        prev_point_frac_coords = self.current_label_frac_points[-1]
        x0, y0 = self._frac_to_canvas_coords(*prev_point_frac_coords)
        self.canvas.create_line(
            x0, y0, x, y, fill=self.fill_colour, width=2.2, tags="animated"
        )

    def finish_poly(self, _event: tk.Event) -> None:
        """Submit current label to data_model, delete in progress gui stuff."""
        self.canvas.delete("in_progress")
        self.canvas.delete("animated")

        h, w = self.current_img_hw
        points = [(int(x * w), int(y * h)) for x, y in self.current_label_frac_points]
        msg = Message("POINTS", points)
        self.queue.put(msg)

        self.current_label_frac_points = []
