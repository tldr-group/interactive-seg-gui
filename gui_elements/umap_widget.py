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
