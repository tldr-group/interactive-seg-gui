# %% IMPORTS
import tkinter as tk
from tkinter import ttk
from matplotlib.colors import ListedColormap
from PIL import Image

from gui_elements.constants import (
    COLOURS,
    MENU_BAR_ROW,
    SIDE_BAR_COL,
    PAD,
    CANVAS_H,
    CANVAS_W,
    CANVAS_H_GRID,
    CANVAS_W_GRID,
)
from gui_elements.interactive_canvas import InteractiveCanvas
from data_model import DataModel, Message


def _make_frame_contents_expand(frame: tk.Tk | tk.Frame | ttk.LabelFrame, i=5):
    for idx in range(i):
        frame.columnconfigure(index=idx, weight=1)
        frame.rowconfigure(index=idx, weight=1)


class App(ttk.Frame):
    """Parent widget for GUI. Contains event scheduler in listen() method."""

    def __init__(
        self, root: tk.Tk, data_model: DataModel, initial_img: Image.Image | None = None
    ) -> None:
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

        self.init_widgets(initial_img)

        self.event_loop()

    def init_widgets(self, initial_img: Image.Image | None = None) -> None:
        self._init_canv(initial_img)
    
    def _init_menubar(self) -> None:

    def _init_canv(self, initial_img: Image.Image | None = None) -> None:
        img_frame = ttk.LabelFrame(self, text="Image", padding=(3.5 * PAD, 3.5 * PAD))
        img_frame.grid(
            row=MENU_BAR_ROW + 1,
            column=0,
            padx=(2 * PAD, PAD),
            pady=(2 * PAD, PAD),
            rowspan=CANVAS_H_GRID,
            columnspan=CANVAS_W_GRID,
            sticky="nsew",
        )
        img_frame.rowconfigure(0, weight=1, minsize=CANVAS_H)
        img_frame.columnconfigure(0, weight=1, minsize=CANVAS_W)

        self.canvas = InteractiveCanvas(
            img_frame, self.data_model.out_queue, initial_img
        )
        self.canvas.grid(row=0, column=0)

    def handle_message(self, message: Message) -> None:
        header = message.category
        match header:
            case "NOTIF":
                print(message.data)
            case "POINTS":
                print(message.data)
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


#%%

class MenuBar(tk.Menu):
    """Menu bar across top of GUI with dropdown commands: load data, classifiers, save segs etc."""

    def __init__(self, root: tk.Tk, app: App) -> None:
        """Attach to root then initialise all the sub menus: data, classifiers, post process & save."""
        super(MenuBar, self).__init__(
            root
        )  # done s.t the menu bar is attached to the root (tk window) rather than the frame
        self.app = app

        data_name_fn_pairs: list[tuple[str, Callable]] = [
            ("Add Image", self._load_images),
            ("Remove Image", _foo),
        ]
        data_menu = self._make_dropdown(data_name_fn_pairs)
        self.add_cascade(label="Data", menu=data_menu)

        classifier_name_fn_pairs: list[tuple[str, Callable]] = [
            ("New Classifier", self.app.classifier_window),
            ("Train Classifier", self.app.data_model.train),
            ("Apply Classifier", self.app.data_model.segment),
            ("Load Classifier", self._load_classifier),
            ("Save Classifier", self._save_classifier),
        ]
        classifier_menu = self._make_dropdown(classifier_name_fn_pairs)
        self.add_cascade(label="Classifier", menu=classifier_menu)

        self.add_command(label="Post-Process", command=_foo)  # type: ignore
        self.add_command(label="Save Segmentation", command=self._save_segmentation)  # type: ignore

    def _make_dropdown(self, name_fn_pair_list: list[tuple[str, Callable]]) -> tk.Menu:
        menu = tk.Menu()
        n_commands: int = len(name_fn_pair_list)
        for i in range(n_commands):
            name, function = name_fn_pair_list[i]
            if name == "sep":
                menu.add_separator()
            else:
                menu.add_command(label=name, command=function)
        return menu

    def _load_images(self) -> None:
        file_paths = open_file_dialog_return_fps(title="Load Images")
        if file_paths == "":  # user closed fd or selected no files
            pass
        else:
            self.app.load_image_from_filepaths(file_paths)

    def _load_classifier(self) -> None:
        file_path = open_file_dialog_return_fp(title="Load Classifier")
        if file_path == "":
            return
        else:
            self.app.data_model.model.load_model(file_path)

    def _save_classifier(self) -> None:
        f = fd.asksaveasfile(mode="wb", defaultextension=".pkl")
        if f is None:
            return
        else:
            self.app.data_model.model.save_model(f)
            f.close()

    def _save_segmentation(self) -> None:
        f = fd.asksaveasfile(mode="wb", defaultextension=".tiff")
        if f is None:
            return
        else:
            self.app.data_model.save_seg(f)
            f.close()
