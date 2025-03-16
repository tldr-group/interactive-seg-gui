# %% IMPORTS
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

import numpy as np
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
    BOTTOM_BAR_IDX,
)
from gui_elements.interactive_canvas import InteractiveCanvas
from data_model import DataModel, Piece, Message, Point

from interactive_seg_backend.file_handling import (
    save_segmentation,
    save_labels,
    load_labels,
)
from interactive_seg_backend.classifiers import load_classifier


from typing import Callable, Literal, Union


# %% FUNCTIONS
def _foo(x):  # placeholder fn to be deleted later
    print("Not implemented")


def _foo_n(x):
    pass


def _no_arg_foo():
    print("Not implemented")


def open_file_dialog_return_fps(
    title: str = "Open",
    file_type_name: str = "Image",
    file_types_string: str = ".tif .tiff .png .jpg",
) -> Union[Literal[""], tuple[str, ...]]:
    """Open file dialog and select n files, returning their file paths then loading them."""
    filepaths: Union[Literal[""], tuple[str, ...]] = fd.askopenfilenames(
        filetypes=[(f"{file_type_name} files:", file_types_string)], title=title
    )

    if filepaths == ():  # if user closed file manager w/out selecting
        return ""
    return filepaths


def open_file_dialog_return_fp(
    title: str = "Open",
    file_type_name: str = "Pickle",
    file_types_string: str = ".pkl",
) -> str:
    """Open file dialog and select n files, returning their file paths then loading them."""
    filepath: str = fd.askopenfilename(
        filetypes=[(f"{file_type_name} files:", file_types_string)], title=title
    )
    return filepath


def _make_frame_contents_expand(frame: tk.Tk | tk.Frame | ttk.LabelFrame, i=5):
    for idx in range(i):
        frame.columnconfigure(index=idx, weight=1)
        frame.rowconfigure(index=idx, weight=1)

    frame.columnconfigure(index=0, weight=0)  # stop sidebar expanding
    frame.rowconfigure(index=BOTTOM_BAR_IDX, weight=0)


# %% APP
class App(ttk.Frame):
    """Parent widget for GUI. Contains event scheduler in listen() method."""

    def __init__(
        self,
        root: tk.Tk,
        data_model: DataModel,
        initial_img_paths: tuple[str, ...] | None = None,
    ) -> None:
        """Take $root and assign it to attr .root. Inits other widgets and starts scheduler."""
        ttk.Frame.__init__(self)
        self.root = root
        self.data_model = data_model

        self.current_piece_idx = tk.IntVar(self, value=0)

        self.overlay_needs_updating: bool = False
        self.seg_overlay_alpha = tk.DoubleVar(self, value=1.0)
        self.label_overlay_alpha: float = 0.7

        self.cmap = ListedColormap(COLOURS)

        self.root.option_add("*tearOff", False)
        _make_frame_contents_expand(self.root)

        self.init_widgets()

        if initial_img_paths is not None:
            self.load_image_from_filepaths(initial_img_paths)

        self.event_loop()

    # %% INIT_WIDGETS
    def init_widgets(self) -> None:
        self._init_menubar()
        self._init_canv()
        self._init_sidebar()
        self._init_bottombar()

    def _init_menubar(self) -> None:
        self.menu_bar = MenuBar(self.root, self)
        self.root.config(menu=self.menu_bar)

    def _init_canv(self) -> None:
        img_frame = ttk.LabelFrame(self, text="Image", padding=(3.5 * PAD, 3.5 * PAD))
        img_frame.grid(
            row=MENU_BAR_ROW + 1,
            column=SIDE_BAR_COL + 1,
            padx=(2 * PAD, PAD),
            pady=(2 * PAD, PAD),
            rowspan=CANVAS_H_GRID,
            columnspan=CANVAS_W_GRID,
            sticky="nsew",
        )
        img_frame.configure(cursor="none")
        img_frame.rowconfigure(0, weight=1, minsize=CANVAS_H)
        img_frame.columnconfigure(0, weight=1, minsize=CANVAS_W)

        self.canvas = InteractiveCanvas(img_frame, self.data_model.out_queue)
        self.canvas.grid(row=0, column=0)

    def _init_sidebar(self) -> None:
        frame = ttk.Frame(self, relief="groove", borderwidth=2)
        frame.grid(
            row=MENU_BAR_ROW + 1,
            column=SIDE_BAR_COL,
            columnspan=CANVAS_W_GRID,
            rowspan=CANVAS_H_GRID,
            sticky="ns",
        )

        class_text = ttk.Label(frame, text="Class:")
        class_text.grid(row=0, pady=(4, 0), padx=(2 + PAD * 1.5, 2 + PAD * 1.5))
        options = [str(i) for i in range(1, 9)]
        class_btn = ttk.Spinbox(
            frame,
            textvariable=self.canvas.label_val,
            width=3,
            values=options,
            state="readonly",
            command=self.set_label_val,
        )
        # class_btn.set(str(self.canvas.label_val))
        class_btn.grid(row=1, pady=(0, PAD))

        erase_text = ttk.Label(frame, text="Erase:")
        erase_text.grid(row=2)
        erase = ttk.Checkbutton(
            frame, variable=self.canvas.erasing, command=self.on_erase_toggle
        )
        erase.grid(row=3, pady=(0, PAD))

        width_text = ttk.Label(frame, text="Width:")
        width_text.grid(row=4)

        brush_width_combo = ttk.Spinbox(
            frame,
            textvariable=self.canvas.brush_width,
            width=3,
            from_=1,
            to=60,
            increment=1,
        )
        brush_width_combo.grid(row=5)

        brush_width_slider = tk.Scale(
            frame,
            from_=1,
            to=60,
            variable=self.canvas.brush_width,
            orient="vertical",
            length=200,
            resolution=1,
            showvalue=False,
        )
        brush_width_slider.grid(row=6, pady=(1, PAD))

        clear_btn = ttk.Button(frame, text="Clear", width=6, command=self.clear)
        clear_btn.grid(row=7, pady=(0, PAD))

    def _init_bottombar(self, n_images: int = 0) -> None:
        frame = ttk.Frame(self, relief="groove", borderwidth=2)
        frame.grid(
            row=BOTTOM_BAR_IDX, column=0, columnspan=CANVAS_W_GRID + 1, sticky="ew"
        )

        opacity_text = ttk.Label(frame, text="  Opacity:")
        opacity_text.grid(column=0)

        opacity_slider = tk.Scale(
            frame,
            from_=0,
            to=1,
            variable=self.seg_overlay_alpha,
            orient="horizontal",
            length=200,
            resolution=0.01,
            showvalue=False,
            command=lambda s: self.set_overlay_alpha(),
        )
        opacity_slider.grid(column=1, row=0)

        tmp_frame = ttk.Frame(frame)

        if n_images > 1:
            image_text = ttk.Label(tmp_frame, text="Image:")
            image_text.grid(row=0, column=0)
            image_spinbox = ttk.Spinbox(
                tmp_frame,
                textvariable=self.current_piece_idx,
                from_=0,
                to=n_images - 1,
                increment=1,
                width=3,
                command=self.set_current_pice,
            )
            image_spinbox.grid(row=0, column=1)

            image_slider = tk.Scale(
                tmp_frame,
                from_=0,
                to=n_images - 1,
                variable=self.current_piece_idx,
                orient="horizontal",
                length=400,
                resolution=1,
                showvalue=False,
                command=lambda s: self.set_current_pice(int(s)),
            )
            image_slider.grid(row=0, column=2)
        tmp_frame.grid(row=0, column=2)
        frame.columnconfigure(2, weight=1)

        train_btn = ttk.Button(frame, text="Train", command=self.data_model.train_)
        train_btn.grid(column=3, row=0, pady=(4, 4))

        apply_btn = ttk.Button(frame, text="Apply")
        apply_btn.grid(column=4, row=0)

    # %% LOGIC

    def get_current_piece(self) -> Piece:
        idx = self.current_piece_idx.get()
        return self.data_model.gallery[idx]

    def set_current_pice(self, new_val: int | None = None) -> None:
        if new_val is None:
            new_val = self.current_piece_idx.get()
        new_piece = self.data_model.gallery[new_val]
        self.set_canvas_image(new_piece, False)

        self.needs_updating = True

    def on_erase_toggle(self) -> None:
        is_erasing = self.canvas.erasing.get()
        if is_erasing:
            self.set_label_val(0)
        else:
            self.set_label_val(1)

    def set_label_val(self, val: int | None = None) -> None:
        if val is None:
            val = self.canvas.label_val.get()
        self.canvas.set_label_class(val)
        return None

    def save_seg(self, path: str) -> None:
        piece = self.data_model.gallery[self.current_piece_idx.get()]
        save_segmentation(piece.seg_arr, path)

    def save_labels(self, path: str) -> None:
        piece = self.data_model.gallery[self.current_piece_idx.get()]
        save_labels(piece.labels_arr, path)

    # %% BUTTONS
    def load_image_from_filepaths(self, paths: tuple[str, ...]) -> None:
        n_imgs_prev = len(self.data_model.gallery)
        for path in paths:
            piece = self.data_model.add_image(path)
        self.set_canvas_image(piece, True)

        n_imgs = len(self.data_model.gallery)
        self.current_piece_idx.set(n_imgs - 1)
        self.set_current_pice(n_imgs - 1)
        if n_imgs > 1:
            self._init_bottombar(n_imgs)

        self.data_model.get_features(n_imgs_prev)

    # def class_changed(self, number: int) -> None:
    def clear(self) -> None:
        current_piece = self.get_current_piece()
        current_piece.labels_arr *= 0
        self.needs_updating = True

    def remove_all(self) -> None:
        self.current_piece_idx.set(0)
        self.data_model.gallery = []
        self.needs_updating = False
        self.canvas.image_available = False
        self.canvas.current_img_hw = (0, 0)
        self.canvas.destroy()
        self._init_canv()
        # self.canvas.__init__(self, self.data_model.out_queue)

    # def remove_image(self) -> None:
    #     self.ch

    # %% CANVAS
    def set_canvas_image(self, piece: Piece | None, new: bool = False) -> None:
        if piece is None:
            return
        self.canvas.set_current_image(piece.img, new)

    def add_label(self, points: list[Point]) -> None:
        idx = self.current_piece_idx.get()
        self.data_model.create_and_add_labels_from_points(
            points,
            idx,
            self.canvas.label_val.get(),
            self.canvas.brush_width.get(),
        )

    def remove_last_label(self) -> None:
        idx = self.current_piece_idx.get()
        self.data_model.remove_last_label(
            idx,
        )

    def set_overlay_alpha(self, val: float | None = None):
        if val is not None:
            self.seg_overlay_alpha.set(val)
        self.needs_updating = True

    def get_img_from_seg(
        self, train_result: np.ndarray, cmap: ListedColormap, alpha_mask: np.ndarray
    ) -> Image.Image:
        """Given a segmentation (i.e H,W arr where entries are ints), map this using the colourmaps to an image (with set opacity)."""
        cmapped = cmap(train_result)
        cmapped[:, :, 3] = alpha_mask
        cmapped = (cmapped * 255).astype(np.uint8)
        pil_image = Image.fromarray(cmapped, mode="RGBA")
        return pil_image

    def update_overlay(self) -> None:
        self.needs_updating = False
        if len(self.data_model.gallery) == 0:  # early return if no data
            return None
        current_piece = self.get_current_piece()
        h, w = (current_piece.h, current_piece.w)

        new_img = Image.new(size=(w, h), mode="RGBA")
        new_img.paste(current_piece.img, (0, 0), current_piece.img)

        if current_piece.segmented is True:
            seg_data = current_piece.seg_arr
            alpha_mask = (
                np.ones_like(seg_data, dtype=np.float16) * self.seg_overlay_alpha.get()
            )
            overlay_seg_img = self.get_img_from_seg(
                seg_data, cmap=self.cmap, alpha_mask=alpha_mask
            )
            new_img.paste(overlay_seg_img, (0, 0), overlay_seg_img)

        if current_piece.labelled is True:
            label_data = current_piece.labels_arr
            alpha_mask = (
                np.where(label_data > 0, 1, 0).astype(np.float16)
                * self.label_overlay_alpha
            )
            overlay_label_img = self.get_img_from_seg(
                label_data, cmap=self.cmap, alpha_mask=alpha_mask
            )
            new_img.paste(overlay_label_img, (0, 0), overlay_label_img)

        self.canvas.set_current_image(new_img)

    def handle_message(self, message: Message) -> None:
        header = message.category
        match header:
            case "NOTIF":
                print(message.data)
            case "POINTS":
                points = message.data
                assert type(points) is list
                if len(points) == 0:
                    return
                self.add_label(points)
            case "UNDO":
                self.remove_last_label()
            case "CLEAR":
                self.clear()
            case "SEGMENT":
                self.needs_updating = True
            case _:
                raise Exception(f"Undefined message type {header}")

    def event_loop(self) -> None:
        queue = self.data_model.out_queue
        while queue.empty() is False:
            data_in = queue.get_nowait()
            self.handle_message(data_in)
            self.needs_updating = True

        if self.needs_updating:
            self.update_overlay()
        self.loop = self.root.after(100, self.event_loop)


# %%


class MenuBar(tk.Menu):
    """Menu bar across top of GUI with dropdown commands: load data, classifiers, save segs etc."""

    def __init__(self, root: tk.Tk, app: App) -> None:
        """Attach to root then initialise all the sub menus: data, classifiers, post process & save."""
        super(MenuBar, self).__init__(
            root
        )  # done s.t the menu bar is attached to the root (tk window) rather than the frame
        self.app = app

        self.add_command(label="[microSeg]")
        self.add_separator()

        data_name_fn_pairs: list[tuple[str, Callable]] = [
            ("Add Image", self._load_images),
            ("Remove Image", _foo),
            ("Remove All", self.app.remove_all),
            ("Load labels", self._load_labels),
        ]
        data_menu = self._make_dropdown(data_name_fn_pairs)
        self.add_cascade(label="Data", menu=data_menu)

        classifier_name_fn_pairs: list[tuple[str, Callable]] = [
            ("New Classifier", self._clear_classifier),
            ("Train Classifier", self._train_classifier),
            ("Apply Classifier", self._apply_classifier),
            ("Load Classifier", self._load_classifier),
        ]
        classifier_menu = self._make_dropdown(classifier_name_fn_pairs)
        self.add_cascade(label="Classifier", menu=classifier_menu)

        # self.add_command(label="Post-Process", command=_foo)  # type: ignore

        save_name_fn_pairs: list[tuple[str, Callable]] = [
            ("Save Segmentation", self._save_segmentation),
            ("Save Labels", self._save_labels),
            ("Save Classifier", self._save_classifier),
        ]
        save_menu = self._make_dropdown(save_name_fn_pairs)
        self.add_cascade(label="Save", menu=save_menu)

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

    def _load_labels(self) -> None:
        file_path = open_file_dialog_return_fps(
            title="Load Labels",
            file_type_name="Labels",
            file_types_string=".tif .tiff .TIFF",
        )[0]
        if file_path == "":
            return
        else:
            labels = load_labels(file_path)
            idx = self.app.current_piece_idx.get()
            piece = self.app.data_model.gallery[idx]
            piece.labels_arr = labels
            piece.labelled = True
            self.app.needs_updating = True

    def _clear_classifier(self) -> None:
        self.app.data_model.classifier = None

    def _train_classifier(self) -> None:
        self.app.data_model.train_()

    def _apply_classifier(self) -> None:
        self.app.data_model.apply_()

    def _load_classifier(self) -> None:
        file_path = open_file_dialog_return_fp(title="Load Classifier")
        if file_path == "":
            return
        else:
            classifier = load_classifier(file_path)
            self.app.data_model.classifier = classifier

    def _save_classifier(self) -> None:
        f = fd.asksaveasfilename(initialfile="classifier.pkl", defaultextension=".pkl")
        if f is None:
            return
        else:
            classifier = self.app.data_model.classifier
            assert classifier is not None
            classifier.save(f)

    def _save_segmentation(self) -> None:
        n = self.app.current_piece_idx.get()
        f = fd.asksaveasfilename(initialfile=f"seg_{n}.tiff", defaultextension=".tiff")
        if f is None:
            return
        else:
            pass
            self.app.save_seg(f)

    def _save_labels(self) -> None:
        n = self.app.current_piece_idx.get()
        f = fd.asksaveasfilename(initialfile=f"seg_{n}.tiff", defaultextension=".tiff")
        if f is None:
            return
        else:
            pass
            self.app.save_labels(f)
