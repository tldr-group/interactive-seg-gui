import numpy as np
from queue import Queue
from tifffile import imread

from skimage.draw import ellipse
from PIL import Image
from os import getcwd, mkdir
from os.path import exists
from shutil import rmtree
from multiprocessing import Queue as MPQueue

from typing import TypeAlias
from gui_elements.constants import N_PREVIEW_SLICES, Message
from dataclasses import dataclass
from dotenv import dotenv_values


from interactive_seg_backend.classifiers import Classifier
from interactive_seg_backend.configs import (
    FeatureConfig,
    TrainingConfig,
    load_training_config_json,
)
from interactive_seg_backend.configs.config import KEYS_TO_CLASSES
from interactive_seg_backend.file_handling import load_featurestack
from interactive_seg_backend.core import (
    get_training_data,
    shuffle_sample_training_data,
    get_model,
    train,
)
from interactive_seg_backend.main import featurise, apply

from extensions.umap_ import get_umap_embedding
from deep_feat_interop import deep_feats, DEEP_FEATS_AVAILABLE


Point: TypeAlias = tuple[float, float]

CWD = getcwd()
# DEFAULT_FEAT_CONFIG = FeatureConfig(mean=True, minimum=True, maximum=True)
# DEFAULT_TRAIN_CONFIG = TrainingConfig(DEFAULT_FEAT_CONFIG, CRF=True, classifier="xgb")
CFG_PATH = dotenv_values()["CFG_PATH"]
DEFAULT_TRAIN_CONFIG = load_training_config_json(CFG_PATH, KEYS_TO_CLASSES)
print(DEFAULT_TRAIN_CONFIG)

# set_start_method("spawn", force=True)


@dataclass
class Label:
    x0: int
    y0: int
    bbox: tuple[int, int, int, int]
    diff: np.ndarray


def draw_points_get_arr(
    points: list[Point],
    box_h: int,
    box_w: int,
    y0: int,
    x0: int,
    label_val: int,
    brush_width: int,
) -> np.ndarray:
    temp_arr = np.zeros((box_h, box_w), dtype=np.int16) - 1
    for p in points:
        rr, cc = ellipse(p[1] - y0, p[0] - x0, brush_width, brush_width)
        temp_arr[rr, cc] = label_val
    return temp_arr


def label_from_points(
    points: list[Point],
    seg_arr: np.ndarray,
    label_val: int,
    brush_width: int,
    update_seg_arr: bool = True,
    non_overwriting: bool = True,
) -> Label:
    o = brush_width
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, y0 = int(min(xs)) - o, int(min(ys)) - o
    x1, y1 = int(max(xs)) + o, int(max(ys)) + o

    arr_h, arr_w = seg_arr.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(arr_w - 1, x1), min(arr_h - 1, y1)

    h, w = (y1 - y0), (x1 - x0)
    bbox = (x0, y0, x1, y1)

    new_label = draw_points_get_arr(points, h, w, y0, x0, label_val, brush_width)
    prev_state = seg_arr[y0:y1, x0:x1]

    diff = new_label - prev_state
    diff *= new_label >= 0

    erasing = label_val == 0
    if non_overwriting and not erasing:
        diff *= ~(prev_state > 0)

    if update_seg_arr:
        seg_arr[y0:y1, x0:x1] += diff
    return Label(x0, y0, bbox, diff)


@dataclass
class Piece:
    """
    Piece.

    Fundamental unit of program. Holds the data associated with the image in $arr, a PIL image of the arr in $img.
    Labels is a list of label objects that belong to this piece. __post_init__ adds some mutable objects like
    segmentations and grid_points that are useful later.
    """

    img: Image.Image
    img_arr: np.ndarray

    labels: list[Label]
    labelled: bool = False
    segmented: bool = False

    def __post_init__(self) -> None:
        """Set these here because dataclasses don't like mutable objects being assigned in __init__."""
        shape: tuple[int, ...] = self.img_arr.shape
        self.h: int = shape[0]
        self.w: int = shape[1]

        # integer arr where 0 = not labelled and N > 0 indicates a label for class N at that pixel
        self.labels_arr: np.ndarray = np.zeros((self.h, self.w), dtype=np.int16)
        # integer arr where value N at pixel P indicates the classifier thinks P is class N
        self.seg_arr: np.ndarray = np.zeros((self.h, self.w), dtype=np.int16)

        # boolean arr where 1 = show this pixel in the overlay and 0 means hide. Used for hiding/showing labels later.
        self.label_alpha_mask = np.ones_like(self.seg_arr, dtype=bool)


def check_if_arr_is_volume(arr: np.ndarray) -> bool:
    shape = arr.shape
    if len(shape) == 3:
        if shape[0] == 1 or shape[-1] == 1:
            return False
        elif shape[-1] == 3 or shape[-1] == 4:
            return False
        else:
            return True
    else:
        return False


def train_from_paths(feature_paths: list[str], labels: list[np.ndarray]) -> Classifier:
    tc = DEFAULT_TRAIN_CONFIG
    print(feature_paths)
    fit, target = get_training_data(feature_paths, labels)
    fit, target = shuffle_sample_training_data(fit, target, tc.shuffle_data, tc.n_samples)
    classifier = get_model(tc.classifier, tc.classifier_params)
    classifier = train(classifier, fit, target, sample_weight=None)
    return classifier


class DataModel(object):
    def __init__(self) -> None:
        self.in_queue: MPQueue[Message] = MPQueue(maxsize=40)
        self.out_queue: Queue[Message] = Queue(maxsize=40)

        init_msg = Message("NOTIF", "microSeg v0.01 04/03/25")
        self.out_queue.put(init_msg)

        self.gallery: list[Piece] = []

        self.classifier: Classifier | None = None

        self.cache_dir = f"{CWD}/.isb_tmp"
        try:
            mkdir(self.cache_dir)
        except FileExistsError:
            rmtree(self.cache_dir)
            mkdir(self.cache_dir)

    # %% I/O
    def add_image(self, filepath: str, add_to_gallery: bool = True) -> Piece:
        extension: str = filepath.split(".")[-1]
        if extension.lower() not in ["png", "jpg", "jpeg", "tif", "bmp", "tiff"]:
            raise Exception(f".{extension} is not a valid image file format")

        if extension.lower() in ["tiff", "tif"]:
            np_array: np.ndarray = imread(filepath)  # type: ignore
            if check_if_arr_is_volume(np_array):
                return self.add_volume(np_array)
            else:
                pil_image = Image.fromarray(np_array).convert("RGBA")
        else:
            pil_image = Image.open(filepath)
            np_array = np.array(pil_image)
            pil_image = pil_image.convert("RGBA")

        new_piece = Piece(pil_image, np_array, [], False, False)
        if add_to_gallery:
            self.gallery.append(new_piece)

        return new_piece

    def add_volume(self, arr: np.ndarray, add_to_gallery: bool = True) -> Piece:
        n_slices = arr.shape[0]
        n_preview = min(N_PREVIEW_SLICES, n_slices)
        slices = np.linspace(0, n_slices - 1, num=n_preview, endpoint=True, dtype=np.uint16)
        for idx in slices:
            slice_arr = arr[idx]
            pil_image = Image.fromarray(slice_arr).convert("RGBA")
            new_piece = Piece(pil_image, slice_arr, [], False, False)
            if add_to_gallery:
                self.gallery.append(new_piece)
        return new_piece

    def remove_image(self, idx: int) -> None:
        self.gallery.pop(idx)

    # %% LABELLING
    def create_and_add_labels_from_points(
        self, points: list[Point], piece_idx: int, label_val: int, brush_width: int
    ) -> None:
        piece = self.gallery[piece_idx]
        label = label_from_points(points, piece.labels_arr, label_val, brush_width, True)
        piece.labels.append(label)
        piece.labelled = True

    def remove_last_label(self, piece_idx: int) -> None:
        piece = self.gallery[piece_idx]
        label = piece.labels.pop(-1)
        x0, y0, x1, y1 = label.bbox
        piece.labels_arr[y0:y1, x0:x1] -= label.diff

        if len(piece.labels) == 0:
            piece.labelled = False

    # %% UMAP
    def do_umap(self, idx: int) -> np.ndarray:
        features = load_featurestack(
            f"{self.cache_dir}/feature_stack_{idx}.npy",
        )
        labels = self.gallery[idx].labels_arr
        embedding = get_umap_embedding(features, labels)
        return embedding

        # want: dim reduce data (to 2 dims?)
        # also want to do supervised dim reduction with existing labels
        # also want to be able to do clustering i.e hdbscan for assignment?
        # and / or nearest neighbour in feature space assignment

    # %% CLASSIFIER INTEROP
    def get_features(self, prev_n: int) -> None:
        start_idx = max(0, prev_n - 1)
        inds = [prev_n + i for i in range(len(self.gallery[start_idx:]))]
        pieces = [self.gallery[i] for i in inds]
        self._get_features(pieces, inds)
        print("Finished featurising")

    def _get_features(self, pieces: list[Piece], save_inds: list[int]) -> None:
        if DEFAULT_TRAIN_CONFIG.add_dino_features and DEEP_FEATS_AVAILABLE:
            extra_feats = [(deep_feats, False)]
        else:
            extra_feats = []

        for idx, piece in zip(save_inds, pieces):
            featurise(
                piece.img_arr,
                DEFAULT_TRAIN_CONFIG,
                False,
                f"{self.cache_dir}/feature_stack_{idx}.npy",
                extra_feats,
            )

    def train_(self) -> None:
        # no point in threading any of these if you just call join lol
        paths: list[str] = []
        labels: list[np.ndarray] = []
        for i, piece in enumerate(self.gallery):
            if piece.labelled:
                paths.append(f"{self.cache_dir}/feature_stack_{i}.npy")
                labels.append(piece.labels_arr)

        for path in paths:
            stack_exists = exists(path)
            if not stack_exists:
                # TODO: make this explict warning
                print("Not finished featurising!")
                return

        classifier = train_from_paths(paths, labels)
        self.classifier = classifier

        self.apply_()

    def apply_(self) -> None:
        assert self.classifier is not None
        for i, piece in enumerate(self.gallery):
            path = f"{self.cache_dir}/feature_stack_{i}.npy"
            stack = load_featurestack(path)
            seg, _ = apply(
                self.classifier,
                stack,
                DEFAULT_TRAIN_CONFIG,
                image=piece.img_arr,
                labels=piece.labels_arr,
            )
            piece.seg_arr = seg + 1
            piece.segmented = True
        self.out_queue.put(Message("SEGMENT", None))

    def reload_cfg(self, verbose: bool = True) -> None:
        global DEFAULT_TRAIN_CONFIG
        DEFAULT_TRAIN_CONFIG = load_training_config_json(CFG_PATH, KEYS_TO_CLASSES)
        if verbose:
            print(DEFAULT_TRAIN_CONFIG)
