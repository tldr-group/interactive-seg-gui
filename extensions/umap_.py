import numpy as np
from skimage.measure import block_reduce
from multiprocessing import cpu_count
from typing import List, Callable, cast, Literal, TypeAlias
from PIL import Image
import io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
from cuml.manifold import UMAP as GPUMAP
from time import time

N_ALLOWED_CPUS = cpu_count() - 2
NormType: TypeAlias = Literal["l1", "l2", "std", None]


def plot_embedding_get_img(
    embedding: np.ndarray,  # shape (N, 2)
    image: np.ndarray,  # shape (H, W)
    label_mask: np.ndarray,  # shape (H, W), int, 0=unlabelled, 1,2,...=classes
    class_colour: List[str],  # list of matplotlib colour strings, index 0 unused
    downsample_factor: int = 2,
) -> Image.Image:
    """
    Plot embedding with points coloured by class_colour for labelled points, and intensity for unlabelled.
    Returns a PIL Image of the plot.
    """
    H, W = image.shape
    embedding_flat = embedding.reshape(-1, 2)

    label_mask = downsample_array(label_mask, downsample_factor, np.max)
    label_mask_flat = label_mask.flatten()

    image_down = downsample_array(image, downsample_factor, np.mean)
    intensity = image_down.flatten()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 8)
    plt.axis("off")
    ax = plt.gca()

    # Plot unlabelled points (label==0) with intensity
    unlabelled_idx = label_mask_flat == 0
    plt.scatter(
        embedding_flat[unlabelled_idx, 0],
        embedding_flat[unlabelled_idx, 1],
        s=1,
        c=intensity[unlabelled_idx],
        cmap="viridis",
        alpha=1,
        label="Unlabelled",
    )

    # Plot labelled points by class colour
    for class_id in range(1, len(class_colour)):
        idx = label_mask_flat == class_id
        if np.any(idx):
            plt.scatter(
                embedding_flat[idx, 0],
                embedding_flat[idx, 1],
                marker="v",
                s=3,
                c=class_colour[class_id],
                label=f"Class {class_id}",
                edgecolors="white",
                linewidths=0.1,
            )

    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.autoscale(False)

    # Convert plot to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True, pad_inches=0.0, dpi=300)
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGBA")
    buf.close()
    return pil_img


def normalize_array(arr: np.ndarray, norm: NormType = "l2") -> np.ndarray:
    match norm:
        case "l1":
            norm_values = np.linalg.norm(arr, ord=1, axis=-1, keepdims=True)
            return arr / (norm_values + 1e-8)
        case "l2":
            norm_values = np.linalg.norm(arr, ord=2, axis=-1, keepdims=True)
            return arr / (norm_values + 1e-8)
        case "std":
            mean = np.mean(arr, axis=-1, keepdims=True)
            std = np.std(arr, axis=-1, keepdims=True)
            return (arr - mean) / (std + 1e-8)
        case None:
            return arr


def downsample_array(
    arr: np.ndarray, downsample_factor: int = 2, reduce_func: Callable[[np.ndarray], np.ndarray] = np.mean
) -> np.ndarray:
    # arr shape: (H, W, C) or (H, W)
    if arr.ndim == 3:
        block_size = (downsample_factor, downsample_factor, 1)
    elif arr.ndim == 2:
        block_size = (downsample_factor, downsample_factor)
    else:
        raise ValueError("Input array must be 2D or 3D.")
    return block_reduce(arr, block_size=block_size, func=reduce_func)


def sample_array(arr: np.ndarray, n_samples: int) -> np.ndarray:
    # Flatten if arr is not 2D
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    idx = np.random.permutation(arr.shape[0])[:n_samples]
    return arr[idx]


def pca_reduce(
    arr: np.ndarray,
    k: int,
    n_samples: int = -1,
    random_state: int = 42,
) -> np.ndarray:
    # Flatten if arr is not 2D
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])

    if n_samples > -1 and n_samples < arr.shape[0]:
        sample = sample_array(arr, n_samples)
        pca = PCA(n_components=k, random_state=random_state)
        pca.fit(sample)
        reduced = pca.transform(arr)
    else:
        pca = PCA(n_components=k, random_state=random_state)
        reduced = pca.fit_transform(arr)
    return reduced


def get_umap_embedding(
    features: np.ndarray,
    labelled_mask: np.ndarray | None = None,
    n_neighbors: int = 20,
    min_dist: float = 0.01,
    n_components: int = 2,
    metric: str = "euclidean",
    downsample_factor: int = 2,
    norm: NormType = "l2",
    pca_dim: int = -1,
    use_gpu: bool = True,
    cast_to: np.dtype = np.float16,
) -> np.ndarray:
    start = time()
    # Downsample features
    if downsample_factor > 1:
        downsampled_features = downsample_array(features, downsample_factor, np.mean)
    else:
        downsampled_features = features
    downsampled_features = normalize_array(downsampled_features, norm)
    downsampled_features = downsampled_features.astype(cast_to)

    h, w, c = downsampled_features.shape
    flat_features = downsampled_features.reshape((h * w, c))
    flat_mask: np.ndarray | None = None
    if labelled_mask is not None:
        downsampled_mask = downsample_array(labelled_mask, downsample_factor, np.max)
        flat_mask = downsampled_mask.reshape((h * w,))
        print(np.count_nonzero(flat_mask))
        flat_mask[flat_mask == 0] = -1  # UMAP expects unlabelled data to be -1

    if pca_dim > 0:
        flat_features = pca_reduce(flat_features, pca_dim, n_samples=40000)

    # Perform UMAP embedding
    if use_gpu:
        umap_model = GPUMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            n_jobs=N_ALLOWED_CPUS,
        )
        embedding = umap_model.fit_transform(flat_features, y=flat_mask)
    else:
        umap_model = UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, n_jobs=N_ALLOWED_CPUS
        )
        embedding = umap_model.fit_transform(flat_features, y=flat_mask)
        embedding = cast(np.ndarray, embedding)

    # Reshape embedding back to image grid
    embedding_image = embedding.reshape((h, w, n_components))
    end = time()
    print(f"UMAP embedding took {end - start:.2f} seconds (downsample={downsample_factor}).")
    return embedding_image
