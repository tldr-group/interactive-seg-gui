import numpy as np
from typing import Any
from torch import no_grad
from PIL import Image

from dotenv import dotenv_values


DEEP_FEATS_AVAILABLE: bool = False
try:
    from vulture import CompleteUpsampler
    from torch import no_grad

    DEEP_FEATS_AVAILABLE = True

    model_type = dotenv_values().get("MODEL_TYPE", "ALIBI_COMPRESSED")
    up_chk_path = dotenv_values()["MODEL_PATH"]
    ac_chk_path = dotenv_values().get("AC_PATH", None)
    dino_chk_path = dotenv_values().get("DINO_PATH", None)
    denoiser_chk_path = dotenv_values().get("DENOISER_PATH", None)

    if model_type == "FEATUP":
        assert up_chk_path is not None, "MODEL_PATH must be set for FEATUP model"
        upsampler = CompleteUpsampler(
            "FEATUP",
            up_chk_path,
            device="cuda:0",
            to_half=True,
            add_flash_attn=True,
        )
    elif model_type == "LOFTUP_COMPRESSED":
        assert up_chk_path is not None, "MODEL_PATH must be set for LOFTUP_COMPRESSED model"
        assert ac_chk_path is not None, "AC_PATH must be set for LOFTUP_COMPRESSED model"
        assert denoiser_chk_path is not None, "DENOISER_PATH must be set for LOFTUP_COMPRESSED model"
        upsampler = CompleteUpsampler(
            "LOFTUP_COMPRESSED",
            up_chk_path,
            autoencoder_chk_or_cfg=ac_chk_path,
            denoiser_chk=denoiser_chk_path,
            device="cuda:0",
            to_half=True,
            add_flash_attn=True,
        )
    elif model_type == "ALIBI_COMPRESSED":
        assert up_chk_path is not None, "MODEL_PATH must be set for ALIBI_COMPRESSED model"
        assert ac_chk_path is not None, "AC_PATH must be set for ALIBI_COMPRESSED model"
        upsampler = CompleteUpsampler(
            "ALIBI_COMPRESSED",
            up_chk_path,
            dino_chk=dino_chk_path,
            autoencoder_chk_or_cfg=ac_chk_path,
            device="cuda:0",
            to_half=True,
            add_flash_attn=False,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

except (ImportError, FileNotFoundError, ValueError) as e:
    print(f"Deep features unavailable: {e}")


def deep_feats(img: np.ndarray, feature_cfg: Any):
    assert upsampler
    img_rgb = Image.fromarray(img).convert("RGB")
    with no_grad():
        torch_feats = upsampler.forward(img_rgb, n_batch=10)
        torch_feats = torch_feats.squeeze(0).permute((1, 2, 0))
        hr_feats = torch_feats.cpu().numpy()
    print(hr_feats.shape)
    return hr_feats
