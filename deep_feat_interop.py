import numpy as np
from typing import Any
from torch import no_grad

from dotenv import dotenv_values


# up_chk_path = "trained_models/lu_reg_ac48.pth"
# denoiser_chk_path = "trained_models/dvt.pth"
# autoenc_chk_path = "trained_models/dac_dv2_denoised_e500.pth"


DEEP_FEATS_AVAILABLE: bool = False
try:
    from vulture import CompleteUpsampler
    from torch import no_grad

    DEEP_FEATS_AVAILABLE = True
    up_chk_path = dotenv_values()["MODEL_PATH"]

    # upsampler = CompleteUpsampler(
    #     "FEATUP",
    #     up_chk_path,
    #     device="cuda:0",
    #     to_half=True,
    #     add_flash_attn=True,
    # )

    ac_chk_path = dotenv_values().get("AC_PATH", None)

    dino_chk_path = dotenv_values().get("DINO_PATH", None)
    upsampler = CompleteUpsampler(
        "ALIBI_COMPRESSED",
        up_chk_path,
        dino_chk=dino_chk_path,
        autoencoder_chk_or_cfg=ac_chk_path,
        device="cuda:0",
        to_half=True,
        add_flash_attn=False,
    )

    denoiser_chk_path = dotenv_values().get("DENOISER_PATH", None)
    # upsampler = CompleteUpsampler(
    #     "LOFTUP_COMPRESSED",
    #     up_chk_path,
    #     autoencoder_chk_or_cfg=ac_chk_path,
    #     denoiser_chk=denoiser_chk_path,
    #     device="cuda:0",
    #     to_half=True,
    #     add_flash_attn=True,
    # )
except ImportError as e:
    print(f"Deep features unavailable: {e}")


def deep_feats(img: np.ndarray, feature_cfg: Any):
    assert upsampler
    with no_grad():
        torch_feats = upsampler.forward(img, n_batch=10)
        torch_feats = torch_feats.squeeze(0).permute((1, 2, 0))
        hr_feats = torch_feats.cpu().numpy()
    print(hr_feats.shape)
    return hr_feats
