import numpy as np
from typing import Any

from dotenv import dotenv_values


# up_chk_path = "trained_models/lu_reg_ac48.pth"
# denoiser_chk_path = "trained_models/dvt.pth"
# autoenc_chk_path = "trained_models/dac_dv2_denoised_e500.pth"
up_chk_path = dotenv_values()["MODEL_PATH"]

DEEP_FEATS_AVAILABLE: bool = False
try:
    from vulture import CompleteUpsampler
    from torch import no_grad

    DEEP_FEATS_AVAILABLE = True
    upsampler = CompleteUpsampler("FEATUP", up_chk_path, None, None, "cuda:0", True, True, True)
except ImportError as e:
    print(f"Deep features unavailable: {e}")


def deep_feats(img: np.ndarray, feature_cfg: Any):
    assert upsampler
    with no_grad():
        torch_feats = upsampler.forward(img)
        torch_feats = torch_feats.squeeze(0).permute((1, 2, 0))
        hr_feats = torch_feats.cpu().numpy()
    return hr_feats
