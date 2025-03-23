import numpy as np
from typing import Any

from yoeo import get_dv2_model, get_upsampler_and_expr, get_hr_feats

DEVICE = "cuda:0"
dv2 = get_dv2_model(True, device=DEVICE)

model_path = "yoeo_models/e5000_fit_reg_f64.pth"
cfg_path = "yoeo_models/upsampler_fewer_features.json"
upsampler, expr = get_upsampler_and_expr(model_path, cfg_path, device=DEVICE)


def deep_feats(img: np.ndarray, feature_cfg: Any):
    torch_feats = get_hr_feats(img, dv2, upsampler, n_batch_lr=10)
    torch_feats = torch_feats.squeeze(0).permute((1, 2, 0))
    hr_feats = torch_feats.cpu().numpy()
    return hr_feats
