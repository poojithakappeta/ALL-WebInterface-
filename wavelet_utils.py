# wavelet_utils.py
import numpy as np
import torch
import pywt
from PIL import Image

def extract_wavelet_features(image, wavelet_dim=4096):
    # Convert to grayscale
    image = np.array(image.convert("L"), dtype=np.float32)
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    features = np.concatenate([LL.flatten(), LH.flatten(), HL.flatten(), HH.flatten()])
    if features.size >= wavelet_dim:
        features = features[:wavelet_dim]
    else:
        features = np.pad(features, (0, wavelet_dim - features.size))
    return torch.tensor(features, dtype=torch.float32)
