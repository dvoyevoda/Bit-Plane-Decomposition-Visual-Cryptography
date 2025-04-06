import numpy as np
import math
from scipy import signal
from PIL import Image

def normxcorr2D(image, template):
    """
    Normalized cross-correlation for 2D PIL images.
    Adapted from: https://github.com/JustinLiang/ComputerVisionProjects/
    
    Parameters:
    - image: A PIL image.
    - template: A PIL image. Elements cannot all be equal.
    
    Returns:
    - The mean normalized cross-correlation coefficient over the image.
    """
    t = np.asarray(template, dtype=np.float64)
    t = t - np.mean(t)
    norm = math.sqrt(np.sum(np.square(t)))
    if norm == 0:
        raise ValueError("Norm of the input is 0")
    t = t / norm

    sum_filter = np.ones(np.shape(t))
    a = np.asarray(image, dtype=np.float64)
    aa = np.square(a)
    a_sum = signal.correlate(a, sum_filter, 'same')
    aa_sum = signal.correlate(aa, sum_filter, 'same')
    numer = signal.correlate(a, t, 'same')
    denom = np.sqrt(aa_sum - np.square(a_sum) / np.size(t))
    tol = np.sqrt(np.finfo(denom.dtype).eps)
    nxcorr = np.where(denom < tol, 0, numer / denom)
    nxcorr = np.where(np.abs(nxcorr - 1.) > np.sqrt(np.finfo(nxcorr.dtype).eps), nxcorr, 0)
    return np.mean(nxcorr)

def psnr(original, contrast):
    """
    Compute the Peak Signal-to-Noise Ratio between two images.
    
    Parameters:
    - original: Numpy array of the original image.
    - contrast: Numpy array of the image to compare.
    
    Returns:
    - PSNR value in decibels.
    """
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100  # Perfect match
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))