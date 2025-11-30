import os, io, json, math
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

TOPK_SCRIPT_PATH = "topk_noise_analysis.py"


import importlib.util
spec = importlib.util.spec_from_file_location("topk_noise_analysis", TOPK_SCRIPT_PATH)
topk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(topk)

# Expose frequently-used names from the module
build_model = topk.build_model
find_best_checkpoint = topk.find_best_checkpoint
load_checkpoint_into_model = topk.load_checkpoint_into_model
evaluate_topk = topk.evaluate_topk
print_topk_report = topk.print_topk_report


### function definitions for all noises considered study

def zhang_suen_thinning(binary):
    """
    Zhang-Suen thinning algorithm (numpy).
    Input: binary image (2D numpy array) with foreground==1, background==0
    Returns: thinned binary image (foreground==1)
    """
    # make a copy to avoid modifying input
    img = binary.copy().astype(np.uint8)
    prev = np.zeros(img.shape, np.uint8)
    diff = None

    # helper neighbors (8-neighborhood)
    rows, cols = img.shape
    def neighbors(y, x):
        # order: P2 P3 P4 P5 P6 P7 P8 P9 as in paper
        return [img[y-1, x]   if y-1>=0 else 0,
                img[y-1, x+1] if (y-1>=0 and x+1<cols) else 0,
                img[y,   x+1] if x+1<cols else 0,
                img[y+1, x+1] if (y+1<rows and x+1<cols) else 0,
                img[y+1, x]   if y+1<rows else 0,
                img[y+1, x-1] if (y+1<rows and x-1>=0) else 0,
                img[y,   x-1] if x-1>=0 else 0,
                img[y-1, x-1] if (y-1>=0 and x-1>=0) else 0]

    def count_nonzero_neigh(nbs):
        return sum(nbs)

    def count_zero_to_one_transitions(nbs):
        # counts number of 0->1 transitions in the ordered sequence P2..P9..P2
        seq = nbs + [nbs[0]]
        transitions = sum((seq[i] == 0 and seq[i+1] == 1) for i in range(len(nbs)))
        return transitions

    changing = True
    while changing:
        changing = False
        to_remove = []
        # step 1
        for y in range(1, rows-1):
            for x in range(1, cols-1):
                P1 = img[y, x]
                if P1 != 1:
                    continue
                nbs = neighbors(y, x)
                C = count_nonzero_neigh(nbs)
                if C < 2 or C > 6:
                    continue
                T = count_zero_to_one_transitions(nbs)
                if T != 1:
                    continue
                if not (nbs[0] * nbs[2] * nbs[4] == 0):
                    continue
                if not (nbs[2] * nbs[4] * nbs[6] == 0):
                    continue
                to_remove.append((y, x))
        if to_remove:
            changing = True
            for (y, x) in to_remove:
                img[y, x] = 0

        # step 2
        to_remove = []
        for y in range(1, rows-1):
            for x in range(1, cols-1):
                P1 = img[y, x]
                if P1 != 1:
                    continue
                nbs = neighbors(y, x)
                C = count_nonzero_neigh(nbs)
                if C < 2 or C > 6:
                    continue
                T = count_zero_to_one_transitions(nbs)
                if T != 1:
                    continue
                if not (nbs[0] * nbs[2] * nbs[6] == 0):
                    continue
                if not (nbs[0] * nbs[4] * nbs[6] == 0):
                    continue
                to_remove.append((y, x))
        if to_remove:
            changing = True
            for (y, x) in to_remove:
                img[y, x] = 0

    return img

def _clamp_img_array(arr):
    """Clamp and convert image array to uint8."""
    return np.uint8(np.clip(arr, 0, 255))

def corruption_gaussian_noise(img: Image.Image, severity: int) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    sigma = [10, 20, 30, 40, 60][max(0, min(4, severity - 1))]
    noise = np.random.normal(0, sigma, arr.shape)
    arr += noise
    arr = _clamp_img_array(arr)
    return Image.fromarray(arr)

def corruption_shot_noise(img: Image.Image, severity: int) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    vals = [60, 40, 25, 15, 8][max(0, min(4, severity-1))]
    noisy = np.random.poisson(arr * vals) / float(vals)
    arr = _clamp_img_array((noisy * 255.0).astype(np.float32))
    return Image.fromarray(arr)

def corruption_impulse_noise(img: Image.Image, severity: int) -> Image.Image:
    arr = np.array(img).copy()
    p = [0.03, 0.06, 0.12, 0.18, 0.25][max(0, min(4, severity-1))]
    mask = np.random.choice([0, 1, 2], size=arr.shape[:2], p=[1-p, p/2, p/2])
    noisy = arr.copy()
    noisy[mask==1] = 0
    noisy[mask==2] = 255
    return Image.fromarray(noisy)

def corruption_motion_blur(img: Image.Image, severity: int) -> Image.Image:
    size = [3, 5, 11, 15, 21][max(0, min(4, severity-1))]
    kernel = np.zeros((size, size))
    kernel[size//2, :] = np.ones(size)
    kernel = kernel / kernel.sum()
    pil_kernel = ImageFilter.Kernel((size, size), kernel.flatten().tolist(), scale=1)
    return img.filter(pil_kernel)

def corruption_gaussian_blur(img: Image.Image, severity: int) -> Image.Image:
    radius = [0.7, 1.5, 2.5, 3.5, 5.0][max(0, min(4, severity-1))]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def corruption_defocus_blur(img: Image.Image, severity: int) -> Image.Image:
    radius = [1, 2, 3, 4, 6][max(0, min(4, severity-1))]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))



def corruption_stroke_thinning(img:Image.Image,severity:int) -> Image.Image:
    sev = max(1, min(5, int(severity)))
    max_kernel = 9
    k = int(round(max_kernel * (1.0 - (sev - 1) / 4.0)))
    if k % 2 == 0:
        k = max(1, k-1)
    dilate_size = max(1, k)
    img_rgb = img.convert("RGB")
    gray = img_rgb.convert("L")
    arr = np.array(gray).astype(np.uint8)
    mean = arr.mean()
    std = arr.std()
    thresh = mean - k * std
    bin_mask = (arr < thresh).astype(np.uint8)
    if bin_mask.sum() == 0:
        thresh = arr.mean()
        bin_mask = (arr < thresh).astype(np.uint8)
    skeleton = zhang_suen_thinning(bin_mask)
    sk_pil = Image.fromarray((skeleton * 255).astype(np.uint8))
    dilated = sk_pil.filter(ImageFilter.MaxFilter(size=dilate_size))
    dilated_arr = np.array(dilated).astype(np.uint8)
    fg_mask = (dilated_arr == 255)
    img_arr = np.array(img_rgb).astype(np.uint8)
    fg3 = np.stack([fg_mask, fg_mask, fg_mask], axis=-1)
    out = img_arr.copy()
    out[~fg3] = 255
    return Image.fromarray(out)

def corruption_contrast(img: Image.Image, severity: int) -> Image.Image:
    factor = [0.7, 0.5, 0.35, 0.2, 0.1][max(0, min(4, severity-1))]
    arr = np.array(img).astype(np.float32)
    mean = arr.mean(axis=(0,1), keepdims=True)
    res = (arr - mean) * factor + mean
    return Image.fromarray(_clamp_img_array(res))

def corruption_pixelate(img: Image.Image, severity: int) -> Image.Image:
    scale = [0.9, 0.7, 0.5, 0.33, 0.2][max(0, min(4, severity-1))]
    w, h = img.size
    small = img.resize((max(1, int(w*scale)), max(1, int(h*scale))), resample=Image.NEAREST)
    return small.resize(img.size, Image.NEAREST)

def corruption_elastic(img: Image.Image, severity: int) -> Image.Image:
    w, h = img.size
    max_shift = [1, 2, 3, 4, 6][max(0, min(4, severity-1))]
    dx = np.random.uniform(-max_shift, max_shift)
    dy = np.random.uniform(-max_shift, max_shift)
    angle = np.random.uniform(-max_shift*2, max_shift*2)
    return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy)).rotate(angle, resample=Image.BILINEAR)
