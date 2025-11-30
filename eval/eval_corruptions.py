import os, io, json, math
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm


TOPK_SCRIPT_PATH = "topk_noise_analysis.py"

# load function defs from the above script 
import importlib.util
spec = importlib.util.spec_from_file_location("topk_noise_analysis", TOPK_SCRIPT_PATH)
topk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(topk)

# Load the required functions
build_model = topk.build_model
find_best_checkpoint = topk.find_best_checkpoint
load_checkpoint_into_model = topk.load_checkpoint_into_model
evaluate_topk = topk.evaluate_topk
print_topk_report = topk.print_topk_report

device = getattr(topk, "device", (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))


### load all corruptions
from corruptions import (corruption_gaussian_noise,
                        corruption_shot_noise,
                        corruption_impulse_noise,
                        corruption_motion_blur,
                        corruption_gaussian_blur,
                         corruption_defocus_blur,
                        corruption_stroke_thinning,
                        corruption_contrast,
                        corruption_pixelate,
                        corruption_elastic)

def apply_corruption(img: Image.Image, corruption: str, severity: int) -> Image.Image:
    if corruption in ("clean", None):
        return img

    severity = max(1, min(5, int(severity)))
    if corruption == "gaussian_noise":
        return corruption_gaussian_noise(img, severity)
    if corruption == "shot_noise":
        return corruption_shot_noise(img, severity)
    if corruption == "impulse_noise":
        return corruption_impulse_noise(img, severity)
    if corruption == "motion_blur":
        return corruption_motion_blur(img, severity)
    if corruption == "gaussian_blur":
        return corruption_gaussian_blur(img, severity)
    if corruption == "defocus_blur":
        return corruption_defocus_blur(img, severity)
    if corruption == "stroke_thinning":
        return corruption_stroke_thinning(img, severity)
    if corruption == "elastic":
        return corruption_elastic(img, severity)
    if corruption == "pixelate":
        return corruption_pixelate(img, severity)
    if corruption == "contrast":
        return corruption_contrast(img, severity)
    if corruption.startswith("scale"):
        size_map = {1:128, 2:64, 3:32, 4:16, 5:8}
        target = size_map.get(int(severity), 32)
        w, h = img.size
        small = img.resize((target, target), resample=Image.BILINEAR)
        up = small.resize((w, h), resample=Image.BICUBIC)
        return up
    return img

# Cell 4 — make_transform, CorruptionDataset and loader
def make_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

class CorruptionDataset(ImageFolder):
    def __init__(self, root, corruption="clean", severity=1, transform=None):
        super().__init__(root, transform=transform)
        self.corruption = corruption
        self.severity = severity

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path).convert("RGB")
        img = apply_corruption(img, self.corruption, self.severity)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def make_corruption_loader(test_dir, corruption, severity, batch_size=128, num_workers=4, img_size=224):
    tfm = make_transform(img_size)
    ds = CorruptionDataset(test_dir, corruption=corruption, severity=severity, transform=tfm)
    # stable ordering; seed generator for reproducibility if desired
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader, len(ds.classes)




def evaluate_models_under_noise(model_names, ckpt_dirs, test_dir, corruptions, severities=(1,2,3,4,5),
                                batch_size=128, num_workers=4, ks=(1,3,5), img_size=224):
    results = {}
    # probe dataset classes to know expected num_classes if needed
    probe_loader, num_classes_ds = make_corruption_loader(test_dir, "clean", 1, batch_size=1, num_workers=0, img_size=img_size)
    for name in model_names:
        print(f"\n=== Model: {name} ===")
        ckpt_dir = ckpt_dirs[name]
        ckpt_path = find_best_checkpoint(ckpt_dir)
        print("Using checkpoint:", ckpt_path)
        # load checkpoint minimally to get num_classes if saved
        raw = torch.load(ckpt_path, map_location="cpu")
        num_classes_ckpt = int(raw.get("num_classes", num_classes_ds)) if isinstance(raw, dict) else num_classes_ds
        model = build_model(name, num_classes=num_classes_ckpt, pretrained=False)
        meta = load_checkpoint_into_model(model, ckpt_path)
        model.to(device)
        model.eval()

        model_results = {}
        for corruption in corruptions:
            model_results[corruption] = {}
            for sev in severities:
                print(f"-> Evaluating {corruption}, severity={sev}")
                loader, _ = make_corruption_loader(test_dir, corruption, sev, batch_size=batch_size, num_workers=num_workers, img_size=img_size)
                accs = evaluate_topk(model, loader, device, ks=ks)
                print_topk_report(f"{name} | {corruption} (s={sev})", accs)
                model_results[corruption][sev] = accs
        results[name] = model_results
    return results

def main():
    test_dir = "../dataset/test/"     # <- update if needed
    ckpt_dirs = {
        "efficientnet_b0": "outputs/efficientnet_b0/checkpoints/",
        "googlenet": "outputs/googlenet/checkpoints/",
        "mnasnet0_5": "outputs/mnasnet0_5/checkpoints/",
        "squeezenet1_0": "outputs/squeezenet1_0/checkpoints/",
        "swin_t": "outputs/swin_t/checkpoints/",
        "regnet_x_400mf": "outputs/regnet_x_400mf/checkpoints/",
        "convnext_tiny": "outputs/convnext_tiny/checkpoints/",
        "shufflenet_v2_x0_5": "outputs/shufflenet_v2_x0_5/checkpoints/",
        "vgg16_bn": "outputs/vgg16_bn/checkpoints/"
    }
    model_names = list(ckpt_dirs.keys())
    batch_size = 128
    num_workers = 4
    ks = (1, 3, 5)
    corruptions = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "gaussian_blur", "defocus_blur", "stroke_thinning", "elastic",
    "pixelate", "contrast", "scale"
    ]
    severities = [1,2,3,4,5]
    
    results_noise = evaluate_models_under_noise(
        model_names, ckpt_dirs, test_dir,
        corruptions=corruptions,
        severities=severities,
        batch_size=batch_size,
        num_workers=num_workers,
        ks=ks,
        img_size=224
    )
    
    # save
    out_path = "noise_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(results_noise, f, indent=2)
    print(f"Saved results -> {out_path}")
    
if __name__ == "__main__":
    main()



