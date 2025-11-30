from torch_cka import CKA

import os, glob, io, random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (     
    mnasnet0_5, MNASNet0_5_Weights,     
    convnext_tiny, ConvNeXt_Tiny_Weights, 
    
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AddGaussianNoise:
    """
    Additive Gaussian noise in image space (uint8 domain before normalization).
    - mean, std in 0..255 scale if apply_before_to_tensor=True
    - If apply_before_to_tensor=False, expects tensor in 0..1 and uses 0..1 scale.
    """
    def __init__(self, mean=0.0, std=8.0, apply_before_to_tensor=True, clip=True):
        self.mean = float(mean)
        self.std = float(std)
        self.apply_before_to_tensor = bool(apply_before_to_tensor)
        self.clip = bool(clip)

    def __call__(self, img):
        if self.apply_before_to_tensor:
            # PIL Image -> numpy uint8
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(self.mean, self.std, size=arr.shape).astype(np.float32)
            arr_noisy = arr + noise
            if self.clip:
                arr_noisy = np.clip(arr_noisy, 0, 255)
            arr_noisy = arr_noisy.astype(np.uint8)
            return Image.fromarray(arr_noisy)
        else:
            # img is expected to be a torch tensor in [0,1]
            import torch
            noise = torch.randn_like(img) * (self.std) + self.mean
            x = img + noise
            if self.clip:
                x = x.clamp(0.0, 1.0)
            return x

def build_test_transforms(invert: bool = False, add_noise: bool = False, noise_std: float = 8.0):
    """
    Clean eval pipeline with optional inversion and optional Gaussian noise.
    Noise is applied before ToTensor by default (pixel scale), then normalized.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfms = []
    if add_noise:
        tfms.append(AddGaussianNoise(mean=0.0, std=noise_std, apply_before_to_tensor=True, clip=True))
    tfms.append(transforms.Resize(256))
    tfms.append(transforms.CenterCrop(224))
    if invert:
        tfms.append(transforms.Lambda(lambda img: Image.fromarray(255 - np.array(img))))
 
    tfms.append(transforms.ToTensor())
    tfms.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(tfms)

def make_test_loader(test_dir: str, batch_size: int = 128, num_workers: int = 4, invert: bool = False, add_noise: bool = False, noise_std: float = 8.0):
    tfm = build_test_transforms(invert=invert, add_noise=add_noise, noise_std=noise_std)
    ds = ImageFolder(test_dir, transform=tfm)
    g = torch.Generator()
    g.manual_seed(0)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=lambda wid: None, generator=g
    )
    return loader, len(ds.classes), ds.classes, ds.class_to_idx


def find_best_checkpoint(ckpt_dir: str) -> str:
    cand = os.path.join(ckpt_dir, "best.pt")
    if os.path.isfile(cand):
        return cand
    # Allow fallback to strict match of best.pt anywhere inside the dir
    matches = glob.glob(os.path.join(ckpt_dir, "**", "best.pt"), recursive=True)
    if matches:
        return max(matches, key=os.path.getmtime)
    raise FileNotFoundError(f"best.pt not found under {ckpt_dir}")

def cleanup_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        elif k.startswith("model."):
            new_state[k[len("model."):]] = v
        else:
            new_state[k] = v
    return new_state

def load_checkpoint_into_model(model: nn.Module, ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta: Dict[str, Any] = {}
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")
    state = None
    for k in ["model_state_dict", "state_dict", "model", "net", "weights"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            print(k)
            state = ckpt[k]
            break
    if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    if state is None:
        raise RuntimeError(f"Could not locate model weights in checkpoint keys: {list(ckpt.keys())}")
    # state = cleanup_state_dict_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        print("Load notice -> missing:", missing, "| unexpected:", unexpected)
    for k in ("epoch","best_epoch","metrics","args","config","class_to_idx","num_classes"):
        if k in ckpt:
            meta[k] = ckpt[k]
    return meta


def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    n = model_name.lower()
    if n == "efficientnet_b0":
        w = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes, bias=True)
    elif n == "googlenet":
        w = GoogLeNet_Weights.DEFAULT if pretrained else None
        m = googlenet(weights=w, aux_logits=False)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(nn.Linear(in_features, num_classes, bias=True))
    elif n == "mnasnet0_5":
        w = MNASNet0_5_Weights.DEFAULT if pretrained else None
        m = mnasnet0_5(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes, bias=True)
    elif n == "squeezenet1_0":
        w = SqueezeNet1_0_Weights.DEFAULT if pretrained else None
        m = squeezenet1_0(weights=w)
        m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    elif n == "swin_t":
        w = Swin_T_Weights.DEFAULT if pretrained else None
        m = swin_t(weights=w)
        m.head = nn.Linear(m.head.in_features, num_classes)
    elif n == "regnet_x_400mf":
        w = RegNet_X_400MF_Weights.DEFAULT if pretrained else None
        m = regnet_x_400mf(weights=w)
        m.fc = nn.Sequential(nn.Linear(m.fc.in_features, num_classes))
    elif n == "convnext_tiny":
        w = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = convnext_tiny(weights=w)
        in_features = m.classifier[2].in_features
        m.classifier[2] = nn.Sequential(nn.Linear(in_features, num_classes, bias=True))
    elif n == "shufflenet_v2_x0_5":
        w = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
        m = shufflenet_v2_x0_5(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif n == "vgg16_bn":
        w = VGG16_BN_Weights.DEFAULT if pretrained else None
        m = vgg16_bn(weights=w)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return m


model_names = [
    "efficientnet_b0",
    "googlenet",
    "mnasnet0_5",
    "squeezenet1_0",
    "swin_t",
    "regnet_x_400mf",
    "convnext_tiny",
    "shufflenet_v2_x0_5",
]
ckpt_dirs = {
    "efficientnet_b0": "outputs/efficientnet_b0/checkpoints/",
    "googlenet": "outputs/googlenet/checkpoints/",
    "mnasnet0_5": "outputs/mnasnet0_5/checkpoints/",
    "squeezenet1_0": "outputs/squeezenet1_0/checkpoints/",
    "swin_t": "outputs/swin_t/checkpoints/",
    "regnet_x_400mf": "outputs/regnet_x_400mf/checkpoints/",
    "convnext_tiny": "outputs/convnext_tiny/checkpoints/",
    "shufflenet_v2_x0_5": "outputs/shufflenet_v2_x0_5/checkpoints/",
}

def main():
    test_dir = "../phd_work/data/tamil/test/"
    convnext = build_model("convnext_tiny", num_classes=156, pretrained=True)
    meta = load_checkpoint_into_model(convnext, "outputs/convnext_tiny/checkpoints/best.pt")
    mnasnet = build_model("mnasnet0_5", num_classes=156, pretrained=True)
    meta = load_checkpoint_into_model(mnasnet, "outputs/mnasnet0_5/checkpoints/best.pt")
    
    loader, _, _, ds_mapping = make_test_loader(test_dir, batch_size=64, num_workers=8, invert=False,
                                                       add_noise=True,noise_std=30)
    cka = CKA(model1=convnext, model2=mnasnet,
          model1_name="ConvNext",   # good idea to provide names to avoid confusion
          model2_name="mnasnet",            
          device='cuda')
    cka.compare(loader) # secondary dataloader is optional

    results = cka.export() 
    print(results)
    cka.plot_results(save_path="cka_convnext_mnasnet_noisy.png")






if __name__ == "__main__":
    main()
