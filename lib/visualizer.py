import os
import timm
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import copy

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from domainbed import algorithms
from torchsummary import summary
from overcomplete.models import DinoV2, ViT, ResNet
from overcomplete.sae import TopKSAE, train_sae
from overcomplete.visualization import (overlay_top_heatmaps, evidence_top_images, zoom_top_images, contour_top_image)
from einops import rearrange
from lib.mlp import train_mlp, test_mlp, prune_weights, check_sparsity
from lib.utils import extract_features
from lib.data_handlers import Load_ImageNet100, Load_PACS
from lib.universal_trainer import train_usae
from lib.activation_generator import Load_activation_dataloader
#from lib.mlp import train_mlp_PACS
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import heapq
import os
import random
import itertools
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from PIL import Image
from torchvision import transforms
from einops import rearrange
from overcomplete.visualization.plot_utils import (interpolate_cv2, get_image_dimensions, show)
from overcomplete.visualization.cmaps import VIRIDIS_ALPHA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

dirs = {
    "art_painting"  : r"C:\Users\sproj_ha\Desktop\SGen_Vision_Interp\Vision_Interp\domainbed\data\PACS\art_painting",
    "sketch"        : r"C:\Users\sproj_ha\Desktop\SGen_Vision_Interp\Vision_Interp\domainbed\data\PACS\sketch",
    "photo"         : r"C:\Users\sproj_ha\Desktop\SGen_Vision_Interp\Vision_Interp\domainbed\data\PACS\photo",
    "cartoon"       : r"C:\Users\sproj_ha\Desktop\SGen_Vision_Interp\Vision_Interp\domainbed\data\PACS\cartoon",
}





def visualize_class_on_concept(concept, class_idx, model, sae, domain_roots=dirs, save_dir=None, n_images=None):
    
    domain_top_images = {}  # domain -> list of (heatmap_sum, img_tensor, heatmap)

    for domain, dir in domain_roots.items():
        classes = os.listdir(dir)
        class_dir = os.path.join(dir, classes[class_idx])

        images = [Image.open(os.path.join(class_dir, path)) for path in os.listdir(class_dir)]

        if n_images is not None and n_images < len(images):
            images = random.sample(images, n_images)

        results = []  # (heatmap_sum, img_tensor, heatmap)

        for i, img in enumerate(images):
            img = img.convert("RGB")
            if hasattr(model, "preprocess"):
                img_tensor = model.preprocess(img).unsqueeze(dim=0)
            else:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                img_tensor = transform(img).unsqueeze(dim=0)

            x = extract_features(model, img_tensor)
            x = rearrange(x, 'n c w h -> (n w h) c')
            
            
            _, z = sae.encode(x)
            z = rearrange(z, '(n w h) d -> n w h d', w=7, h=7)
            
       
            width, height = img_tensor.shape[-1], img_tensor.shape[-2]
            heatmap = interpolate_cv2(z[:, :, :, concept], (width, height))

            heatmap_sum = heatmap.sum()
            if heatmap_sum > 0:
                results.append((heatmap_sum, img_tensor, heatmap))

        # Sort by activation and keep top 8
        results.sort(key=lambda x: x[0], reverse=True)
        domain_top_images[domain] = results[:8]

    # Build grid: rows = domains, cols = top-8 images
    domains = list(domain_top_images.keys())
    n_domains = len(domains)
    n_cols = 8

    fig, axes = plt.subplots(n_domains, n_cols, figsize=(n_cols * 2, n_domains * 2))

    # Ensure axes is always 2D
    if n_domains == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, domain in enumerate(domains):
        top = domain_top_images[domain]
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(top):
                _, img_tensor, heatmap = top[col]
                # Convert image tensor to HWC numpy for display
                img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                ax.imshow(img_np)
                ax.imshow(heatmap.squeeze(), cmap=VIRIDIS_ALPHA, alpha=0.6)
            if col == 0:
                ax.set_title(domain, fontsize=8, loc='left', pad=2)

    plt.suptitle(f"Class {class_idx} — Concept {concept} | Top 8 Activations per Domain", fontsize=11, y=1.01)
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"Class{class_idx}_Concept{concept}_Grid.png"), bbox_inches="tight")
        plt.close()
    else:
        plt.show()