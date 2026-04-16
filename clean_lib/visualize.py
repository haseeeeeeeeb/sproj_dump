import os
import torch
import random
import numpy as np
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt
from torchvision import transforms
from overcomplete.visualization.cmaps import VIRIDIS_ALPHA
from overcomplete.visualization.plot_utils import (interpolate_cv2, show)
from clean_lib.utils import extract_features


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()



dirs = {
    "art_painting"  : r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS\art_painting",
    "sketch"        : r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS\sketch",
    "photo"         : r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS\photo",
    "cartoon"       : r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS\cartoon",
}

def visualize_concept_on_class(concept, class_idx, sae_manager, ckpt, domain_roots=dirs, save_dir=None, n_images=None):
    

    sae = sae_manager.get_sae(ckpt)
    backbone = sae_manager.get_backbone(ckpt)
    backbone.to(device)
    rearrange_string = sae_manager.rearrange_string
    w = sae_manager.w

    domain_top_images = {}  # domain -> list of (heatmap_sum, img_tensor, heatmap)
    
    # 1. Build the global sorted class list EXACTLY like the dataloader
    all_classes = set()
    for d_path in domain_roots.values():
        for entry in os.scandir(d_path):
            if entry.is_dir():
                all_classes.add(entry.name)
    sorted_classes = sorted(list(all_classes))
    
    # 2. Map the index to the actual string name
    target_class_name = sorted_classes[class_idx]

    for domain, dir_path in domain_roots.items():
        # 3. Safely build the path using the string name
        class_dir = os.path.join(dir_path, target_class_name)
        
        if not os.path.exists(class_dir):
            continue
            
        images = [Image.open(os.path.join(class_dir, path)) for path in os.listdir(class_dir)]

        if n_images is not None and n_images < len(images):
            images = random.sample(images, n_images)

        results = []  # (heatmap_sum, img_tensor, heatmap)

        for i, img in enumerate(images):
            img = img.convert("RGB")
            transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            ])

            img_tensor = transform(img).unsqueeze(dim=0).to(device) # Don't forget to send to device!

            x = extract_features(backbone, img_tensor)
            x = sae.normalizer(x)
            
            x = rearrange(x, rearrange_string)
            
            _, z = sae.encode(x)
            
            # FIX 2: Use dynamic spatial dimensions
            z = rearrange(z, '(n w h) d -> n w h d', w=w, h=w)
            
            width, height = img_tensor.shape[-1], img_tensor.shape[-2]
            
            # FIX 3: Isolate the specific 2D image, detach, move to CPU, and convert to numpy
            heatmap_2d = z[0, :, :, concept].detach().cpu().numpy()
            
            heatmap = interpolate_cv2(heatmap_2d, (width, height))
            heatmap_sum = heatmap.sum()

            if heatmap_sum > 0:
                results.append((heatmap_sum, img_tensor.cpu(), heatmap)) # Move img_tensor back to CPU for storage/plotting

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
                show(img_tensor, ax=ax)
                show(heatmap, ax=ax, cmap=VIRIDIS_ALPHA, alpha=1.0)
            if col == 0:
                ax.set_title(domain, fontsize=8, loc='left', pad=2)

    plt.suptitle(f"Class {target_class_name} — Concept {concept} | Top 8 Activations", fontsize=11, y=1.01)
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"Class{class_idx}_Concept{concept}_Grid.png"), bbox_inches="tight")
        plt.close()
    else:
        plt.show()