import torch
from lib.data_handlers import Load_PACS
from tqdm import tqdm
from einops import rearrange

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
domains = ["photo", "art_painting", "cartoon", "sketch"]



def calculate_concept_invariance(backbone, sae, save_dir):

    num_classes = 7
    probabilities = torch.zeros((num_classes, 4, 7680), device=device)
    strength = torch.zeros((num_classes, 4, 7680), device=device)

    for d, domain in enumerate(domains):
        z_d = torch.zeros((num_classes, 4, 7680), device=device)
        z_present = torch.zeros((num_classes, 4, 7680), device=device)
        num_patches = torch.zeros((num_classes,), device=device)

        loader = Load_PACS(domains=d)

        for i, batch in enumerate(tqdm(loader)):
            with torch.no_grad():
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                _, heatmaps = sae.encode(x)
                heatmaps = rearrange(heatmaps, '(n t) d -> n t d', t=256)  # (n, t, d)

                # --- One-hot mask for all classes at once ---
                y_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes).bool()  # (n, 7)

                # --- Expand to match heatmaps ---
                # heatmaps: (n, t, d)
                # y_onehot: (n, 7)
                # → broadcast to (7, n, t, d)
                heatmaps_expanded = heatmaps.unsqueeze(0).expand(num_classes, -1, -1, -1)
                class_mask = y_onehot.T[:, :, None, None]  # (7, n, 1, 1)

                # Apply mask
                heatmaps_filtered = heatmaps_expanded * class_mask  # (7, n, t, d), zeros out non-class samples

                # Count positive activations
                present = heatmaps_filtered > 0  # (7, n, t, d)
                present_per_image = present.any(dim=2)  # (7, n, d)
                count_per_d = present_per_image.sum(dim=1)  # (7, d)

                # Accumulate per domain
                z_present[:, d] += count_per_d
                z_d[:, d] += heatmaps_filtered.sum(dim=(1, 2))  # sum over n and t
                num_patches += (class_mask.sum(dim=(1, 2, 3)) * 256)  # total pixels per class

        # Normalize per class
        for c in range(num_classes):
            if num_patches[c] > 0:
                z_d[c, d] /= num_patches[c]

        probabilities += z_d
        strength += z_present


    # probabilities = {}
    # strength = {}


    # for cls in range(7):
    #     probabilities[cls] = {}     
    #     strength[cls] = {}
        
    #     z_d = torch.zeros((4, 7680)).to(device)
    #     z_present = torch.zeros((4, 7680)).to(device)

    #     for d, domain in enumerate(domains):

    #         num_patches = 0.0
    #         loader = Load_PACS(domains=d)
            
    #         for i, batch in enumerate(tqdm(loader)):
                
    #             with torch.no_grad():
    #                 x, y = batch
    #                 x = x.to(device)

    #                 _, heatmaps = sae.encode(x)
                    
    #                 mask = (y == cls).squeeze().to(device)  # (batch_size,)
    #                 heatmaps = rearrange(heatmaps, '(n t) d -> n t d', t=256)  # (n, t, d)

    #                 # Apply mask at sample level
    #                 heatmaps_filtered = heatmaps[mask]  # (n_cls, t, d)
                    
                    
    #                 present = heatmaps_filtered > 0   # (n, t, d) boolean
    #                 present_per_image = present.any(dim=1)   # (n, d) boolean
    #                 count_per_d = present_per_image.sum(dim=0)   # (d,)

    #                 # Accumulate
    #                 z_present += count_per_d
    #                 z_d[d] += heatmaps_filtered.sum(dim=0).sum(dim=0)
    #                 num_patches += heatmaps_filtered.shape[0] * heatmaps_filtered.shape[1]

    #         score = z_d[d] / num_patches
            
    #     print(f"DinoV2-{model} scores for class {cls} on all concept for all domains ")
    #     print("PD Shape: ", z_d.shape)
    #     probabilities[cls] = z_d
    #     strength[cls] = z_present





#     eps = 1e-12
# model = "pre"
# import math
# mask = probabilities[model][0] != 0.25
# processed = probabilities[model][0] * mask
# strength_0 = strength["pre"][0]

# entropies = []

# for i in range(7680):
#     if processed[:, i].sum() == 0:
#         continue
    
#     score = processed[:, i] / processed[:, i].sum()
#     entropy = -1 / torch.log(torch.tensor(4)) * (score * torch.log(score + eps)).sum()

#     #scaled_entropy = processed[:, i].sum() * entropy
#     scaled_entropy = entropy

#     entropies.append((i, scaled_entropy.item()))



# # Sort by entropy value
# entropies.sort(key=lambda x: x[1], reverse=True)

# print("PRE ==================================")

# # Now entropies is sorted
# for idx, val in entropies:
#     print(f"Concept {idx}: Entropy = {val}", f"|| Strength: {strength_0.sum(dim=0)[idx]}"
# )
