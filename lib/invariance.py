import math
import torch
from tqdm import tqdm
from einops import rearrange
from lib.data_handlers import Load_PACS
import json
import os
import json
from collections import defaultdict, Counter
from lib.utils import extract_features

domains = ["photo", "art_painting", "cartoon", "sketch"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Normalizer():
#     def __init__(self):
#         self.mean = None
#         self.std = None
#     def populate(self, activations):
#         flat = activations.flatten()
#         self.mean = flat.mean()
#         self.std = flat.std()
#     def run(self, activations):
#         activations = (activations - self.mean)
#         activations = activations / (self.std + 1e-12)
#         return activations

def calculate_mean_activations(backbone, sae, rearrange_string, w=14, domains=domains, nb_concepts=7680):
    probabilities = {}
    strength = {}
    for cls in range(7):
        probabilities[cls] = {}
        strength[cls] = {}
        z_d = torch.zeros((len(domains), nb_concepts)).to(device)
        z_present = torch.zeros((len(domains), nb_concepts)).to(device)
        for d, domain in enumerate(domains):
            num_patches = 0.0
            loader, _ = Load_PACS(domains=[domain])
            # normal = Normalizer()
            for i, batch in enumerate(tqdm(loader)):
                with torch.no_grad():
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    x = extract_features(backbone, x)
                    # if i == 0:
                    #     normal.populate(x)
                    # x = normal.run(x)
                    
                    # x = x.permute(0, 2, 3, 1)
                    x = rearrange(x, rearrange_string)

                    _, heatmaps = sae.encode(x)

                    
                    mask = (y == cls).squeeze().to(device)  # (batch_size,)
                    heatmaps = rearrange(heatmaps, '(n w h) d -> n w h d', w=w, h=w)  # (n, t, d)

                    
                    heatmaps_filtered = heatmaps[mask]  # (n_cls, t, d)
                    present = heatmaps_filtered > 0   # (n, t, d) boolean
                    present_per_image = present.any(dim=(1, 2))   # (n, d) boolean
                    count_per_d = present_per_image.sum(dim=0)   # (d,)
                    z_present += count_per_d
                    z_d[d] += heatmaps_filtered.sum(dim=0).sum(dim=0).sum(dim=0)
                    num_patches += heatmaps_filtered.shape[0] * heatmaps_filtered.shape[1] * heatmaps_filtered.shape[2]
        probabilities[cls] = z_d
        strength[cls] = z_present
    return probabilities, strength

def save_json(data, filepath):
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved logs to {filepath}")
    except TypeError as e:
        print(f"Error saving JSON: {e}. Check for non-serializable types (like tensors).")
    except Exception as e:
        print(f"An error occurred: {e}")

def calculate_invariance(probabilities, strength, ent_thresh=0.7, act_thresh=0, domains=domains, nb_concepts=7680):
    clss = 7
    logs = {
        "model_invariance" : 0,
        "final_invariance_per_class": {},
        "thresholded_concept_entropies": {}
    }
    for cls in range(clss):
        # mask = probabilities[cls] != 0.25
        processed = probabilities[cls] # * mask
        str_val = strength[cls]  # Renamed from 'str'

        sum_entropy = 0.0
        class_concept_logs = []

        for i in range(nb_concepts):
            if processed[:, i].sum() == 0:
                continue

            score = processed[:, i] / processed[:, i].sum()

            entropy = -1 / torch.log(torch.tensor(len(domains))) * (score * torch.log(score + 1e-12)).sum()

            if entropy > ent_thresh and processed[:, i].sum() > act_thresh:
                sum_entropy += entropy

                # --- LOG INDIVIDUAL ENTROPY ---
                class_concept_logs.append({
                    "concept_index": i,
                    "entropy": entropy.item(),
                    "scores": [s.item() for s in score],
                    "mean_acts": [val.item() for val in processed[:, i]]
                })

        invariance_val = (sum_entropy)
        if isinstance(invariance_val, torch.Tensor):
            invariance_float = invariance_val.item()
        else:
            invariance_float = invariance_val # It might already be a float (if sum_entropy was 0.0)

        # --- LOG FINAL INVARIANCE ---
        logs["final_invariance_per_class"][cls] = invariance_float
        logs["model_invariance"] += invariance_float
        # --- LOG ALL CONCEPT ENTROPIES FOR THIS CLASS ---
        logs["thresholded_concept_entropies"][cls] = class_concept_logs

        print(f"Total Thresholded Entropy (INVARIANCE) for class {cls}: {invariance_float}")


    logs["model_invariance"] /= 7
    return logs

