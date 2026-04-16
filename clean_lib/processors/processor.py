import json
import os
import torch
from pathlib import Path
from clean_lib.data import pacs_domains


class Processor:
    def __init__(self, sae_manager, ckpt, process_domains, file_path, dataset="PACS"):
        

        self.sae_manager = sae_manager
        self.ckpt = ckpt

        self.sae = self.sae_manager.get_sae(self.ckpt)
        self.backbone = self.sae_manager.get_backbone(self.ckpt)
        
        ## All Processing Needs to be in Eval Mode
        self.sae.eval()
        self.backbone.eval()


        self.dataset = dataset
        
        # Configure Dataset-specific parameters
        if self.dataset == "PACS":
            print("Configured for PACS dataset.")
            self.classes = 7
            self.all_domains = pacs_domains



        ## Configure FIles
        self.file_path = file_path
        self.create_template()

        self.process_domains = process_domains
        self.domains = [self.all_domains[e] for e in self.process_domains]



    def create_template(self):
        path_obj = Path(self.file_path)
        
        if path_obj.exists():
            return

        path_obj.parent.mkdir(parents=True, exist_ok=True)

        template = {
            str(cls_idx): {
                str(concept_idx): {}
                for concept_idx in range(self.sae_manager.nb_concepts)
            }
            for cls_idx in range(self.classes)
        }

        with open(self.file_path, "w") as f:
            json.dump(template, f, indent=4)



    def dump(self, scores: torch.Tensor, name: str):

        assert scores.shape[0] == self.classes, (f"Expected first dim {self.classes}, got {scores.shape[0]}")
        assert scores.shape[1] == self.sae_manager.nb_concepts, (f"Expected second dim {self.sae_manager.nb_concepts}, got {scores.shape[1]}")

        scores_np = scores.detach().cpu()

        def to_dumpable(tensor):
            if tensor.dim() == 0:
                return tensor.item()
            return [to_dumpable(tensor[i]) for i in range(tensor.shape[0])]

        with open(self.file_path, "r") as f:
            data = json.load(f)

        # Clear all existing values for this name before writing new ones
        for cls_idx in range(len(data)):
            for concept_idx in range(len(data[str(cls_idx)])):
                data[str(cls_idx)][str(concept_idx)].pop(name, None)

        for cls_idx in range(self.classes):
            for concept_idx in range(self.sae_manager.nb_concepts):
                value = to_dumpable(scores_np[cls_idx, concept_idx])
                data[str(cls_idx)][str(concept_idx)][name] = value

        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=4)








import math
import torch
from tqdm import tqdm
from einops import rearrange
from lib.data_handlers import Load_PACS
import json
import os
import json
from collections import defaultdict, Counter
from overcomplete.visualization.plot_utils import (interpolate_cv2, get_image_dimensions, show)
from overcomplete.visualization.cmaps import VIRIDIS_ALPHA



domains = ["photo", "art_painting", "cartoon", "sketch"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_mean_activations(backbone, sae, rearrange_string, w=14, domains=domains, nb_concepts=7680):
    activations = {}
    for cls in range(7):
        
        activations[cls] = {}
        z_d = torch.zeros((len(domains), nb_concepts)).to(device)
        
        for d, domain in enumerate(domains):
            loader, _ = Load_PACS(domains=[domain])
            for i, batch in enumerate(tqdm(loader)):
                with torch.no_grad():
                    img, y = batch
                    img, y = img.to(device), y.to(device)
                    
                    x = extract_features(backbone, img)
                    x = sae.normalizer(x)
                    x = rearrange(x, rearrange_string)

                    _, heatmaps = sae.encode(x)

                    mask = (y == cls).squeeze().to(device)  # (batch_size,)
                    heatmaps = rearrange(heatmaps, '(n w h) d -> n w h d', w=w, h=w)  # (n, t, d)
                    heatmaps_filtered = heatmaps[mask]  # (n_cls, t, d)
                    
                    z_d[d] += heatmaps_filtered.sum(dim=0).sum(dim=0).sum(dim=0)
                    

        activations[cls] = z_d
        
    return activations

def save_json(data, filepath):
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved logs to {filepath}")
    except TypeError as e:
        print(f"Error saving JSON: {e}. Check for non-serializable types (like tensors).")
    except Exception as e:
        print(f"An error occurred: {e}")

def calculate_invariance(activations, ent_thresh=0.7, act_thresh=0, domains=domains, nb_concepts=7680):
    clss = 7
    logs = {
        "model_invariance" : 0,
        "final_invariance_per_class": {},
        "thresholded_concept_entropies": {}
    }
    for cls in range(clss):
        # mask = probabilities[cls] != 0.25
        processed = activations[cls] # * mask
        
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

