import os
from tqdm import tqdm
import torch
import torch.nn as nn
from einops import rearrange
from overcomplete.sae import TopKSAE
from lib.gpu_pacs import get_pacs_gpuloader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import time
from lib.utils import extract_features

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


domains = {
    0: "art_painting", 
    1: "cartoon",
    2: "photo",
    3: "sketch"
}

# TEST TO TRAIN ENVS
envs = {
    "T0": [0],
    "T1": [1],
    "T2": [2],
    "T3": [3],
    "T23": [0, 1], 
    "T13": [0, 2], 
    "T12": [0, 3], 
    "T03": [1, 2], 
    "T02": [1, 3], 
    "T01": [2, 3]
} 


class Normalizer():
    def __init__(self):
        self.mean = None
        self.std = None

    def populate(self, activations):
        flat = activations.flatten() 
        self.mean = flat.mean() 
        self.std = flat.std()

    def run(self, activations):
        activations = (activations - self.mean)
        activations = activations / (self.std + 1e-12)
        return activations


def train_pacs_SAEs(backbone, save_path, name, rearrange_string='n t d -> (n t) d'):
    
    algo_name, bakcbone_name, test_envs = name.split("_")

    trainenvs = envs[test_envs]
     
    domain_train_loader = get_pacs_gpuloader(domains=[domains[e] for e in trainenvs], batch_size=256, drop_last=True)

    print(f"test envs: {test_envs}")
    print(f"train envs: {trainenvs}")
    
    sae = TopKSAE(768, nb_concepts=768*10, top_k=64, device="cuda",)
    sae.train()


    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)


    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6 / 3e-4, end_factor=1.0, total_iters=10)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=250, eta_min=1e-6)
    scheduler = SequentialLR(optimizer,schedulers=[warmup_scheduler, cosine_scheduler], milestones=[25],)
    

    criterion = nn.L1Loss(reduction="mean")  
    epoch_loss = 0.0


    normal = Normalizer()
    sample, _ = next(iter(domain_train_loader))
    normal.populate(sample)

    backbone.to(device)

    for epoch in tqdm(range(250)):
        for i, (images, _) in enumerate(domain_train_loader):
            
            activations = extract_features(backbone, images) # Forward Pass
            activations = normal.run(activations) # Normalize
            activations = rearrange(activations, rearrange_string) # Rearrange

            optimizer.zero_grad()
            z_pre, z = sae.encode(activations)
            activations_hat = sae.decode(z)

            loss = criterion(activations_hat, activations)
            
            loss.backward()        
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()

    torch.save(sae, os.path.join(save_path, f"SAE_{algo_name}_{bakcbone_name}_{test_envs}.pt"))
