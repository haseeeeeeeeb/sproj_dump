import os
import torch
from pathlib import Path
from domainbed.algorithms import DANN, CORAL, Mixup, MMD, IRM, ERM, SagNet
algo_classes = {"DANN": DANN, "CORAL": CORAL, "Mixup": Mixup, "MMD": MMD, "IRM": IRM, "ERM": ERM, "SagNet": SagNet}
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
from domainbed.networks import Identity

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# TEST TO TRAIN ENVS
envs = {
    "T0": [1, 2, 3],
    "T1": [0, 2, 3],
    "T2": [0, 1, 3],
    "T3": [0, 1, 2],
    "T23": [0, 1], 
    "T13": [0, 2], 
    "T12": [0, 3], 
    "T03": [1, 2], 
    "T02": [1, 3], 
    "T01": [2, 3]
} 



class BackboneManager():
   
    def __init__(self, directory: str):
        
        super().__init__()
        self.directory = directory
        self.name = Path(self.directory).name
        self.algorithm, self.architecture, self.split = self.name.split("_")
        self.trainenvs = envs[self.split]


    def load_checkpoints(self, checkpoints: list[int]):
        models = {}
        for checkpoint in checkpoints:
            models[checkpoint] = self.load_backbone(ckpt_number=checkpoint)

        return models


    def load_backbone(self, ckpt_number: int):    
        
        
        checkpoint = torch.load(os.path.join(self.directory, f"model_step{ckpt_number}.pkl"))

        ModelClass = algo_classes[self.algorithm]

        backbone = ModelClass(
            input_shape=checkpoint["model_input_shape"],
            hparams=checkpoint["model_hparams"],
            num_domains=checkpoint["model_num_domains"],
            num_classes=checkpoint["model_num_classes"]
        )

        if backbone.featurizer.__class__.__name__ == "ResNet":
            del backbone.featurizer.network.global_pool
            backbone.featurizer.network.global_pool = Identity()

        backbone.load_state_dict(checkpoint["model_dict"])

        return backbone


    def get_top_k_checkpoints(self, envs: list[int] = None, k: int = 5):
        

        if envs is None:
            envs = self.trainenvs
        
        target_cols = [f'env{env}_out_acc' for env in envs]
        results = []  # (avg_acc, step, per_env_accs)
        
        with open(os.path.join(self.directory, "out.txt"), 'r') as f:
            lines = f.readlines()
            
        header_indices = {}
        is_parsing_table = False
        
        for line in lines:
            parts = line.split()
            if not parts:
                continue
                
            # Identify the table header
            if 'env0_in_acc' in parts and 'step' in parts:
                is_parsing_table = True
                header_indices = {col: i for i, col in enumerate(parts)}
                continue
                
            # Parse table rows
            if is_parsing_table:
                try:
                    row_vals = [float(x) for x in parts]
                    
                    if len(row_vals) < len(header_indices):
                        continue
                        
                    step = int(row_vals[header_indices['step']])
                    accs = [row_vals[header_indices[col]] for col in target_cols]
                    avg_acc = sum(accs) / len(accs)

                    per_env_accs = {env: row_vals[header_indices[f'env{env}_out_acc']] for env in envs}
                    results.append((avg_acc, step, per_env_accs))
                    
                except (ValueError, KeyError):
                    continue
                    
                
        results.sort(key=lambda x: (-x[0], x[1]))
        top_results = results[:k]
        top_k_steps = [step for _, step, _ in top_results]
        accs_dict = {step: {"avg": avg, **per_env_accs} for avg, step, per_env_accs in top_results}

        return top_k_steps, accs_dict
