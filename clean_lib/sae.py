import os
import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
from einops import rearrange
from overcomplete import TopKSAE
from clean_lib.data import Load_PACS, pacs_domains
from clean_lib.utils import extract_features
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

os.environ["WANDB_DISABLED"] = "true"



class Normalizer(nn.Module): 
    
    def __init__(self, model, dataset, domains=None):
        super().__init__() 
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.dataset = dataset
        if self.dataset == "PACS":
            dl, _ = Load_PACS(domains=domains, batch_size=1024)
            x, _ = next(iter(dl))
            
            model.to(device)
            activations = extract_features(model, x.to(self.device))

        flat = activations.flatten()
        
        self.register_buffer('mean', flat.mean())
        self.register_buffer('std', flat.std())
                
    def forward(self, activations): 
        activations = (activations - self.mean)
        activations = activations / (self.std + 1e-12)
        return activations

    def denormalize(self, normalized_activations):
        return (normalized_activations * (self.std + 1e-12)) + self.mean




class SparseAEs():
    def __init__(self, feature_dim, sae_dim, topk, nb_concepts, rearrange_string, checkpointManager, train_envs, w):
        self.topk = topk
        self.feature_dim = feature_dim
        self.sae_dim = sae_dim
        self.nb_concepts = nb_concepts
        self.rearrange_string = rearrange_string
        self.train_envs = train_envs
        self.checkpointManager = checkpointManager
        self.backbones = checkpointManager.get_models()
        self.SAEs = None
        self.w = w

    def get_sae(self, ckpt):
        return self.SAEs[ckpt]

    
    def get_backbone(self, ckpt):
        return self.backbones[ckpt]


    def load_checkpoint(self, checkpoint_path):
        self.SAEs = {}
        for key in self.backbones.keys():
            checkpoints = torch.load(checkpoint_path, weights_only=False)
            self.SAEs[key] = checkpoints[key]
            self.SAEs[key].train()
        

    def save_checkpoint(self, save_path, flag):
        run_name = f"{flag}_{self.checkpointManager.algorithm}_{self.checkpointManager.architecture}_T{''.join([str(e) for e in self.checkpointManager.testenvs])}"        
        final_save_path = os.path.join(save_path, f"USAE_{run_name})")
        torch.save(self.SAEs, final_save_path)
        print(f"SAEs saved successfully to {final_save_path}")


    def configure_training(self, learning_rate=3e-4):
        self.optimizers = {}
        self.schedulers = {}

        if self.SAEs is None:
            self.SAEs = {}
            for key in self.backbones.keys():
                self.SAEs[key] = TopKSAE(self.feature_dim, nb_concepts=self.nb_concepts, top_k=self.topk, device="cuda")
                self.SAEs[key].train()
                self.SAEs[key].normalizer = Normalizer(self.backbones[key], "PACS")


        for key in self.backbones.keys():    

            self.optimizers[key] = torch.optim.Adam(self.SAEs[key].parameters(), lr=learning_rate)
            warmup_scheduler = LinearLR(self.optimizers[key], start_factor=1e-6 / learning_rate, end_factor=1.0, total_iters=10)
            cosine_scheduler = CosineAnnealingLR(self.optimizers[key], T_max=50, eta_min=1e-6)
            self.schedulers[key] = SequentialLR(self.optimizers[key], schedulers=[warmup_scheduler, cosine_scheduler], milestones=[25])
        

            self.backbones[key].to(device)
            self.SAEs[key].to(device)

        self.criterion = nn.L1Loss(reduction="mean")  


    def train(self, flag="USAE", epochs=250, batch_size=64, full_data_gpu=True, save_dir="./SAEs", dataset="PACS"):
        
        run_name = f"{flag}_{self.checkpointManager.algorithm}_{self.checkpointManager.architecture}_T{''.join([str(e) for e in self.checkpointManager.testenvs])}"
        
        # wandb.init(
        #     project="SAE_training",
        #     name=run_name,
        #     config={
        #         "flag": flag,
        #         "epochs": epochs,
        #         "batch_size": batch_size,
        #         "algorithm": self.checkpointManager.algorithm,
        #         "architecture": self.checkpointManager.architecture,
        #         "test_envs": self.checkpointManager.testenvs
        #     },
        #     mode="offline"
        # )
        # wandb.watch(list(self.SAEs.values()), log="all")

        if dataset == "PACS":
            train_dl, test_dl = Load_PACS(domains=[pacs_domains[e] for e in self.train_envs], batch_size=batch_size, drop_last=True)


        rotator = 0
        pbar = tqdm(range(epochs), desc=f"Training: {run_name}")
        for epoch in pbar:
            epoch_loss = 0.0
            for i, (images, _) in enumerate(train_dl):
                total_loss = 0.0
                names = list(self.optimizers.keys()) 
                
                images = images.to(device)

                for k in names:
                    self.optimizers[k].zero_grad()
                    
                # Current SAE
                current = names[rotator]

                # Current SAE SAEs
                backbone = self.backbones[current]
                sae = self.SAEs[current]
                sae.train()
                
                # Encoder Forward Pass
                x = extract_features(backbone, images) # Forward Pass
                x = sae.normalizer(x) # Normalize
                x = rearrange(x, self.rearrange_string) # Rearrange
                _, z = sae.encode(x)

                # Decoder across all models & accumulate loss
                for n, m in self.SAEs.items():
                    if n == current:
                        x_hat = m.decode(z)
                    else:
                        x_hat = m.decode(z.detach())

                    loss = self.criterion(x_hat, x)
                    total_loss += loss

                total_loss.backward()
                
                self.optimizers[current].step()
                if self.schedulers:
                    self.schedulers[current].step()

                # Rotator Update
                rotator += 1
                rotator = rotator % len(names)

                # FIX: Accumulate batch loss for logging!
                epoch_loss += total_loss.item()
                
                # --- W&B STEP LOG ---
                # wandb.log({
                #     "batch_loss": total_loss.item(),
                #     "epoch": epoch
                # })

            # Update tqdm with loss info
            avg_epoch_loss = epoch_loss / len(train_dl)
            pbar.set_postfix(loss=f"{avg_epoch_loss:.6f}")

            # --- W&B EPOCH LOG ---
            # wandb.log({
            #     "epoch_loss": avg_epoch_loss,
            #     "epoch": epoch
            # })

        self.save_checkpoint(save_path=save_dir, flag=flag)
        wandb.finish()