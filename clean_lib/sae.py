import torch
import torch.nn as nn
from clean_lib.data import Load_PACS
from clean_lib.utils import extract_features

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"



class Normalizer(nn.Module): 
    def __init__(self, model, dataset, domains=None):
        super().__init__() 

        self.dataset = dataset
        if self.dataset == "PACS":
            dl, _ = Load_PACS(domains=domains, batch_size=1024)
            x, _ = next(iter(dl))
            
            model.to(device)
            activations = extract_features(model, x.to(device))

        flat = activations.flatten()
        
        self.register_buffer('mean', flat.mean())
        self.register_buffer('std', flat.std())
                
    def forward(self, activations): 
        activations = (activations - self.mean)
        activations = activations / (self.std + 1e-12)
        return activations

    def denormalize(self, normalized_activations):
        return (normalized_activations * (self.std + 1e-12)) + self.mean







#     backbones,
#     r"C:\Users\sproj_ha\Desktop\SGen_Vision_Interp\Vision_Interp\SAEs\normalization_testing", 
#     trainenvs=[0, 1, 2],
#     nb_concepts=2048*8,
#     top_k=16,
#     learning_rate=3e-4,
#     epochs=250,
#     checkpoint_path=r"C:\Users\sproj_ha\Desktop\SGen_Vision_Interp\Vision_Interp\SAEs\normalization_testing\USAE_ERM_Multi_test_3300_2100.pt",
#     rearrange_string='n c w h -> (n w h) c',
#     batch_size=64,
#     full_data_gpu=True,
#     enable_logging=True,
#     logging_dir=r"C:\Users\sproj_ha\Desktop\SGen_Vision_Interp\Vision_Interp\logs\normal_USAE_loaded",
#     name_flag="3300_2100"
# )

# Initialize or load SAE
    SAEs = {}
    if checkpoint_path is not None:
        for key in backbones.keys():
            checkpoints = torch.load(checkpoint_path, weights_only=False)
            SAEs[key] = checkpoints[key]
            SAEs[key].train()
            if not hasattr(SAEs[key], "normalizer"):
                SAEs[key].normalizer = Normalizer(backbones[key], "PACS")

            # Assuming device is defined globally, otherwise add to args
            SAEs[key].to(device)
            backbones[key].to(device)
    
    else:
        fdim = nb_concepts // 8
        for key in backbones.keys():
            SAEs[key] = 
            SAEs[key].train()

            if not hasattr(SAEs[key], "normalizer"):
                SAEs[key].normalizer = Normalizer(backbones[key], "PACS")
            backbones[key].to(device)


    optimizers = {}
    schedulers = {}

    for key in backbones.keys():
        optimizers[key] = torch.optim.Adam(SAEs[key].parameters(), lr=learning_rate)
        warmup_scheduler = LinearLR(optimizers[key], start_factor=1e-6 / learning_rate, end_factor=1.0, total_iters=10)
        cosine_scheduler = CosineAnnealingLR(optimizers[key], T_max=epochs-25, eta_min=1e-6)
        schedulers[key] = SequentialLR(optimizers[key], schedulers=[warmup_scheduler, cosine_scheduler], milestones=[25])
    
    criterion = nn.L1Loss(reduction="mean")  
    


class SAEs:
    def __init__(self, feature_dim, sae_dim, topk, optimizer, scheduler):
        self.topk = topk
        self.feature_dim = feature_dim
        self.sae_dim = sae_dim
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.nb_concepts = nb_concepts
    

        self.model = TopKSAE(sae_dim, nb_concepts=self.nb_concepts, top_k=self.top_k, device="cuda")
        self.optimizer


domains = {
    0: "art_painting", 
    1: "cartoon",
    2: "photo",
    3: "sketch"
}

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


















def train_pacs_USAEs(backbones, 
                     save_path, 
                     trainenvs, 
                     rearrange_string='n t d -> (n t) d', 
                     nb_concepts=None, 
                     top_k=None, 
                     learning_rate=None, 
                     epochs=None, 
                     checkpoint_path=None, 
                     batch_size=64, 
                     full_data_gpu=False,
                     enable_logging=False, 
                     logging_dir=None, 
                     name_flag=None,
                     # Added missing arguments used in string formatting
                     name="USAE",
                     algo_name="ERM",
                     backbone_name="Multi",
                     test_envs="test"):
    """
    Train a Sparse Autoencoder on PACS data.
    """
        
    # Choose loader based on full_data_gpu flag
    if full_data_gpu:
        domain_train_loader = get_pacs_gpuloader(domains=[domains[e] for e in trainenvs], batch_size=batch_size, drop_last=True)
    else:
        domain_train_loader = get_pacs_standard_loader(domains=[domains[e] for e in trainenvs], batch_size=batch_size, drop_last=True)

    print(f"train envs: {trainenvs}")
    
    # Validate SAE configuration
    if nb_concepts is None or top_k is None:
        raise ValueError(
            "SAE configuration (nb_concepts and top_k) must be explicitly provided via command-line arguments. "
            "No defaults are assumed. Use --nb-concepts and --top-k to specify values."
        )
    
    # Set defaults for learning rate and epochs if not provided
    if learning_rate is None:
        learning_rate = 3e-4
    if epochs is None:
        epochs = 250
    
    print(f"SAE Configuration: nb_concepts={nb_concepts}, top_k={top_k}")
    print(f"Training Configuration: lr={learning_rate}, epochs={epochs}")
    
    # Validate logging configuration and load history if resuming
    loss_history = []
    starting_epoch = 0
    if enable_logging:
        if logging_dir is None:
            raise ValueError("logging_dir must be provided when enable_logging=True")
        os.makedirs(logging_dir, exist_ok=True)
        loss_log_file = os.path.join(logging_dir, f"loss_{algo_name}_{backbone_name}_{test_envs}" + (f"_{name_flag}" if name_flag else "") + ".txt")
        loss_curve_file = os.path.join(logging_dir, f"loss_curve_{algo_name}_{backbone_name}_{test_envs}" + (f"_{name_flag}" if name_flag else "") + ".png")
        
        # Load previous loss history if resuming from checkpoint
        if checkpoint_path is not None and os.path.exists(loss_log_file):
            print(f"Loading existing loss history from checkpoint resume...")
            with open(loss_log_file, 'r') as f:
                loss_lines = f.readlines()
            for line in loss_lines:
                try:
                    loss_value = float(line.split(': ')[-1].strip())
                    loss_history.append(loss_value)
                except:
                    pass
            starting_epoch = len(loss_history)
            print(f"Resumed from epoch {starting_epoch}. Will train for {epochs} more epochs.")
    else:
        loss_log_file = None
        loss_curve_file = None

    # FIX: Initialize rotator
    rotator = 0

    pbar = tqdm(range(epochs), desc=f"Training SAE: {name}")
    for epoch in pbar:
        actual_epoch_num = starting_epoch + epoch + 1  # For display (1-indexed)
        total_epochs = starting_epoch + epochs
        epoch_loss = 0.0
        
        for i, (images, _) in enumerate(domain_train_loader):
            total_loss = 0.0
            names = list(optimizers.keys()) # FIX: Need list() to index dict keys
            
            images = images.to(device)

            for k in names:
                optimizers[k].zero_grad()
                
            # Current SAE
            current = names[rotator]

            # Current SAE model
            backbone = backbones[current]
            sae = SAEs[current]
            sae.train()
            
            # Encoder Forward Pass
            x = extract_features(backbone, images) # Forward Pass
            x = sae.normalizer(x) # Normalize
            x = rearrange(x, rearrange_string) # Rearrange
            _, z = sae.encode(x)

            # Decoder across all models & accumulate loss
            for n, m in SAEs.items():
                if n == current:
                    x_hat = m.decode(z)
                else:
                    x_hat = m.decode(z.detach())

                loss = criterion(x_hat, x)
                total_loss += loss

            total_loss.backward()
            
            optimizers[current].step()
            if schedulers:
                schedulers[current].step()

            # Rotator Update
            rotator += 1
            rotator = rotator % len(names)

            # FIX: Accumulate batch loss for logging!
            epoch_loss += total_loss.item()
            
        # Average loss over batches
        avg_epoch_loss = epoch_loss / len(domain_train_loader)
        loss_history.append(avg_epoch_loss)
        
        # Log and update plot if logging is enabled
        if enable_logging:
            # Write to log file (appends to existing)
            with open(loss_log_file, 'a') as f:
                f.write(f"Epoch {actual_epoch_num}/{total_epochs}: {avg_epoch_loss:.6f}\n")
            
            # Update loss curve every 10 epochs or at the end
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                plt.figure(figsize=(10, 6))
                plt.plot(loss_history, linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Training Loss - {name}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(loss_curve_file, dpi=100)
                plt.close() # Important to close so we don't leak memory
        
        # Update tqdm with loss info
        pbar.set_postfix(loss=f"{avg_epoch_loss:.6f}")

    # FIX: Save the entire dictionary, not just the last SAE, to match loading logic
    final_save_path = os.path.join(save_path, f"USAE_{algo_name}_{backbone_name}_{test_envs}" + (f"_{name_flag}" if name_flag else "") + ".pt")
    torch.save(SAEs, final_save_path)
    print(f"SAEs saved successfully to {final_save_path}")