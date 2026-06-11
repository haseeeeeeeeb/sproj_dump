"""
Sparse Autoencoder (SAE) Training Script with Full Command-Line Control

USAGE EXAMPLES:

# Batch Mode: Train all models in a directory
python SAE_Train_V2.py --base-dir ./oracle_selection_models --nb-concepts 7680 --top-k 64 --save-path ./oracle_saes

# Single Model Mode: Train a specific model from pickle path
python SAE_Train_V2.py --model-pickle-path ./oracle_selection_models/ERM_ResNet50_T01/model.pkl --model-name ERM_ResNet50_T01 --nb-concepts 7680 --top-k 64 --save-path ./oracle_saes

# Resume from Checkpoint: Continue training from a saved checkpoint
python SAE_Train_V2.py --model-pickle-path ./oracle_selection_models/ERM_ResNet50_T01/model.pkl --model-name ERM_ResNet50_T01 --nb-concepts 7680 --top-k 64 --checkpoint ./oracle_saes/SAE_ERM_ResNet50_T01.pt --epochs 100 --save-path ./oracle_saes

# Training with Logging: Log losses to file and generate live loss curve
python SAE_Train_V2.py --model-pickle-path ./oracle_selection_models/ERM_ResNet50_T01/model.pkl --model-name ERM_ResNet50_T01 --nb-concepts 7680 --top-k 64 --enable-logging --logging-dir ./logs --save-path ./oracle_saes

# Resume from Checkpoint with Logging: Continues logging from previous session
python SAE_Train_V2.py --model-pickle-path ./oracle_selection_models/ERM_ResNet50_T01/model.pkl --model-name ERM_ResNet50_T01 --nb-concepts 7680 --top-k 64 --checkpoint ./oracle_saes/SAE_ERM_ResNet50_T01.pt --epochs 100 --enable-logging --logging-dir ./logs --save-path ./oracle_saes

# With Custom Name Flag: Append identifier to saved filename (e.g., SAE_ERM_ResNet50_T01_v2.pt)
python SAE_Train_V2.py --model-pickle-path ./oracle_selection_models/ERM_ResNet50_T01/model.pkl --model-name ERM_ResNet50_T01 --nb-concepts 7680 --top-k 64 --save-path ./oracle_saes --name-flag v2

"""


import os
from tqdm import tqdm
import torch
import torch.nn as nn
from einops import rearrange
from overcomplete.sae import TopKSAE
from lib.gpu_pacs import get_pacs_gpuloader, get_pacs_standard_loader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import warnings
import argparse
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)
from domainbed.algorithms import DANN, CORAL, Mixup, MMD, IRM, ERM, SagNet
algo_classes = {"DANN": DANN, "CORAL": CORAL, "Mixup": Mixup, "MMD": MMD, "IRM": IRM, "ERM": ERM, "SagNet": SagNet}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
from lib.loaders import load_backbone




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


def extract_features(backbone, images):
    
    with torch.no_grad():

        if hasattr(backbone, 'featurizer'): # for models trained using domainbed
            if backbone.featurizer.__class__.__name__ == "DinoV2" :
                activations = backbone.featurizer.network.forward_features(images.to(device))['x_norm_patchtokens']
            elif backbone.featurizer.__class__.__name__ == "ViT":
                activations = backbone.featurizer.network.forward_features(images.to(device))[:, 1:, :]
            else:
                activations = backbone.network[0](images.to(device))

        if hasattr(backbone, 'forward_features'): # for models directly from the overcomplete library
            activations = backbone.forward_features(images.to(device))



    return activations


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


def train_pacs_SAEs(backbone, save_path, name, rearrange_string='n t d -> (n t) d', 
                    nb_concepts=None, top_k=None, learning_rate=None, epochs=None, 
                    checkpoint_path=None, batch_size=64, full_data_gpu=False,
                    enable_logging=False, logging_dir=None, name_flag=None):
    """
    Train a Sparse Autoencoder on PACS data.
    
    Args:
        backbone: The neural network backbone
        save_path: Path where to save the trained SAE
        name: Model name for naming the SAE file
        rearrange_string: String for rearranging activations
        nb_concepts: Number of concepts (required if not using defaults)
        top_k: Top-k sparsity value (required if not using defaults)
        learning_rate: Learning rate for optimizer (defaults to 3e-4)
        epochs: Number of training epochs (defaults to 250)
        checkpoint_path: Path to checkpoint to resume training from
        batch_size: Batch size for training (defaults to 64)
        full_data_gpu: If True, load all data to GPU; if False, use standard CPU-based loader (defaults to False)
        enable_logging: If True, log loss to file and create loss curve (defaults to False)
        logging_dir: Directory to save logs and loss curve (required if enable_logging=True)
        name_flag: Optional flag to append to filename (e.g., 'SAE_ERM_ViT_T3_FLAG.pt' if name_flag='FLAG')
    """
    
    # Parse name to get train environments
    algo_name, backbone_name, test_envs = name.split("_")
    trainenvs = envs[test_envs]
    
    # Choose loader based on full_data_gpu flag
    if full_data_gpu:
        domain_train_loader = get_pacs_gpuloader(domains=[domains[e] for e in trainenvs], batch_size=batch_size, drop_last=True)
    else:
        domain_train_loader = get_pacs_standard_loader(domains=[domains[e] for e in trainenvs], batch_size=batch_size, drop_last=True)

    print(f"test envs: {test_envs}")
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
    
    # Initialize or load SAE
    if checkpoint_path is not None:
        print(f"Loading SAE from checkpoint: {checkpoint_path}")
        sae = torch.load(checkpoint_path, weights_only=False)
        sae.train()
    else:
        fdim = nb_concepts // 8
        print(f"FDIM : {fdim}")
        sae = TopKSAE(fdim, nb_concepts=nb_concepts, top_k=top_k, device="cuda")
        sae.train()

    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6 / learning_rate, end_factor=1.0, total_iters=10)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs-25, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[25])
    
    criterion = nn.L1Loss(reduction="mean")  
    epoch_loss = 0.0

    # normal = Normalizer()
    # sample, _ = next(iter(domain_train_loader))
    # normal.populate(sample)

    backbone.to(device)
    
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

    for epoch in tqdm(range(epochs), desc=f"Training {name}"):
        actual_epoch_num = starting_epoch + epoch + 1  # For display (1-indexed)
        total_epochs = starting_epoch + epochs
        epoch_loss = 0.0
        for i, (images, _) in enumerate(domain_train_loader):
            
            activations = extract_features(backbone, images) # Forward Pass
            #activations = normal.run(activations) # Normalize
            activations = rearrange(activations, rearrange_string) # Rearrange

            optimizer.zero_grad()
            z_pre, z = sae.encode(activations)
            activations_hat = sae.decode(z)

            loss = criterion(activations_hat, activations)
            
            loss.backward()        
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        # Average loss over batches
        avg_epoch_loss = epoch_loss / len(domain_train_loader)
        loss_history.append(avg_epoch_loss)
        
        # Update tqdm postfix to show loss
        if hasattr(epoch, '__len__'):
            pass  # In case tqdm object behaves differently
        
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
                plt.close()
        
        # Update tqdm with loss info (always show, even without logging)
        tqdm.write(f"Epoch {actual_epoch_num}/{total_epochs} - Loss: {avg_epoch_loss:.6f}")

    torch.save(sae, os.path.join(save_path, f"SAE_{algo_name}_{backbone_name}_{test_envs}" + (f"_{name_flag}" if name_flag else "") + ".pt"))
    print(f"SAE saved to {os.path.join(save_path, f'SAE_{algo_name}_{backbone_name}_{test_envs}' + (f'_{name_flag}' if name_flag else '') + '.pt')}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Sparse Autoencoders (SAEs) on PACS domain adaptation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default behavior (original script logic)
  python SAE_Train_V2.py
  
  # Train all models in custom base directory with custom SAE config
  python SAE_Train_V2.py --base-dir ./my_models --nb-concepts 7680 --top-k 64 --save-path ./my_saes
  
  # Train specific model from pickle path with custom name
  python SAE_Train_V2.py --model-pickle-path ./path/to/model.pkl --model-name ERM_ResNet50_T01 \\
    --nb-concepts 7680 --top-k 64 --save-path ./my_saes
  
  # Resume training from checkpoint
  python SAE_Train_V2.py --model-pickle-path ./path/to/model.pkl --model-name ERM_ResNet50_T01 \\
    --nb-concepts 7680 --top-k 64 --checkpoint ./saved_sae.pt --epochs 100
  
  # Control training hyperparameters
  python SAE_Train_V2.py --base-dir ./my_models --nb-concepts 7680 --top-k 64 \\
    --learning-rate 1e-3 --epochs 500 --save-path ./my_saes
        """
    )
    
    # Base directory arguments
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory containing model directories (for batch training). "
             "If not specified, defaults to './oracle_selection_models'"
    )
    
    # Single model arguments
    parser.add_argument(
        "--model-pickle-path",
        type=str,
        default=None,
        help="Direct path to a model pickle file (model.pkl). "
             "When used, only this single model will be trained. "
             "Must be used with --model-name."
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom name for the model in the format: ALGO_BACKBONE_TESTENVS "
             "(e.g., 'ERM_ResNet50_T01'). Required when using --model-pickle-path."
    )
    
    # SAE configuration arguments
    parser.add_argument(
        "--nb-concepts",
        type=int,
        default=None,
        help="Number of concepts in the SAE. REQUIRED - no defaults assumed. "
             "Example: 7680 (for 768*10)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sparsity value for the SAE. REQUIRED - no defaults assumed. "
             "Example: 64"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for the optimizer. Defaults to 3e-4 if not specified."
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. Defaults to 250 if not specified."
    )
    
    # Checkpoint and save paths
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a saved SAE checkpoint to resume training from."
    )
    
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory where trained SAEs will be saved. "
             "If not specified, defaults to './oracle_saes'"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training. Defaults to 64 if not specified."
    )
    
    parser.add_argument(
        "--full-data-gpu",
        action="store_true",
        default=False,
        help="If set, load all data to GPU at start (faster but requires more VRAM). "
             "By default, uses standard CPU-based loader with parallel workers."
    )
    
    parser.add_argument(
        "--enable-logging",
        action="store_true",
        default=False,
        help="If set, enable loss logging to file and create live loss curve. "
             "Requires --logging-dir to be specified."
    )
    
    parser.add_argument(
        "--logging-dir",
        type=str,
        default=None,
        help="Directory where loss logs and loss curve plots will be saved. "
             "Required if --enable-logging is set."
    )
    
    parser.add_argument(
        "--name-flag",
        type=str,
        default=None,
        help="Optional flag to append to the saved SAE filename. "
             "For example, '--name-flag test01' saves as 'SAE_ERM_ViT_T3_test01.pt'"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate argument combinations."""
    # Check model-specific arguments
    if args.model_pickle_path is not None:
        if args.model_name is None:
            raise ValueError(
                "--model-name is required when using --model-pickle-path. "
                "Format: ALGO_BACKBONE_TESTENVS (e.g., 'ERM_ResNet50_T01')"
            )
        if not os.path.exists(args.model_pickle_path):
            raise FileNotFoundError(f"Model pickle not found at: {args.model_pickle_path}")
    
    # Check SAE configuration
    if args.nb_concepts is None or args.top_k is None:
        raise ValueError(
            "SAE configuration parameters --nb-concepts and --top-k are REQUIRED. "
            "No defaults are assumed. Please specify both values explicitly."
        )
    
    # Check checkpoint exists if provided
    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
    
    # Check logging configuration
    if args.enable_logging:
        if args.logging_dir is None:
            raise ValueError(
                "--logging-dir is required when --enable-logging is set. "
                "Please specify the directory where logs and loss curves will be saved."
            )


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate arguments
    validate_arguments(args)
    
    # Set default paths
    base_dir = args.base_dir if args.base_dir is not None else "./oracle_selection_models"
    save_path = args.save_path if args.save_path is not None else "./oracle_saes"
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Single model training mode
    if args.model_pickle_path is not None:
        print(f"\n{'='*60}")
        print(f"Training single model from: {args.model_pickle_path}")
        print(f"Model name: {args.model_name}")
        print(f"{'='*60}\n")
        
        model_path = args.model_pickle_path
        name = args.model_name
        
        checkpoint = torch.load(model_path)
        algo_name, model_name, _ = name.split("_")   

        # Parse model name to get algorithm
        try:
            backbone = load_backbone(name, model_path)
        except ValueError:
            raise ValueError(
                f"Model name '{name}' is invalid. Expected format: ALGO_BACKBONE_TESTENVS "
                "(e.g., 'ERM_ResNet50_T01') OR potential issue with loadbackbone function"
            )
        

        rearrange_string='n t d -> (n t) d'
        if 'ResNet' in model_name:
            rearrange_string = 'n c w h -> (n w h) c'

        train_pacs_SAEs(
            backbone, 
            save_path, 
            name,
            rearrange_string=rearrange_string,
            nb_concepts=args.nb_concepts,
            top_k=args.top_k,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            full_data_gpu=args.full_data_gpu,
            enable_logging=args.enable_logging,
            logging_dir=args.logging_dir,
            name_flag=args.name_flag
        )
    
    # Batch training mode (default behavior)
    else:
        print(f"\n{'='*60}")
        print(f"Batch training mode - processing models in: {base_dir}")
        print(f"{'='*60}\n")
        
        for name in os.listdir(base_dir):
            if name.startswith("SagNet") or name.startswith("1transfer"):
                continue

            sae_filename = f"SAE_{name}.pt"
            if sae_filename in os.listdir(save_path):
                print(f"Skipping {name} - SAE already exists")
                continue
            else:
                print(f"Processing {name}")

            try:
                algo_name, backbone_name, testenv = name.split("_")   
            except:
                print(f"Skipping {name} - could not parse model name")
                continue

            model_path = os.path.join(base_dir, name, "model.pkl")
            
            if not os.path.exists(model_path):
                print(f"Skipping {name} - model.pkl not found at {model_path}")
                continue
            
            checkpoint = torch.load(model_path)

            ModelClass = algo_classes[algo_name]
            
            backbone = ModelClass(
                input_shape=checkpoint["model_input_shape"],
                hparams=checkpoint["model_hparams"],
                num_domains=checkpoint["model_num_domains"],
                num_classes=checkpoint["model_num_classes"]
            )
            
            train_pacs_SAEs(
                backbone, 
                save_path, 
                name,
                nb_concepts=args.nb_concepts,
                top_k=args.top_k,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                checkpoint_path=args.checkpoint,
                batch_size=args.batch_size,
                full_data_gpu=args.full_data_gpu,
                enable_logging=args.enable_logging,
                logging_dir=args.logging_dir,
                name_flag=args.name_flag
            )


if __name__ == "__main__":
    main()
