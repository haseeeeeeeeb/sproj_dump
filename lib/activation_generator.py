import torch
from lib.data_handlers import Load_ImageNet100
from overcomplete.models import DinoV2, ViT, ResNet
from torch.utils.data import DataLoader, TensorDataset
import os
from torch.utils.data import Dataset, DataLoader
import glob
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
    

# def get_normalization_parameters(model):
#     image_loader = Load_ImageNet100(transform=None, batch_size=1024, shuffle=True)
    
#     img, _ = next(iter(image_loader))
    
#     if hasattr(model, 'featurizer'):
#         activations = model.featurizer(img.to(device))

#     if hasattr(model, 'forward_features'):
#         activations = model.forward_features(img.to(device))
    
#     flat = activations.flatten() 
#     mean = flat.mean() 
#     std = flat.std()
#     return mean, std


def generate_activations(models, image_dataloader, max_seq_len, save_dir, rearrange_string):
    os.makedirs(save_dir, exist_ok=True)

    for _, model in models.items():
        model.to(device)

    mean, std = 0.0, 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(image_dataloader, desc="Processing Batches")):
            images, labels = images.to(device), labels.to(device)
            batch_data = {'images': images.cpu(), 'labels': labels.cpu()}
            normal = {}
            for key, model in models.items():

                normal[key] = Normalizer()
                
                if hasattr(model, 'featurizer'): # for models trained using domainbed
                    if model.featurizer.__class__.__name__ == "DinoV2" :
                        activations = model.featurizer.network.forward_features(images.to(device))['x_norm_patchtokens']
                    elif model.featurizer.__class__.__name__ == "ViT":
                        activations = model.featurizer.network.forward_features(images.to(device))[:, 1:, :]
                    else:
                        activations = model.featurizer(images.to(device))

                if hasattr(model, 'forward_features'): # for models directly from the overcomplete library
                    activations = model.forward_features(images.to(device))

                # Interpolate Activation Tokens
                if max_seq_len is not None:
                    activations = activations.permute(0, 2, 1)  # Shape: (300, 382, 192)
                    x_interp = F.interpolate(activations, size=max_seq_len, mode='linear', align_corners=False)
                    activations = x_interp.permute(0, 2, 1)

                # Normalize Activations (if first batch else reuse the same mean and std)
                if i == 0:
                    normal[key].populate(activations)
                
                activations = normal[key].run(activations)
                
                # Batch Collapse
                activations = rearrange(activations, rearrange_string)
                batch_data[f"activations_{key}"] = activations.cpu()

            torch.save(batch_data, os.path.join(save_dir, f"batch_{i:05d}.pt"))

class PTFilesDataset(Dataset):
    def __init__(self, directory_path):
        """
        Args:
            directory_path (string): Path to the directory containing .pt files
        """
        self.file_paths = sorted(glob.glob(os.path.join(directory_path, '*.pt')))
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load the dictionary from the .pt file
        data_dict = torch.load(self.file_paths[idx])
        
        # Assuming the dictionary contains tensors that can be directly used
        # You might need to modify this part based on your specific data structure
        return data_dict


def Load_activation_dataloader(models, image_dataloader, save_dir, generate, max_seq_len, rearrange_string=None):
    

    if generate == True and rearrange_string is not None:
        generate_activations(models, image_dataloader, max_seq_len, save_dir, rearrange_string)
    
    dataset = PTFilesDataset(directory_path=save_dir)

    activation_dataloader = DataLoader(
        dataset,
        batch_size=1,          # Adjust batch size as needed
        shuffle=True,           # Shuffle if training
        drop_last=True,        # Whether to drop last incomplete batch
    )

    return activation_dataloader