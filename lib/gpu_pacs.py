import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class PACSAllDomains(Dataset):
    def __init__(self, root_dir, domains=None, transform=None, device='cpu'):
        if domains is None:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']
        
        self.device = torch.device(device)
        self.transform = transform
        
        samples_info = []
        class_to_idx = {}
        classes = set()

        for domain in domains:
            domain_dir = os.path.join(root_dir, domain)
            for class_name in sorted(entry.name for entry in os.scandir(domain_dir) if entry.is_dir()):
                classes.add(class_name)
                class_dir = os.path.join(domain_dir, class_name)
                for file_name in os.listdir(class_dir):
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        samples_info.append((os.path.join(class_dir, file_name), class_name))

        classes = sorted(classes)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        self.data = []
        self.labels = []
        
        for img_path, class_name in samples_info:
            image = Image.open(img_path).convert("RGB")
            
            if self.transform:
                image_tensor = self.transform(image)
            else:
                image_tensor = transforms.ToTensor()(image) 
            
            self.data.append(image_tensor.to(self.device))
            self.labels.append(class_to_idx[class_name])
            
        if self.data:
            self.data = torch.stack(self.data)
            self.labels = torch.tensor(self.labels, dtype=torch.long, device=self.device)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_pacs_gpuloader(
    root_dir=r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS",
    batch_size=64,
    domains=None,
    shuffle=True,
    transform=None,
    train_split=0.8,
    seed=42,
    drop_last=True
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    dataset = PACSAllDomains(root_dir=root_dir, transform=transform, domains=domains, device=device)

    # --- Single split only ---
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    # Just use CPU generator or none at all
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Return *one* loader — training only
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )

    return train_loader


def get_pacs_standard_loader(
    root_dir=r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS",
    batch_size=64,
    domains=None,
    shuffle=True,
    transform=None,
    train_split=0.8,
    seed=42,
    drop_last=True,
    num_workers=4
):
    """Standard data loader that keeps data on CPU and loads batches dynamically."""
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    dataset = PACSAllDomains(root_dir=root_dir, transform=transform, domains=domains, device='cpu')

    # --- Single split only ---
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Return loader with parallel workers for efficient CPU-GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader