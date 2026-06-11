import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from collections import defaultdict



class ImageNet100(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.class_to_idx = {}
        self.classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        
        for idx, class_name in enumerate(self.classes):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, file_name), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def Load_CIFAR(train = True, batch_size = 64, shuffle: bool = True, transform=None):
    """
    Loads the CIFAR-10 dataset and returns a DataLoader.

    Args:
        train (bool): If True, loads training data. If False, loads test data.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        transform (callable, optional): Optional transform to apply to the data.

    Returns:
        DataLoader: PyTorch DataLoader for CIFAR-10.
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

    dataset = datasets.CIFAR10(
        root='./datasets',
        train=train,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def Load_MNIST(train = True, batch_size = 64, shuffle = True, transform=None):
    """
    Loads the MNIST dataset and returns a DataLoader.

    Args:
        train (bool): If True, loads training data. If False, loads test data.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        transform (callable, optional): Optional transform to apply to the data.

    Returns:
        DataLoader: PyTorch DataLoader for MNIST.
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalization
        ])

    dataset = datasets.MNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader




def Load_ImageNet100(root_dir=r"C:\Users\sproj_ha\Desktop\vision_interp\datasets\imagenet100", train=True, batch_size=64, shuffle=False, transform=None, dataset_allow=False):
    """
    Returns a DataLoader for the custom ImageNet-100 dataset.

    Args:
        root_dir (str): Path to dataset root.
        train (bool): Whether to load training or test split.
        batch_size (int): Batch size.
        shuffle (bool): Shuffle the dataset.
        transform (callable, optional): Image transform.

    Returns:
        DataLoader: PyTorch DataLoader for ImageNet-100.
    """
    
    if train:
        root_dir = os.path.join(root_dir, "train.X1")
    else:
        root_dir = os.path.join(root_dir, "val.X")


    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
        ])

    dataset = ImageNet100(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    if not dataset_allow:
        return dataloader
    else:
        return dataloader, dataset



def Load_ImageNet100Sketch(root_dir=r"C:\Users\sproj_ha\Desktop\vision_interp\datasets\imagenetsketch\sketch", train=True, batch_size=64, shuffle=False, transform=None):
    """
    Returns a DataLoader for the custom ImageNet-100 dataset.

    Args:
        root_dir (str): Path to dataset root.
        train (bool): Whether to load training or test split.
        batch_size (int): Batch size.
        shuffle (bool): Shuffle the dataset.
        transform (callable, optional): Image transform.

    Returns:
        DataLoader: PyTorch DataLoader for ImageNet-100.
    """
    
    

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
        ])

    dataset = ImageNet100(root_dir=root_dir, transform=transform)
    
    if train == False:
        train_size = int(0.9 * len(dataset))  # 90% train
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader

    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader



class PACSAllDomains(Dataset):
    def __init__(self, root_dir, domains=None, transform=None):
        """
        Loads PACS dataset from specified domains.

        Args:
            root_dir (str): Root directory of PACS dataset.
            domains (list of str, optional): List of domains to include. 
                                             Defaults to all 4 PACS domains.
            transform (callable, optional): Transform to apply to images.
        """
        if domains is None:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']
        
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = set()

        for domain in domains:
            domain_dir = os.path.join(root_dir, domain)
            for class_name in sorted(entry.name for entry in os.scandir(domain_dir) if entry.is_dir()):
                self.classes.add(class_name)
                class_dir = os.path.join(domain_dir, class_name)
                for file_name in os.listdir(class_dir):
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(class_dir, file_name), class_name))

        self.classes = sorted(self.classes)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = [(path, self.class_to_idx[label]) for path, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def Load_PACS(root_dir=r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS",
              batch_size=64,
              domains=None,
              shuffle=True,
              transform=None,
              train_split=0.8,
              seed=42,
              drop_last=True
              ):
    """
    Loads the PACS dataset and returns train/test DataLoaders.

    Args:
        root_dir (str): Path to the root directory of the PACS dataset.
        batch_size (int): Batch size.
        domains (list of str, optional): Domains to include.
        shuffle (bool): Shuffle training data.
        transform (callable, optional): Image transform.
        train_split (float): Proportion of data to use for training.
        seed (int): Random seed for reproducibility.

    Returns:
        train_loader (DataLoader), test_loader (DataLoader)
    """

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    dataset = PACSAllDomains(root_dir=root_dir, transform=transform, domains=domains)

    # compute sizes
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    # reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)

    return train_loader, test_loader

def get_pacs_data_loaders(
    root_dir=r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS",
    batch_size=64,
    train_domains=['photo', 'art_painting'],
    test_domains=['cartoon', 'sketch'], # <-- Change the argument name and type
    train_split=0.8,
    seed=42):
    """
    Loads the PACS dataset and returns data loaders as specified in the request.

    Args:
        root_dir (str): Path to the root directory of the PACS dataset.
        batch_size (int): Batch size for the data loaders.
        train_domains (list of str): List of domains to use for training.
        test_domains (list of str): The domains to use as the designated test sets.
        train_split (float): Proportion of data to use for training within each train domain.
        seed (int): Random seed for reproducibility.

    Returns:
        A dictionary containing:
            'train_loader': The combined train loader from all train domains.
            'test_loaders': A dictionary of test loaders, one for each domain (train and test).
    """
    
    # Ensure reproducibility
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # Standard transforms for ViT models
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    all_pacs_domains = ['photo', 'art_painting', 'cartoon', 'sketch']

    # Dictionary to hold datasets for each domain
    domain_datasets = {}
    for domain in all_pacs_domains:
        domain_datasets[domain] = PACSAllDomains(
            root_dir=root_dir, 
            domains=[domain], 
            transform=transform
        )
    
    # 1. Create a single train loader from all training domains
    train_datasets = []
    domain_test_datasets = defaultdict(list)

    for domain in train_domains:
        dataset = domain_datasets[domain]
        # Calculate split sizes for this domain
        train_size = int(train_split * len(dataset))
        test_size = len(dataset) - train_size
        
        # Split the dataset for the current domain
        train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=generator)
        
        train_datasets.append(train_ds)
        domain_test_datasets[domain] = test_ds

    # Combine all training splits into a single dataset
    combined_train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        combined_train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        generator=generator # Ensure shuffling is also reproducible
    )

    # 2. Create test loaders for all domains
    test_loaders = {}
    
    # Test loaders for the training domains (using the test splits)
    for domain, dataset in domain_test_datasets.items():
        test_loaders[f'test_{domain}'] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False
        )

    # Test loader for the designated test domains (using the whole dataset)
    for domain in test_domains:
        test_loaders[f'test_{domain}'] = DataLoader(
            domain_datasets[domain], 
            batch_size=batch_size, 
            shuffle=False
        )
    
    # Add a combined test loader for all test domains
    combined_test_dataset = ConcatDataset([domain_datasets[d] for d in test_domains])
    test_loaders['test_all_unseen'] = DataLoader(
        combined_test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return {'train_loader': train_loader, 'test_loaders': test_loaders}
