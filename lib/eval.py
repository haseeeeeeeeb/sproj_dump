import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def evaluate_models(models_dict,
                    val_root,
                    batch_size: int = 128,
                    num_workers: int = 4,
                    device: torch.device = None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])
    val_dataset = datasets.ImageFolder(val_root, transform=val_transforms)
    val_loader  = DataLoader(val_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    accuracies = {}
    for name, model in models_dict.items():
        model = model.to(device)
        model.eval()

        correct = 0
        total   = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs    = imgs.to(device)
                targets = targets.to(device)

                outputs = model.model(imgs)
                _, preds = outputs.max(1)

                correct += (preds == targets).sum().item()
                total   += targets.size(0)

        accuracies[name] = correct / total

    return accuracies
