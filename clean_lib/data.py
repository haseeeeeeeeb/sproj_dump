import os
import torch
from PIL import Image
from torchvision import transforms
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		

class PACSDataset(Dataset):
	def __init__(
		self,
		root_dir: str,
		domains: Optional[List[str]] = None,
		transform=None,
		preload_to_gpu: bool = False,
	) -> None:
		
		if domains is None:
			domains = ["photo", "art_painting", "cartoon", "sketch"]

		self.root_dir = root_dir
		self.domains = domains
		self.transform = transform
		self.preload_to_gpu = preload_to_gpu
		self.device = device

		samples_info = []  # (path, class_name)
		classes = set()

		for domain in domains:
			domain_dir = os.path.join(root_dir, domain)
			for class_entry in sorted(os.scandir(domain_dir), key=lambda e: e.name):
				if not class_entry.is_dir():
					continue
				class_name = class_entry.name
				classes.add(class_name)
				class_dir = os.path.join(domain_dir, class_name)
				for file_name in os.listdir(class_dir):
					if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
						samples_info.append((os.path.join(class_dir, file_name), class_name))

		classes = sorted(classes)
		class_to_idx = {name: idx for idx, name in enumerate(classes)}

		self.preloaded = preload_to_gpu and torch.cuda.is_available()

		if self.preloaded:
			data_tensors = []
			labels = []
			for img_path, class_name in samples_info:
				image = Image.open(img_path).convert("RGB")
				if self.transform is not None:
					tensor = self.transform(image)
				else:
					tensor = transforms.ToTensor()(image)
				data_tensors.append(tensor.to(self.device))
				labels.append(class_to_idx[class_name])

			self.data = torch.stack(data_tensors) if data_tensors else torch.empty(0)
			self.labels = torch.tensor(labels, dtype=torch.long, device=self.device)
			self._length = self.data.shape[0]
		else:
			self.samples = [(path, class_to_idx[class_name]) for path, class_name in samples_info]
			self._length = len(self.samples)

	def __len__(self) -> int:
		return self._length

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		if self.preloaded:
			return self.data[idx], self.labels[idx]

		img_path, label = self.samples[idx]
		image = Image.open(img_path).convert("RGB")
		if self.transform is not None:
			tensor = self.transform(image)
		else:
			tensor = transforms.ToTensor()(image)
		return tensor, torch.tensor(label, dtype=torch.long)


def Load_PACS(
	root_dir: str = r"C:\Users\sproj_ha\Desktop\DomainBed\domainbed\data\PACS",
	domains: Optional[List[str]] = None,
	batch_size: int = 64,
	train_split: float = 0.8,
	seed: int = 42,
	shuffle_train: bool = True,
	drop_last: bool = True,
	preload_to_gpu: bool = False,
	num_workers: int = 4,
):
	if train_split <= 0.0 or train_split >= 1.0:
		raise ValueError("train_split must be in (0, 1)")

	if domains is None:
		domains = ["photo", "art_painting", "cartoon", "sketch"]

	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	dataset = PACSDataset(
		root_dir=root_dir,
		domains=domains,
		transform=transform,
		preload_to_gpu=preload_to_gpu,
	)

	train_size = int(train_split * len(dataset))
	test_size = len(dataset) - train_size

	# fixed generator for reproducible splits
	g = torch.Generator().manual_seed(seed)
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=g)

	if preload_to_gpu and torch.cuda.is_available():
		train_loader = DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=shuffle_train,
			drop_last=drop_last,
			num_workers=0,
		)
		test_loader = DataLoader(
			test_dataset,
			batch_size=batch_size,
			shuffle=False,
			drop_last=drop_last,
			num_workers=0,
		)
	else:
		pin_memory = False
		train_loader = DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=shuffle_train,
			drop_last=drop_last,
			num_workers=num_workers,
			pin_memory=pin_memory,
		)
		test_loader = DataLoader(
			test_dataset,
			batch_size=batch_size,
			shuffle=False,
			drop_last=drop_last,
			num_workers=num_workers,
			pin_memory=pin_memory,
		)

	return train_loader, test_loader

