import torch
from torch.utils.data import Dataset
from einops import rearrange

class ZSAEHeatmapExtractor(Dataset):
    def __init__(self, activation_loader, sae_model, device, patch_width):
        self.device = device
        self.sae = sae_model.to(device)
        self.patch_width = patch_width

        self.heatmaps = []

        self._extract_heatmaps(activation_loader)

    def _extract_heatmaps(self, loader):
        self.sae.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                print(f"Processing batch {batch_idx + 1}/{len(loader)}")

                x = batch['activations_Z_SAE'].squeeze().to(self.device)  # [B, D]
                _, heatmap = self.sae.encode(x)  # [B * w * h, latent_dim]

                # Rearrange: (n w h) d → n w h d
                heatmap = rearrange(
                    heatmap,
                    '(n w h) d -> n w h d',
                    w=self.patch_width,
                    h=self.patch_width
                )

                self.heatmaps.append(heatmap.cpu())  # store on CPU for saving

        self.heatmaps = torch.cat(self.heatmaps, dim=0)  # [N, w, h, d]

    def __len__(self):
        return self.heatmaps.shape[0]

    def __getitem__(self, idx):
        return self.heatmaps[idx]
