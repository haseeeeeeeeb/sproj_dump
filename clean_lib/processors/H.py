import torch
from tqdm import tqdm
from clean_lib.data import Load_PACS
from clean_lib.utils import extract_features
from clean_lib.processors.processor import Processor
from einops import rearrange

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class H(Processor):
    def __init__(self, sae_manager, ckpt, process_domains, file_path, dataset="PACS"):
        super().__init__(sae_manager, ckpt, process_domains, file_path, dataset)

    @classmethod
    def from_processor(cls, processor: Processor):
        return cls(
            sae_manager=processor.sae_manager,
            ckpt=processor.ckpt,
            process_domains=processor.process_domains,
            file_path=processor.file_path,
            dataset=processor.dataset,
    )

    def calculate_mean_activations(self):
        z = torch.zeros((7, self.sae_manager.nb_concepts, len(self.domains))).to(device)
        counts = torch.zeros((7, self.sae_manager.nb_concepts)).to(device)
        n_images = torch.zeros((7, len(self.domains))).to(device)
        self.backbone.to(device)
        self.sae.to(device)


        for d, domain in enumerate(self.domains):

            if self.dataset == "PACS":
                loader, _ = Load_PACS(domains=[domain])

            for i, batch in enumerate(tqdm(loader)):
                with torch.no_grad():
                    img, y = batch
                    img, y = img.to(device), y.to(device)

                    x = extract_features(self.backbone, img)
                    x = self.sae.normalizer(x)
                    x = rearrange(x, self.sae_manager.rearrange_string)

                    _, heatmaps = self.sae.encode(x)

                    heatmaps = rearrange(heatmaps, '(n w h) d -> n w h d', w=self.sae_manager.w, h=self.sae_manager.w)
                    heatmaps_summed = heatmaps.sum(dim=1).sum(dim=1)  # (n, d)

                    for cls in range(7):
                        mask = (y == cls).squeeze()
                        if mask.sum() == 0:
                            continue

                        cls_heatmaps = heatmaps_summed[mask]           # (n_cls, d)
                        z[cls, :, d] += cls_heatmaps.sum(dim=0)
                        n_images[cls, d] += mask.sum()
                        counts[cls] += (cls_heatmaps > 0).sum(dim=0)

        n_images_safe = n_images.unsqueeze(1).clamp(min=1)
        z = z / n_images_safe

        return z, counts
    
    def calculate_invariance(self, z):

        invariance = torch.zeros((self.classes, self.sae_manager.nb_concepts)).to(z.device)

        for cls in range(self.classes):
            processed = z[cls]  # (nb_concepts, domains)

            for i in range(self.sae_manager.nb_concepts):
                if processed[i, :].sum() == 0:
                    continue

                score = processed[i, :] / processed[i, :].sum()
                entropy = -1 / torch.log(torch.tensor(float(len(self.domains)))) * (score * torch.log(score + 1e-12)).sum()
                invariance[cls, i] = entropy

        return invariance

    def process(self):
        activations, counts = self.calculate_mean_activations()
        invariance = self.calculate_invariance(activations)

        self.dump(invariance, "H")
        self.dump(activations, "H_mean_acts")
        self.dump(counts, "H_counts")
        