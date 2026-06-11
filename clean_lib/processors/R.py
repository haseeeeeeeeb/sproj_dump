import torch
from tqdm import tqdm
from clean_lib.data import Load_PACS
from clean_lib.utils import extract_features
from clean_lib.processors.processor import Processor
from einops import rearrange
from torch.nn import functional as F
from timm.layers import SelectAdaptivePool2d


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class R(Processor):
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

    def calculate_r_scores(self):
        
        discrimination_scores, counts = self._calculate_discrimination_scores_per_domain()

        r_scores = torch.zeros(self.classes, self.sae_manager.nb_concepts, device=device)

        for cls in range(self.classes):
            for i in range(self.sae_manager.nb_concepts):
                concept_scores = discrimination_scores[cls, i, :]  # (nb_domains,)
                
                if (concept_scores == 0.0).all():
                    continue

                score = concept_scores / concept_scores.sum()

                if (score < 0).any():
                    print(f"Warning: Negative score for class {cls}, concept {i}. Setting to zero.")
                    print(concept_scores)
                    entropy = -1

                else:
                    entropy = -1 / torch.log(torch.tensor(float(len(self.domains)))) * (score * torch.log(score + 1e-12)).sum()

                r_scores[cls, i] = entropy

        return r_scores, discrimination_scores, counts

    def _calculate_discrimination_scores_per_domain(self):

        pool = SelectAdaptivePool2d(pool_type='avg', flatten=True)
        accumulated_scores = torch.zeros(self.classes, self.sae_manager.nb_concepts, len(self.domains), device=device)
        activation_counts  = torch.zeros(self.classes, self.sae_manager.nb_concepts, len(self.domains), device=device)

        self.sae.to(device)
        self.backbone.to(device)

        for d, domain in enumerate(self.domains):
            if self.dataset == "PACS":
                dataloader, _ = Load_PACS(domains=[domain], batch_size=64)

            for x, y in tqdm(dataloader, desc=f"R scores [{domain}]"):
                x, y = x.to(device), y.to(device)
                n = x.size(0)

                with torch.no_grad():
                    z_raw  = extract_features(self.backbone, x)
                    z_norm = self.sae.normalizer(z_raw)
                    _, _, h, w = z_norm.shape

                    z_flat = rearrange(z_norm, 'n c h w -> (n h w) c')
                    _, z_sae = self.sae.encode(z_flat)

                    z_recon_flat = self.sae.decode(z_sae)
                    z_recon = rearrange(z_recon_flat, '(n h w) c -> n c h w', n=n, h=h, w=w)
                    z_recon = self.sae.normalizer.denormalize(z_recon)

                    logits_unmasked = self.backbone.classifier(pool(z_recon))
                    probs_unmasked  = F.softmax(logits_unmasked, dim=1)
                    p_true_unmasked = probs_unmasked[torch.arange(n), y]

                    z_sae_img = rearrange(z_sae, '(n h w) c -> n (h w) c', n=n, h=h, w=w)
                    concept_max_per_img, _ = z_sae_img.max(dim=1)
                    active_concepts_batch  = (concept_max_per_img.max(dim=0)[0] > 0).nonzero(as_tuple=False).squeeze(1)

                    for c in active_concepts_batch:
                        active_img_mask = concept_max_per_img[:, c] > 0

                        if not active_img_mask.any():
                            continue

                        original_col = z_sae[:, c].clone()
                        z_sae[:, c] = 0

                        z_recon_flat_m = self.sae.decode(z_sae)
                        z_recon_m = rearrange(z_recon_flat_m, '(n h w) c -> n c h w', n=n, h=h, w=w)
                        z_recon_m = self.sae.normalizer.denormalize(z_recon_m)

                        logits_masked = self.backbone.classifier(pool(z_recon_m))
                        probs_masked  = F.softmax(logits_masked, dim=1)
                        p_true_masked = probs_masked[torch.arange(n), y]

                        z_sae[:, c] = original_col

                        score_drop    = p_true_unmasked - p_true_masked
                        active_classes = y[active_img_mask]
                        active_drops   = score_drop[active_img_mask]

                        # Scatter-add into the domain slice
                        accumulated_scores[:, c, d].scatter_add_(0, active_classes, active_drops)
                        activation_counts[:, c, d].scatter_add_(0, active_classes, torch.ones_like(active_drops))

        return accumulated_scores / activation_counts.clamp(min=1), activation_counts
    
    def process(self):
        r_scores, disc_scores, counts = self.calculate_r_scores()
        self.dump(r_scores, name="R")
        self.dump(disc_scores, name="R_acts")
        self.dump(counts, name="R_counts")