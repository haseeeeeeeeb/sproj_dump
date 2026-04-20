import torch
from tqdm import tqdm
from clean_lib.data import Load_PACS
from clean_lib.utils import extract_features
from clean_lib.processors.processor import Processor
from einops import rearrange
from torch.nn import functional as F
from timm.layers import SelectAdaptivePool2d


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class M(Processor):
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

    def calculate_m_scores(self):
        # delta_p:      (nb_classes, nb_concepts, nb_domains) normalised by TOTAL class images per domain
        delta_p, class_counts = self._calculate_discrimination_scores_per_domain()
        class_counts_exp = class_counts.unsqueeze(1)                                                              # (nb_classes, 1, nb_domains)
        delta_p_mean = (delta_p * class_counts_exp).sum(dim=2) / class_counts_exp.sum(dim=2).clamp(min=1)         # (nb_classes, nb_concepts)

        print(delta_p_mean.shape)
        
        signal_scores       = torch.zeros(self.classes, self.sae_manager.nb_concepts, device=device)
        interference_scores = torch.zeros(self.classes, self.sae_manager.nb_concepts, device=device)
        contrast_scores     = torch.zeros(self.classes, self.sae_manager.nb_concepts, device=device)

        per_class_profiles = delta_p_mean.T.unsqueeze(0).expand(self.classes, -1, -1).clone()
        
        for i in range(self.sae_manager.nb_concepts):
            concept_scores = delta_p_mean[:, i]                             # (nb_classes,)

            if (concept_scores == 0.0).all():
                continue

            k_star = concept_scores.argmax()
            signal = concept_scores[k_star]

            other_mask         = torch.ones(self.classes, dtype=torch.bool, device=device)
            other_mask[k_star] = False
            interference       = concept_scores[other_mask].min()          # typically <= 0 for conflicted concepts

            contrast = signal - interference                                 # WTA contrast (Option 3)

            # Broadcast the concept-level scalar across all class rows
            signal_scores[:, i]       = signal
            interference_scores[:, i] = interference
            contrast_scores[:, i]     = contrast

        return contrast_scores, signal_scores, interference_scores, delta_p_mean, per_class_profiles

    def _calculate_discrimination_scores_per_domain(self):

        pool               = SelectAdaptivePool2d(pool_type='avg', flatten=True)
        accumulated_scores = torch.zeros(self.classes, self.sae_manager.nb_concepts, len(self.domains), device=device)
        class_counts       = torch.zeros(self.classes, len(self.domains), device=device)

        self.sae.to(device)
        self.backbone.to(device)

        for d, domain in enumerate(self.domains):
            if self.dataset == "PACS":
                dataloader, _ = Load_PACS(domains=[domain], batch_size=64)

            for x, y in tqdm(dataloader, desc=f"M scores [{domain}]"):
                x, y = x.to(device), y.to(device)
                n = x.size(0)

                # Count every image toward its class total, regardless of concept activation
                class_counts[:, d].scatter_add_(0, y, torch.ones(n, device=device))

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
                        z_recon_m      = rearrange(z_recon_flat_m, '(n h w) c -> n c h w', n=n, h=h, w=w)
                        z_recon_m      = self.sae.normalizer.denormalize(z_recon_m)

                        logits_masked = self.backbone.classifier(pool(z_recon_m))
                        probs_masked  = F.softmax(logits_masked, dim=1)
                        p_true_masked = probs_masked[torch.arange(n), y]

                        z_sae[:, c] = original_col

                        score_drop     = p_true_unmasked - p_true_masked    # positive = concept was helping
                        active_classes = y[active_img_mask]
                        active_drops   = score_drop[active_img_mask]

                        # Accumulate raw drops only for active images;
                        # division by total class_counts (below) handles the inactive-image penalty
                        accumulated_scores[:, c, d].scatter_add_(0, active_classes, active_drops)

        # Normalise by TOTAL class images per domain (not activation count)
        norm    = class_counts.unsqueeze(1).clamp(min=1)                    # (nb_classes, 1, nb_domains)
        delta_p = accumulated_scores / norm                                  # (nb_classes, nb_concepts, nb_domains)

        return delta_p, class_counts

    def process(self):
        contrast_scores, signal_scores, interference_scores, delta_p_mean, per_class_profiles = self.calculate_m_scores()

        self.dump(contrast_scores,                  name="M_contrast")      # (nb_classes, nb_concepts)             WTA contrast, broadcast across class rows
        self.dump(signal_scores,                    name="M_signal")        # (nb_classes, nb_concepts)             best-class delta_p, broadcast across class rows
        self.dump(torch.abs(interference_scores),   name="M_interference")  # (nb_classes, nb_concepts)             mean other-class delta_p, broadcast across class rows
        self.dump(delta_p_mean,                     name="M_delta_p")       # (nb_classes, nb_concepts)             mean other-class delta_p, broadcast across class rows
        self.dump(per_class_profiles,               name="M_profiles")      # (nb_classes, nb_concepts, nb_classes) full delta_p profile per concept, indexed by class