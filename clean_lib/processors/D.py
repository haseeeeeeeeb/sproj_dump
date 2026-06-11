
import torch
from tqdm import tqdm
from clean_lib.data import Load_PACS
from clean_lib.utils import extract_features
from clean_lib.processors.processor import Processor
from einops import rearrange
from torch.nn import functional as F
from timm.layers import SelectAdaptivePool2d


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class D(Processor):
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


    def calculate_discrimination_scores(self):
         
        pool = SelectAdaptivePool2d(pool_type='avg', flatten=True)
        accumulated_scores = torch.zeros((self.classes, self.sae_manager.nb_concepts), device=device)
        activation_counts = torch.zeros((self.classes, self.sae_manager.nb_concepts), device=device)
        

        dataloader, _ = Load_PACS(domains=self.domains, batch_size=64)
        
        self.sae.to(device)
        self.backbone.to(device)

        for x, y in tqdm(dataloader, desc="Calculating Concept Discrimination"):
            x, y = x.to(device), y.to(device)
            n = x.size(0)
            
            with torch.no_grad():
                # --- Base Unmasked Pass ---
                z_raw = extract_features(self.backbone, x)
                z_norm = self.sae.normalizer(z_raw)
                _, _, h, w = z_norm.shape
                
                z_flat = rearrange(z_norm, "n c h w -> (n h w) c")
                _, z_sae = self.sae.encode(z_flat)
                
                z_recon_flat = self.sae.decode(z_sae)
                z_recon = rearrange(z_recon_flat, '(n h w) c -> n c h w', n=n, h=h, w=w)
                z_recon = self.sae.normalizer.denormalize(z_recon)
                


                logits_unmasked = self.backbone.classifier(pool(z_recon))
                probs_unmasked = F.softmax(logits_unmasked, dim=1)
                p_true_unmasked = probs_unmasked[torch.arange(n), y] 
                
                # --- Fast Masking Setup ---
                z_sae_img = rearrange(z_sae, '(n h w) c -> n (h w) c', n=n, h=h, w=w)
                concept_max_per_img, _ = z_sae_img.max(dim=1) # Shape: (n, nb_concepts)
                
                # Find concepts active at least once in the batch
                active_concepts_batch = torch.where(concept_max_per_img.max(dim=0)[0] > 0)[0]
                
                for c in active_concepts_batch:
                    active_img_mask = concept_max_per_img[:, c] > 0
                    
                    if not active_img_mask.any():
                        continue
                        
                    # OPTIMIZATION 1: In-place masking to save memory
                    original_col = z_sae[:, c].clone()
                    z_sae[:, c] = 0 
                    
                    # Forward pass with the single masked concept
                    z_recon_flat_m = self.sae.decode(z_sae) # z_sae is currently masked
                    z_recon_m = rearrange(z_recon_flat_m, '(n h w) c -> n c h w', n=n, h=h, w=w)
                    z_recon_m = self.sae.normalizer.denormalize(z_recon_m)
                    

                    if self.backbone.featurizer.__class__.__name__ == "ResNet":
                        logits_masked = self.backbone.classifier(pool(z_recon_m))
                    else:
                        logits_masked = self.backbone.classifier(z_recon_m)
    

    
                    probs_masked = F.softmax(logits_masked, dim=1)
                    p_true_masked = probs_masked[torch.arange(n), y]
                    
                    # Restore the original concept activations for the next loop iteration
                    z_sae[:, c] = original_col
                    
                    # Calculate drop
                    score_drop = p_true_unmasked - p_true_masked 
                    
                    # Filter to active images
                    active_classes = y[active_img_mask]
                    active_drops = score_drop[active_img_mask]
                    
                    # OPTIMIZATION 2: Vectorized GPU accumulation (No CPU sync)
                    accumulated_scores[:, c].scatter_add_(0, active_classes, active_drops)
                    activation_counts[:, c].scatter_add_(0, active_classes, torch.ones_like(active_drops))

        # Average the scores (clamp denominator to prevent div by zero)
        final_discrimination_scores = accumulated_scores / activation_counts.clamp(min=1)
        
        return final_discrimination_scores, activation_counts


    def process(self):
        scores, counts = self.calculate_discrimination_scores()
        self.dump(scores, "D")
        self.dump(counts, "D_counts")

