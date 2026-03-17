import time
import torch
from overcomplete.sae.trackers import DeadCodeTracker
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from lib.logger import TrainingLogger
import os
from typing import Dict, List, Sequence

# Assume I have a batch inside a dataloader corresponding to: image_model1, activations_model1 etc


def train_usae(
    names: Sequence[str],
    models: Dict[str, torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    schedulers: Dict[str, any] | None = None,
    nb_epochs: int = 20,
    clip_grad: float | None = None,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> None:

    logger = TrainingLogger(log_root="logs", seed=seed)
    logger.log_hyperparams(
        {
            "names": list(names),
            "nb_epochs": nb_epochs,
            "clip_grad": clip_grad,
            "device": str(device),
            "criterion": criterion.__class__.__name__,
        }
    )
    logger.log_config(
        sae_configs=models,
        optimizers=optimizers,
        schedulers=schedulers or {},
        criterion=criterion,
    )
    logger.log_dataloader(dataloader)

    total_loss_per_epoch = []  # Total loss over all models
    individual_loss_per_epoch = defaultdict(list)  # Per-model losses per epoch

    for epoch in range(nb_epochs):

        start_time = time.time()
        epoch_loss = 0.0
        epoch_model_losses = {name: 0.0 for name in names}  # Reset per-model epoch loss
        dead_tracker = None
        rotator = 0

        for name in names:
            models[name].to(device)

        # tqdm dataloader
        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{nb_epochs}",
        )

        for batch_count, batch in progress_bar:
            # Reset Everything
            total_loss = 0.0
            for name in names:
                optimizers[name].zero_grad()
                
            # Current SAE
            current = names[rotator]

            # Current SAE model
            sae = models[current]
            sae.train()

            # Encoder Forward Pass
            z_pre, z = sae.encode(
                batch[f"activations_{names[rotator]}"].squeeze().to(device)
            )
            # print("Z Shape: ", z.shape)

            # Decoder across all models & accumulate loss
            for n, m in models.items():

                # if n == current:
                #     x_hat = m.decode(z)
                # else:
                #     #with torch.no_grad():
                        #x_hat = m.decode(z.detach())
                x_hat = m.decode(z)

                target = batch[f"activations_{n}"].squeeze().to(device)
                loss = criterion(x_hat, target)
            
                epoch_model_losses[n] += loss.item()
                total_loss += loss

            total_loss.backward()
            
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), clip_grad)

            optimizers[current].step()
            if schedulers:
                schedulers[current].step()

            # Rotator Update
            rotator += 1
            rotator = rotator % len(names)

            # # Dead feature tracking
            # if dead_tracker is None:
            #     dead_tracker = DeadCodeTracker(z.shape[1], device)
            # dead_tracker.update(z.detach())

            # Update metrics and tqdm bar
            batch_loss = total_loss.item()
            epoch_loss += batch_loss
            progress_bar.set_postfix(loss=batch_loss)
            logger.log_batch_loss(current, batch_loss)

        # epoch summary
        #dead_features = dead_tracker.get_dead_ratio()
        epoch_time = time.time() - start_time
        print(
            f"\n[Epoch {epoch+1}] Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s%"
        )

        total_loss_per_epoch.append(epoch_loss)
        for name in names:
            individual_loss_per_epoch[name].append(epoch_model_losses[name])

        logger.log_epoch_metrics(
            epoch,
            epoch_loss,
            epoch_model_losses,
            epoch_time=epoch_time,
        )
        logger.plot_losses()

    logger.finalize()
    return
