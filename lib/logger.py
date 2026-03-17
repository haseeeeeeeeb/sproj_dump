from __future__ import annotations

"""TrainingLogger — safe YAML dumping & strict log paths
=======================================================
A fully self‑contained run‑logger that is robust against
PyYAML *RepresenterError* by sanitising every object before
writing YAML.  All artefacts live inside `self.log_dir`.
"""

import json
import os
import platform
import random
import socket
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Mapping, Sequence
from torch.optim.lr_scheduler import SequentialLR, ChainedScheduler  # new

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _json_default(obj: Any):
    """Fallback converter for objects that JSON can't handle natively."""
    if isinstance(obj, (torch.device, torch.dtype)):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return str(obj)


def _to_yaml_safe(obj: Any):
    """Recursively turn *obj* into YAML‑serialisable primitives.

    Scalars (str, int, float, bool, None) are passed through.  For
    *Mapping* and *Sequence* we sanitise their elements, otherwise we
    stringify the value.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, Mapping):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, Sequence):
        return [_to_yaml_safe(v) for v in obj]
    return str(obj)


# -----------------------------------------------------------------------------
# main logger
# -----------------------------------------------------------------------------


class TrainingLogger:
    """File‑system based run logger focused on reproducibility."""

    # ---------------------------------------------------------------------
    # initialise
    # ---------------------------------------------------------------------

    def __init__(
        self,
        log_root: str = "logs",
        run_name: str | None = None,
        seed: int | None = None,
    ):
        self.start_time = datetime.utcnow()
        self.run_name = run_name or self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_root, f"run_{self.run_name}")
        os.makedirs(self.log_dir, exist_ok=True)

        # seed all RNGs (Python / NumPy / Torch)
        self.seed = self._set_seed(seed)

        # internal state (lightweight – everything ends up on disk)
        self.history: Dict[str, Any] = {
            "run_name": self.run_name,
            "start_time": self.start_time.isoformat(timespec="seconds"),
            "seed": self.seed,
            "system": {},
            "hyperparams": {},
            "config": {},
            "dataloader": {},
            "epoch_metrics": [],
        }
        self.batch_step: Dict[str, int] = {}

        # capture environment once
        self._log_system_info()

    # ---------------------------------------------------------------------
    # public api
    # ---------------------------------------------------------------------

    def log_hyperparams(self, params: Dict[str, Any]):
        self.history["hyperparams"].update(params)
        self._dump("hyperparams.yaml", self.history["hyperparams"])

    def log_config(
        self,
        sae_configs: Dict[str, torch.nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Dict[str, Any] | None,
        criterion: torch.nn.Module,
    ):
        cfg: Dict[str, Any] = {
            "models": {},
            "optimizers": {},
            "schedulers": {},
            "criterion": {},
        }

        for name, sae in sae_configs.items():
            cfg["models"][name] = {
                k: v
                for k, v in sae.__dict__.items()
                if not k.startswith("_")
                and isinstance(v, (int, float, str, bool, torch.device))
            }

        for name, opt in optimizers.items():
            cfg["optimizers"][name] = {
                "type": opt.__class__.__name__,
                **{
                    k: v
                    for k, v in opt.defaults.items()
                    if isinstance(v, (int, float, str, bool))
                },
            }

        if schedulers:
            for name, sch in schedulers.items():
                if isinstance(sch, SequentialLR):
                    seq_info = {
                        "milestones": list(getattr(sch, "milestones", [])),
                        "schedulers": [
                            {
                                "type": child.__class__.__name__,
                                "params": {
                                    k: v
                                    for k, v in child.__dict__.items()
                                    if not k.startswith("_") and isinstance(v, (int, float, str, bool))
                                },
                            }
                            for child in sch._schedulers
                        ],
                    }
                    cfg["schedulers"][name] = {"type": "SequentialLR", "config": seq_info}
                    continue

                if isinstance(sch, ChainedScheduler):
                    chain_info = {
                        "schedulers": [c.__class__.__name__ for c in sch._schedulers]
                    }
                    cfg["schedulers"][name] = {"type": "ChainedScheduler", "config": chain_info}
                    continue

                sch_cfg = {
                    k: v
                    for k, v in sch.__dict__.items()
                    if not k.startswith("_") and isinstance(v, (int, float, str, bool))
                }
                cfg["schedulers"][name] = {"type": sch.__class__.__name__, "config": sch_cfg}

        cfg["criterion"] = {"type": criterion.__class__.__name__}
        self.history["config"] = cfg
        self._dump("config.yaml", cfg)

    def log_dataloader(self, dataloader: torch.utils.data.DataLoader):
        info = {
            "batch_size": dataloader.batch_size,
            "shuffle": getattr(dataloader, "shuffle", None),
            "num_workers": dataloader.num_workers,
            "drop_last": dataloader.drop_last,
            "sampler": dataloader.sampler.__class__.__name__,
            "dataset": dataloader.dataset.__class__.__name__,
        }
        self.history["dataloader"] = info
        self._dump("dataloader.yaml", info)

    def log_batch_loss(self, model_name: str, loss: float):
        step = self.batch_step.get(model_name, 0)
        record = {"step": step, "loss": loss}
        self.batch_step[model_name] = step + 1
        fn = os.path.join(self.log_dir, f"{model_name}_batch_losses.jsonl")
        with open(fn, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_json_default) + "\n")

    def log_epoch_metrics(
        self,
        epoch: int,
        total_loss: float,
        individual_losses: Dict[str, float],
        *,
        dead_ratio: float | None = None,
        epoch_time: float | None = None,
    ):
        entry = {
            "epoch": epoch,
            "total_loss": total_loss,
            "individual_losses": individual_losses,
            "dead_ratio": dead_ratio,
            "epoch_time_sec": epoch_time,
        }
        self.history["epoch_metrics"].append(entry)
        self._dump("epoch_metrics.json", self.history["epoch_metrics"])

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------

    def plot_losses(self):
        if not self.history["epoch_metrics"]:
            return
        epochs = [e["epoch"] for e in self.history["epoch_metrics"]]
        totals = [e["total_loss"] for e in self.history["epoch_metrics"]]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, totals, label="Total", linewidth=2)
        for model_name in self.history["epoch_metrics"][0]["individual_losses"]:
            plt.plot(
                epochs,
                [
                    e["individual_losses"][model_name]
                    for e in self.history["epoch_metrics"]
                ],
                label=model_name,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "loss_curve.png"))
        plt.close()

    # ------------------------------------------------------------------
    # finalise
    # ------------------------------------------------------------------

    def finalize(self):
        self.history["end_time"] = datetime.utcnow().isoformat(timespec="seconds")
        runtime = datetime.utcnow() - self.start_time
        self.history["runtime_h"] = round(runtime.total_seconds() / 3600, 4)
        self._dump("manifest.yaml", self.history)

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _dump(self, fname: str, obj: Any):
        """Write *obj* into *fname* inside the run folder."""
        path = os.path.join(self.log_dir, fname)
        if fname.endswith(".json"):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, indent=4, default=_json_default)
        else:
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(
                    _to_yaml_safe(obj), fh, sort_keys=False, allow_unicode=True
                )

    @staticmethod
    def _set_seed(seed: int | None):
        seed = seed if seed is not None else random.randrange(1_000_000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return seed

    def _log_system_info(self):
        sys_info = {
            "python": sys.version.split()[0],
            "pytorch": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": str(torch.version.cuda),
            "cudnn_version": str(torch.backends.cudnn.version()),
            "device_count": torch.cuda.device_count(),
            "hostname": socket.gethostname(),
            "os": platform.platform(),
        }
        try:
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            sys_info["git_commit"] = commit
        except Exception:
            sys_info["git_commit"] = None
        self.history["system"] = sys_info
        self._dump("system.yaml", sys_info)
