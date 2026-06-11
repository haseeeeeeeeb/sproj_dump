import os
import torch
from domainbed.algorithms import DANN, CORAL, Mixup, MMD, IRM, ERM, SagNet
from overcomplete import TopKSAE
algo_classes = {"DANN": DANN, "CORAL": CORAL, "Mixup": Mixup, "MMD": MMD, "IRM": IRM, "ERM": ERM, "SagNet": SagNet}
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
from domainbed.networks import Identity

def load_backbone(name, path):    
    algo_name, _, _ = name.split("_")   
    checkpoint = torch.load(path)

    ModelClass = algo_classes[algo_name]

    backbone = ModelClass(
        input_shape=checkpoint["model_input_shape"],
        hparams=checkpoint["model_hparams"],
        num_domains=checkpoint["model_num_domains"],
        num_classes=checkpoint["model_num_classes"]
    )

    if backbone.featurizer.__class__.__name__ == "ResNet":
        del backbone.featurizer.network.global_pool
        backbone.featurizer.network.global_pool = Identity()

    
    backbone.load_state_dict(checkpoint["model_dict"])


    return backbone



def load_sae(path):
    sae = torch.load(path, weights_only=False)
    return sae
