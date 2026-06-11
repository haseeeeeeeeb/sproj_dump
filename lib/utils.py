import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(backbone, images):
    
    with torch.no_grad():
        if hasattr(backbone, 'featurizer'): # for models trained using domainbed
            if backbone.featurizer.__class__.__name__ == "DinoV2" :
                activations = backbone.featurizer.network.forward_features(images.to(device))['x_norm_patchtokens']
            elif backbone.featurizer.__class__.__name__ == "ViT":
                activations = backbone.featurizer.network.forward_features(images.to(device))[:, 1:, :]
            else:
                activations = backbone.network[0](images.to(device))

        if hasattr(backbone, 'forward_features'): # for models directly from the overcomplete library
            activations = backbone.forward_features(images.to(device))

    return activations

