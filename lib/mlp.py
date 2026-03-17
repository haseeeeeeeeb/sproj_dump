from lib.data_handlers import Load_ImageNet100
import torch
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def remap_predictions(logits, internal_map=None):

    ## Logits should be Logits in the form B x 1000
    assert logits.shape[1] == 1000

    ## Get the internal class to index mapping
    if internal_map is None:
        _, image_dataset = Load_ImageNet100(transform=None, batch_size=256, shuffle=True, dataset=True)
        internal_map = image_dataset.class_to_idx


    ## Get the imagenet idx to class map
    import json
    with open("./datasets/imagenet-class-index-json/imagenet_class_index.json") as f:
        imagenet_idx_map = json.load(f)
    

    ## Extract the WNID labels from output logits
    predicted = torch.argmax(logits, dim=1)
    predicted_classes = [imagenet_idx_map[str(idx)][0] for idx in predicted.tolist()]

    ## Remap them to the internal index
    updated_idxs = [internal_map.get(label, -1) for label in predicted_classes] # Map to -1 if the label does not exist

    return torch.tensor(updated_idxs)



def train_mlp(image_loader, mlp, optimizer, loss_fn, vit_full, sae, internal_map, alpha=1.0, epoch=1):
    mlp.to(device)
    vit_full.to(device)
    sae.to(device)
    
    
    epoch_loss = 0.0
    pbar = tqdm(image_loader, desc=f"Epoch {epoch}", unit="batch")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        vit_full.eval()
        with torch.no_grad():
            # Get latent features & ViT predictions
            latent = vit_full.forward_features(images)
            predicted = vit_full.forward_head(latent) 

            # Remap predictions to dataset indices
            predicted = remap_predictions(predicted, internal_map=internal_map)
            predicted = predicted.to(device)  # ensure predictions on device

            #print(predicted.shape)

            latent = latent[:, 1:]  # remove CLS token
            
            ## Filter for invalid Predictions
            valid_mask = predicted != -1
            if valid_mask.any():
                predicted = predicted[valid_mask]
                latent = latent[valid_mask, :, :]

            else:
                continue


            # SAE encode
            latent = rearrange(latent, 'n t d -> (n t) d')
            _, z = sae.encode(latent)
            z = z.to(device)
            z = rearrange(z, '(n t) d -> n t d', t=196)
            z = z.sum(dim=1, keepdim=True)
            z = z.flatten(start_dim=1)
            #print("Latent Output Shape: ", z.shape)

        mlp.train()
        optimizer.zero_grad()

        y_hat = mlp(z)
        
        l1_norm = sum(param.abs().sum() for name, param in mlp.named_parameters() if "weight" in name)
        loss = loss_fn(y_hat, predicted) + alpha*l1_norm

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"Epoch {epoch} finished. Average loss: {epoch_loss / len(image_loader):.4f}")
    return epoch_loss




def test_mlp(image_loader, mlp, loss_fn, vit_full, sae, internal_map):
    mlp.to(device)
    vit_full.to(device)
    sae.to(device)
    mlp.eval()
    vit_full.eval()
    sae.eval()
    
    all_softmax = []
    all_entropy = []

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(image_loader, desc="Testing", unit="batch")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Extract ViT features + predictions
            latent = vit_full.forward_features(images)
            predicted = vit_full.forward_head(latent)
            predicted = remap_predictions(predicted, internal_map=internal_map)
            predicted = predicted.to(device)

            latent = latent[:, 1:]  # remove CLS token
            
            # Filter invalid preds
            valid_mask = predicted != -1
            predicted = predicted[valid_mask]
            latent = latent[valid_mask, :, :]

            if predicted.numel() == 0:  # skip batch if all invalid
                continue

            # SAE encode
            latent = rearrange(latent, 'n t d -> (n t) d')
            _, z = sae.encode(latent)
            z = z.to(device)
            z = rearrange(z, '(n t) d -> n t d', t=196)
            z = z.sum(dim=1, keepdim=True)
            z = z.flatten(start_dim=1)

            # MLP forward
            y_hat = mlp(z)

            probs = torch.nn.functional.softmax(y_hat, dim=1)
            all_softmax.append(probs.cpu())
            entropy = -(probs * probs.log()).sum(dim=1)
            all_entropy.append(entropy.cpu())

            # Loss
            loss = loss_fn(y_hat, predicted)
            total_loss += loss.item()

            # Accuracy
            preds = torch.argmax(y_hat, dim=1)
            correct += (preds == predicted).sum().item()
            total += predicted.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(image_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"Test finished. Avg loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy, torch.cat(all_softmax, dim=0), torch.cat(all_entropy, dim=0)





def prune_weights(model, k=0.5):    
    W = model.weight.data  
    absW = W.abs()

    threshold = k * absW.std()
    
    sparse_W = W.clone()
    sparse_W[absW < threshold] = 0.0

    model.weight.data = sparse_W

    return model



def check_sparsity(model):
    sparse_W = model.weight.data
    num_zero = (sparse_W == 0).sum().item()
    num_total = sparse_W.numel()
    sparsity = 100.0 * num_zero / num_total
    return sparsity, num_zero


def plot_weight_histogram(model, bins=250):
    W = model.weight.data.cpu().numpy().flatten()

    plt.figure(figsize=(6, 4))
    plt.hist(W, bins=bins, edgecolor='black')
    plt.title("Histogram of MLP Weights")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_weight_histogram_before_after(original_model, pruned_model, bins=250):
    W_before = original_model.weight.data.cpu().numpy().flatten()
    W_after  = pruned_model.weight.data.cpu().numpy().flatten()
    total_weights = W_before.size

    plt.figure(figsize=(7, 5))
    plt.hist(W_before, bins=bins, alpha=0.5, label="Before Pruning", edgecolor='black')
    plt.hist(W_after,  bins=bins, alpha=0.5, label="After Pruning",  edgecolor='black')
    
    plt.title(f"Histogram of MLP Weights (Total Weights = {total_weights})")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()



def aggregator(sparse_mlps):

    # Matrices
    matrices = [w.weight.detach().cpu() for _, w in sparse_mlps.items()]

    # Aggregator
    aggregator = torch.stack(matrices, dim=0)
    
    print("D x C x F")
    print(aggregator.shape)


    aggregated = aggregator.sum(dim=0)

    print(aggregated.shape)
    print()




def train_mlp_PACS(image_loader, mlp, optimizer, loss_fn, model, sae, alpha=1.0, epoch=1):
    mlp.to(device)
    model.to(device)
    sae.to(device)
    
    
    epoch_loss = 0.0
    pbar = tqdm(image_loader, desc=f"Epoch {epoch}", unit="batch")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        model.eval()
        with torch.no_grad():
            # Get latent features & ViT predictions
            if hasattr(model, "forward_features"):
                latent = model.forward_features(images)
                predicted = model.forward_head(latent) 
            else:
                if model.featurizer.__class__.__name__ == "DinoV2":
                        latent = model.featurizer.network.forward_features(images.to(device))['x_norm_patchtokens']
                else:
                    pass

                predicted = model.featurizer(images)
                predicted = model.classifier(predicted)

            # SAE encode
            latent = rearrange(latent, 'n t d -> (n t) d')
            _, z = sae.encode(latent)
            z = z.to(device)
            z = rearrange(z, '(n t) d -> n t d', t=256)
            z = z.sum(dim=1, keepdim=True)
            z = z.flatten(start_dim=1)
            #print("Latent Output Shape: ", z.shape)

        mlp.train()
        optimizer.zero_grad()


        y_hat = mlp(z)
        
        l1_norm = sum(param.abs().sum() for name, param in mlp.named_parameters() if "weight" in name)
        loss = loss_fn(y_hat, predicted) + alpha*l1_norm

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"Epoch {epoch} finished. Average loss: {epoch_loss / len(image_loader):.4f}")
    return epoch_loss




def test_mlp_PACS(image_loader, mlp, loss_fn, vit_full, sae, internal_map):
    mlp.to(device)
    vit_full.to(device)
    sae.to(device)
    mlp.eval()
    vit_full.eval()
    sae.eval()
    
    all_softmax = []
    all_entropy = []

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(image_loader, desc="Testing", unit="batch")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Extract ViT features + predictions
            latent = vit_full.forward_features(images)
            predicted = vit_full.forward_head(latent)
            predicted = remap_predictions(predicted, internal_map=internal_map)
            predicted = predicted.to(device)

            latent = latent[:, 1:]  # remove CLS token
            
            # Filter invalid preds
            valid_mask = predicted != -1
            predicted = predicted[valid_mask]
            latent = latent[valid_mask, :, :]

            if predicted.numel() == 0:  # skip batch if all invalid
                continue

            # SAE encode
            latent = rearrange(latent, 'n t d -> (n t) d')
            _, z = sae.encode(latent)
            z = z.to(device)
            z = rearrange(z, '(n t) d -> n t d', t=196)
            z = z.sum(dim=1, keepdim=True)
            z = z.flatten(start_dim=1)

            # MLP forward
            y_hat = mlp(z)

            probs = torch.nn.functional.softmax(y_hat, dim=1)
            all_softmax.append(probs.cpu())
            entropy = -(probs * probs.log()).sum(dim=1)
            all_entropy.append(entropy.cpu())

            # Loss
            loss = loss_fn(y_hat, predicted)
            total_loss += loss.item()

            # Accuracy
            preds = torch.argmax(y_hat, dim=1)
            correct += (preds == predicted).sum().item()
            total += predicted.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(image_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"Test finished. Avg loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy, torch.cat(all_softmax, dim=0), torch.cat(all_entropy, dim=0)

