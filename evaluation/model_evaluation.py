from evaluation.metrics import compute_metrics_final
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from torch.cuda.amp import autocast, GradScaler



def model_evaluation(
        model,
        data_loader: DataLoader,
        criterion,
        device: torch.device
) -> dict:
    """Evaluates model using given data and loss criterion."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    running_loss = 0
    all_targets = []
    all_logits = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Validation", unit="batch")):
            labels = batch['label'].to(device).unsqueeze(1).float()
            # Move data to device
            batch["images"] = batch["images"].to(device)
            batch["sites"] = batch["sites"].to(device)
            batch["mask"] = batch["mask"].to(device)
            batch["label"] = batch["label"].to(device)
            
            # Forward pass
            scores, attention = model(batch["images"], batch["sites"], batch["mask"])
            labels = batch['label'].to(device).unsqueeze(1).float()
            all_targets.append(labels.detach())
            all_logits.append(scores.detach())

            loss = criterion(scores, labels)
            running_loss += loss.item()

    # Transfer to CPU and convert to numpy at the end
    all_targets = torch.cat(all_targets).cpu().numpy()
    all_logits = torch.cat(all_logits).cpu().numpy()

    epoch_metrics = compute_metrics_final(all_targets, all_logits)
    epoch_metrics['loss'] = running_loss / len(data_loader)
    print(epoch_metrics)
    return epoch_metrics, all_logits

def run_and_save_predictions(
        loader, 
        model, 
        device,
        criterion, 
        save_dir):
    
    pred_list = []
    pred_proba_list = []
    targets_list = []
    indices_list = []
    sites_list = []
    logits_list = []
    running_loss = 0
    #p_ids_list = [index for index in loader.dataset.indices]
    model.eval()
    with torch.no_grad():
        for batch in iter(loader):
            labels = batch['label'].to(device).unsqueeze(1).float()
            # Move data to device
            batch["images"] = batch["images"].to(device)
            batch["sites"] = batch["sites"].to(device)
            batch["mask"] = batch["mask"].to(device)
            batch["label"] = batch["label"].to(device)
            scores, _ = model(batch["images"], batch["sites"], batch["mask"])
            labels = batch['label'].to(device).unsqueeze(1).float()
            #masked_indices = batch["mask"].bool()
            #valid_scores = scores[masked_indices]
            #valid_labels = batch["label"][masked_indices]
            loss = criterion(scores, labels)
            running_loss += loss.item()
            try:
                indices_list.append(batch['ids'])
            except:
                indices_list.append(batch['id'])


            #valid_ids = batch['ids'][masked_indices]
            #valid_sites = batch['sites'][masked_indices]

            logits_list.append(scores.detach())

            probabilities = torch.sigmoid(scores).detach()
            pred_proba_list.append(probabilities)

            predictions = (probabilities > 0.5).float()
            pred_list.append(predictions)

            targets_list.append(labels.detach())
            sites_list.append(batch['sites'].detach())
            # indices_list.append(ids.detach())
            # sites_list.append(valid_sites.detach())

          
    epoch_metrics = compute_metrics_final(torch.cat(targets_list).cpu().numpy(), torch.cat(logits_list).cpu().numpy())
    epoch_metrics['loss'] = running_loss / len(loader)

    # Convert all lists of tensors to flat numpy arrays
    targets_array = torch.cat(targets_list).view(-1).cpu().numpy()
    probabilities_array = torch.cat(pred_proba_list).view(-1).cpu().numpy()
    predictions_array = torch.cat(pred_list).view(-1).cpu().numpy()
    try:
        indices_array = [item for sublist in indices_list for item in sublist]
    except:
        indices_array = torch.cat(indices_list).view(-1).cpu().numpy()

    try:
        sites_array = torch.cat(sites_list).view(-1).cpu().numpy()
    except:
        sites_array = []
    logits_array = torch.cat(logits_list).view(-1).cpu().numpy()


    # Prepare and save DataFrame
    try:
        df = pd.DataFrame({
        'indices': indices_array,
        'sites': sites_array,
        'targets': targets_array,
        'predictions': predictions_array,
        'predictions_proba': probabilities_array
    })
    except:
         df = pd.DataFrame({
        'indices': indices_array,
        #'sites': sites_array,
        'targets': targets_array,
        'predictions': predictions_array,
        'predictions_proba': probabilities_array
    })


    df.to_csv(save_dir, index=False)

    return epoch_metrics, logits_array




            


