# the validation function. This is the only thing here

# imports:
from torch import nn
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score

# validate:
def validation(model, val_loader, device, weighted_class, permute_orientation, if_add_t1=False): #
    class_loss = nn.BCEWithLogitsLoss(weight=weighted_class)
    valid_loss = 0
    y_pred_list = []
    y_list = []
    with torch.no_grad():
        for batch_num in val_loader: # maybe add in batch? need to fix because of torchIO?
            # batch is a dict of torchio.Tensors
            img_batch = batch_num['t2'].permute(*permute_orientation)  # shape [B, 1, D, H, W]
            seg_batch = batch_num['t2_seg'].permute(*permute_orientation)  # shape [B, 1, D, H, W]
            y_batch = batch_num['class_label']  # shape [B, num_classes]
            img, segment, y = img_batch.to(device), seg_batch.to(device), y_batch.to(device)
            if if_add_t1:
                t1_img_batch = batch_num['t1'].permute(
                    *permute_orientation)  # shape [B, 1, D, H, W] for nnUnet, [B, 1, W, D, H] for MAE
                t1 = t1_img_batch.to(device)
                y_pred, _ = model(t1, img, segment)
            else:
                y_pred, _ = model(img, segment)
            valid_loss += class_loss(y_pred, y).item()
            y_pred_numpy, y_numpy = y_pred.detach().cpu().numpy(), y.detach().cpu().numpy()

            y_pred_list.append(y_pred_numpy)
            y_list.append(y_numpy)

    y_true = np.concatenate(y_list, axis=0)
    y_score = np.concatenate(y_pred_list, axis=0)

    score = average_precision_score(y_true, y_score, average='weighted') # Macro = average over classes, micro = for balanced or weighted classes

    return valid_loss/len(val_loader), score


