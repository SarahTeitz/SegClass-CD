# testing+metrics

# imports
import os
import numpy as np
from models import GEM_multiple_classification, GEM_multiple_T1_T2_classification
import argparse
import pickle
import torch
import torchio as tio
from utilities import make_subjects, test_tranformations, pad_to_batch_max_collate
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix, cohen_kappa_score, average_precision_score
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='', help='experiment name_name') # used for snapshot
parser.add_argument('--model', type=str, default='small_intestine_model', help='model_name')#  use this to determine if using multi-contrast or single contrast model
parser.add_argument('--frozen_decoder', type=bool, default=False, help= 'is the decoder frozen or not')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--dilation', type=int, default=3, help= 'segmentation dilation')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size of data per gpu')
parser.add_argument('--spacing', type=float, default= [1,1,5.5], nargs="+", help= '[H, W, D]')
parser.add_argument('--gpu', type=str, default='2', help='GPU to use')
parser.add_argument('--T1_only', type=bool, default=False, help='if training single contrast with T1')
parser.add_argument('--modality_type', type=str, default='attention', help='attention, non, linear or learned')
args = parser.parse_args()

# paths:
img_dir = 'T2_images_path' #specifically with my dataset - the T2 images are together with the T2 segments. See make_subjects in utilities
T1_dir = 'T1_images_path'
seg_dir = 'segmenation_path' # extracted prior to training the classification model
pretrain_nnunet_path = "T2_path_to_nnUNetTrainer__nnUNetPlans__3d_fullers"
pretrain_t1_nnunet_path = "T1_path_to_nnUNetTrainer__nnUNetPlans__3d_fullers"

# load data split and paths
if args.labelnum == 1:
    with open("/argusdata4/sarah.teitz/data/classification/ileum_thick_splits_5folds.pkl", 'rb') as f:
        splits = pickle.load(f)
elif args.labelnum == 8:
    with open("/argusdata4/sarah.teitz/data/classification/t1_t2_splits_5folds.pkl", 'rb') as f:
        splits = pickle.load(f)


# snapshot:
snapshot_path = "snapshot_path_here"

# parameters:
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.gpu.lower() == 'cpu' or args.gpu == '-1':
    device = torch.device('cpu')
else:
    # set BEFORE any CUDA calls (e.g., is_available) so indexing remaps
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # first visible GPU is index 0 from here on
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters:
batch_size = args.batch_size
num_classes = args.labelnum
model_name = args.model
frozen_decoder = args.frozen_decoder
dilation = args.dilation
spacing = args.spacing  # H, W, D
if_add_t1 = False
label_map = {1:0, 3:0, 4:0, 5:0, 6:0, 7:2}
learn_disp = True
permute_orientation =  (0,1,4,2,3)
in_channels =  800 if model_name=='GEM_multiple_classification' else 1600 #if model_name=='t1_t2_classification'
if_T1_only = args.T1_only
modality_type = args.modality_type


def youden_thresholds(y_true_bin: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """
    Pick per-label thresholds that maximize Youden's J = TPR - FPR (equiv to TPR + TNR - 1).

    y_true_bin: (N, C) in {0,1}
    y_score:    (N, C) probabilities/scores (higher => more positive)

    Returns:
      thresholds: (C,) float thresholds (one per label)
    """
    y_true_bin = (y_true_bin > 0).astype(int)
    y_true_bin = np.atleast_2d(y_true_bin)
    y_score = np.atleast_2d(y_score)

    # fix rare (1, N) shape
    if y_true_bin.shape[0] == 1 and y_score.shape[0] > 1 and y_true_bin.shape[1] == y_score.shape[0]:
        y_true_bin = y_true_bin.T
        y_score = y_score.T

    N, C = y_true_bin.shape
    thrs = np.full(C, 0.5, dtype=float)

    for c in range(C):
        yt = y_true_bin[:, c]
        ys = y_score[:, c]

        # If only one class present in this fold for this label, ROC is undefined.
        # Fall back to 0.5 (or you can choose something else).
        if np.unique(yt).size < 2:
            thrs[c] = 0.5
            continue

        fpr, tpr, thresholds = roc_curve(yt, ys)
        J = tpr - fpr
        best_idx = int(np.argmax(J))
        thrs[c] = float(thresholds[best_idx])

    return thrs

def per_label_metrics(y_true, y_pred, class_names=None, threshold=None):
    """
    y_true: (N, 9) ints/bools (0/1)
    y_pred: (N, 9) ints/bools or probabilities in [0,1]
    threshold: if y_pred has floats, use this (default 0.5)
    """
    ##################
    y_true_bin = (y_true > 0).astype(int)
    y_true_bin = np.atleast_2d(y_true_bin)
    y_pred = np.atleast_2d(y_pred)

    # handle rare (1, N) -> (N, 1)
    if y_true_bin.ndim == 2 and y_true_bin.shape[0] == 1 and y_pred.shape[0] > 1:
        y_true_bin = y_true_bin.T
        y_pred = y_pred.T

    N, C = y_true_bin.shape

    # names
    if class_names is None:
        names = [f"class_{i}" for i in range(C)]
    else:
        names = list(class_names)[:C]

    # make thresholds vector
    if threshold is None:
        if y_pred.dtype.kind in "fc":
            thr_vec = np.full(C, 0.5, dtype=float)
            y_pred_bin = (y_pred >= 0.5).astype(int)
        else:
            thr_vec = np.full(C, np.nan, dtype=float)  # not meaningful
            y_pred_bin = (y_pred > 0).astype(int)
    else:
        thr_vec = np.array(threshold, dtype=float)
        if thr_vec.ndim == 0:
            thr_vec = np.full(C, float(thr_vec), dtype=float)
        if thr_vec.shape[0] != C:
            raise ValueError(f"threshold must be scalar or shape (C,), got {thr_vec.shape}, C={C}")
        y_pred_bin = (y_pred >= thr_vec[None, :]).astype(int)

        #################

    # per-label metrics
    p, r, f1, supp = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average=None, zero_division=0
    )

    # confusion-derived NPV
    mcm = multilabel_confusion_matrix(y_true_bin, y_pred_bin)
    tn, fp, fn, tp = mcm[:, 0, 0], mcm[:, 0, 1], mcm[:, 1, 0], mcm[:, 1, 1]
    npv = tn / np.where(tn + fn == 0, 1, tn + fn)

    # Cohen's kappa per label + macro
    kappas = np.array([cohen_kappa_score(y_true_bin[:, i], y_pred_bin[:, i]) for i in range(C)])
    kappa_macro = float(np.mean(kappas))

    # per-label AUPRC (works with C=1; ensure 2D inputs)
    auprc = average_precision_score(y_true_bin, y_pred, average=None)
    print(auprc)
    auprc = np.atleast_1d(auprc)

    # --- keep length-1 vectors as length-1 (avoid scalars) ---
    p, r, f1, supp, npv, kappas, auprc = map(np.atleast_1d, (p, r, f1, supp, npv, kappas, auprc))

    # if user provided fewer names (e.g., 1) but C>len(names), slice metrics to match names
    L = len(names)
    p, r, f1, supp, npv, kappas, auprc = [arr[:L] for arr in (p, r, f1, supp, npv, kappas, auprc)]

    # --- micro metrics ---
    micro_ppv, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true_bin.ravel(),
        y_pred_bin.ravel(),
        average="binary",
        zero_division=0
    )

    micro_auprc = average_precision_score(
        y_true_bin.ravel(),
        y_pred.ravel()
    )

    micro_kappa = cohen_kappa_score(
        y_true_bin.ravel(),
        y_pred_bin.ravel()
    )
    df_micro = pd.DataFrame([{
        "kappa_macro": float(kappa_macro),
        "micro_auprc": float(micro_auprc),
        "micro_PPV": float(micro_ppv),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "micro_cohen_kappa": float(micro_kappa),
    }])
    # return df_micro, {"kappa_macro": kappa_macro}
    #-------------------

    # build dataframe WITHOUT setting index (avoids length-mismatch headaches)
    df = pd.DataFrame(
        {
            "class": names,
            "support": supp,
            "precision(PPV)": p,
            "recall(sensitivity)": r,
            "NPV": npv,
            "cohen_kappa": kappas,
            "auprc": auprc,
            "f1": f1,
            "threshold": threshold
        }
    )

    return df, df_micro,{"kappa_macro": kappa_macro}


if __name__ == "__main__":
    for fold_num, fold in enumerate(splits["folds"]):

        fold_path = os.path.join(snapshot_path, f'fold_{fold_num}')

        # load best model:
        best_model_path = os.path.join(fold_path, '{}_best_model.pth'.format(args.model))

        # model:

        if model_name == 'GEM_multiple_classification':
            if if_T1_only:
                model = GEM_multiple_classification(pretrain_t1_nnunet_path, frozen_decoder, in_channels,
                                                    num_classes).to(device)
            else:
                model = GEM_multiple_classification(pretrain_nnunet_path, frozen_decoder, in_channels, num_classes).to(
                    device)
            collate_multiples = (128, 128, 16)

        elif model_name == 't1_t2_classification':
            model = GEM_multiple_T1_T2_classification(pretrain_nnunet_path,pretrain_t1_nnunet_path, frozen_decoder, in_channels, num_classes,modality_type).to(device)
            collate_multiples = (128, 128, 16)
            if_add_t1 = True



        model.load_state_dict(torch.load(best_model_path), strict=False) # add in result and: print("Missing:", result.missing_keys) print("Unexpected:", result.unexpected_keys)

        model.eval()


        #bring the data + transformations:
        transform = test_tranformations(
            spacing=spacing, label_map=label_map, dilation=dilation, if_add_t1=if_add_t1)
        test_ids, test_labels = fold['val']
        test_subjects = make_subjects(test_ids, test_labels, seg_dir, T1_dir, if_add_t1, if_T1_only)
        test_data = tio.SubjectsDataset(test_subjects, transform=transform)
        test_loader = tio.SubjectsLoader(test_data, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False,
                                        collate_fn=lambda b: pad_to_batch_max_collate(b, ref_key='t2',
                                                                                      multiples=collate_multiples))


        # cuda everything and run:
        y_pred_list = []
        y_list = []
        for batch_num in test_loader:  # maybe add in batch? need to fix because of torchIO?
            # batch is a dict of torchio.Tensors
            img_batch = batch_num['t2'].permute(*permute_orientation)  # shape [B, 1, D, H, W]
            seg_batch = batch_num['t2_seg'].permute(*permute_orientation)  # shape [B, 1, D, H, W]
            y_batch = batch_num['class_label']  # shape [B, num_classes]
            # print(img_batch.dtype)

            img, segment, y = img_batch.to(device), seg_batch.to(device), y_batch.to(device)
            if if_add_t1:
                t1_img_batch = batch_num['t1'].permute(*permute_orientation)  # shape [B, 1, D, H, W] for nnUnet, [B, 1, W, D, H] for MAE
                t1 = t1_img_batch.to(device)
                y_pred_logits, _ = model(t1, img, segment)
            else:
                y_pred_logits, _ = model(img, segment)
            y_pred = torch.sigmoid(y_pred_logits)
            y_pred_numpy, y_numpy = y_pred.detach().cpu().numpy(), y.detach().cpu().numpy()
            y_pred_list.append(y_pred_numpy)
            y_list.append(y_numpy)

        y_true = np.concatenate(y_list, axis=0)
        y_score = np.concatenate(y_pred_list, axis=0)

        print(y_score.shape, y_true.shape)

        # get metrics:
        threshold = youden_thresholds(y_true, y_score)
        if num_classes == 8:
            class_names = ['Ileum inflammation',
                           'Ileum fistula',
                           'Ileum mesenteric edema OR fat stranding',
                           'Ileum stenosis',
                           'Ileum comb sign',
                           'Ileum wall thickness',
                           'Ileum wall enhancement',
                           'Ileum pre stenotic dil'
                           ]
        elif num_classes == 1:
            class_names = ['Ileum wall thickness']
        df, df_micro,  kappa_macro = per_label_metrics(y_true, y_score, class_names=class_names, threshold=threshold)

        print(f'kappa_macro: {kappa_macro}')
        df.to_csv(f"{snapshot_path}/fold_{fold_num}/metrics1.csv", index=False) #add this in!!!!!
        df_micro.to_csv(f"{snapshot_path}/fold_{fold_num}/micro1.csv", index=False)  # add this in!!!!!

    # get the mean and std for all the folds:
    excel_name = "metrics.csv"  # same name in each fold
    class_col = "class"  # adjust if your column name differs
    out_name = "summary_mean_std.csv"

    snapshot_path = Path(snapshot_path)
    files = sorted(snapshot_path.glob("fold_*/metrics.csv"))
    files_micro = sorted(snapshot_path.glob("fold_*/micro1.csv"))
    print("Found files:")
    for f in files:
        print(" ", f)

    if not files:
        raise FileNotFoundError(f"No files found matching */{excel_name} under {snapshot_path}")

    # ---- 1) Read each fold file ----
    dfs = []
    fold_names = []

    for f in files:
        fold_name = f.parent.name
        fold_names.append(fold_name)

        df = pd.read_csv(f)  # add sheet_name=... if needed
        if class_col not in df.columns:
            raise ValueError(f"'{class_col}' column not found in {f}. Columns: {list(df.columns)}")

        df = df.set_index(class_col)

        # Keep only numeric metric columns (avoid issues with text columns)
        df = df.select_dtypes(include=[np.number])

        dfs.append(df)

    # ---- 2) Stack folds: (fold, class) index ----
    all_folds = pd.concat(dfs, keys=fold_names, names=["fold", "class"])

    # ---- 3) Compute mean and std over folds for each class ----
    mean_df = all_folds.groupby("class", sort=False).mean(numeric_only=True)
    std_df = all_folds.groupby("class", sort=False).std(ddof=1, numeric_only=True)  # sample std

    # ---- 4) Build "mean ± std" table (strings) ----
    combined = mean_df.copy()
    for col in combined.columns:
        combined[col] = (
                mean_df[col].round(3).astype(str)
                + " ± "
                + std_df[col].round(3).astype(str)
        )

    combined = combined.reset_index()  # bring class back as a column

    # ---- 5) Export to Excel ----
    out_path = snapshot_path / out_name
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {out_path}")

    for f in files_micro:
        print(" ", f)

    if not files_micro:
        raise FileNotFoundError(f"No files found matching */{excel_name} under {snapshot_path}")

    df_micro = []
    for f in files_micro:
        fold_name = f.parent.name

        df = pd.read_csv(f)

        # if the file has exactly one row of micro metrics
        df["fold"] = fold_name
        df_micro.append(df)

    all_micro = pd.concat(df_micro, ignore_index=True)

    # keep only numeric metric columns
    metric_cols = all_micro.select_dtypes(include=[np.number]).columns

    mean_s = all_micro[metric_cols].mean()
    std_s = all_micro[metric_cols].std(ddof=1)

    combined_micro = pd.DataFrame({
        "metric": metric_cols,
        "mean": mean_s.values,
        "std": std_s.values,
        "mean ± std": [
            f"{mean_s[col]:.3f} ± {std_s[col]:.3f}"
            for col in metric_cols
        ]
    })

    out_path = snapshot_path / "combined_micro_metrics.csv"
    combined_micro.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_path}")
