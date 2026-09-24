# every single function to prepare the data for training and training process

# imports:
import SimpleITK as sitk
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset
from scipy.ndimage import binary_dilation
import os, random
import warnings

# training transformation
def img_tranformations(spacing, label_map, dilation, if_add_t1): #for training
    if if_add_t1:
        image_keys = ('t2', 't1')
    else:
        image_keys = ('t2')

    tranformations = tio.Compose([
    # resample:

    ResampleToSpacingWithSegMatch(spacing=spacing, image_keys=image_keys, seg_key='t2_seg'),

    # remove extra labels:
    ChangeLabels(label_map),

    MultiLabelDilation(iterations=dilation, include_background=False),

    # crop out the background:
    tio.CropOrPad(target_shape=None, mask_name='t2_seg'),

    # 4) Random gamma contrast (approx nnU-Net’s gamma)
    tio.RandomGamma(log_gamma=(-0.3, 0.3)),

    # 5) Add Gaussian noise
    tio.RandomNoise(mean=0.0, std=(0, 0.1), p=0.15),

    # 6) (Optionally) Bias field
    tio.RandomBiasField(coefficients=0.3),

    # 7) Z-score normalization
    tio.ZNormalization(),
    ])
    return tranformations

# test transformation
def test_tranformations(spacing, label_map, dilation, if_add_t1): #
    if if_add_t1:
        image_keys = ('t2', 't1')
    else:
        image_keys = ('t2')
    tranformations = tio.Compose([
    # resample:
    ResampleToSpacingWithSegMatch(spacing=spacing, image_keys=image_keys, seg_key='t2_seg'),

    # remove extra labels:
    ChangeLabels(label_map),

    MultiLabelDilation(iterations=dilation, include_background=False),

    # crop out the background:
    tio.CropOrPad(target_shape=None, mask_name='t2_seg'),

    # 7) Z-score normalization
    tio.ZNormalization(),
    ])
    return tranformations

# helper augmentation 1
class ResampleToSpacingWithSegMatch(tio.Transform):
    def __init__(self, spacing=(1.0, 1.0, 1.0), image_keys=('t2'), seg_key='t2_seg', seg_match_image_key='t2'):
        super().__init__()
        if isinstance(image_keys, (str, bytes)):
            image_keys = [image_keys]
        self.spacing = spacing
        self.image_keys = list(image_keys)
        self.seg_key = seg_key
        self.seg_match_image_key = seg_match_image_key or (self.image_keys[0] if len(self.image_keys) > 0 else None)

    def apply_transform(self, subject):
        # Resample the T2 image:
        ref_img = subject[self.seg_match_image_key]
        ref_img_res = tio.Resample(self.spacing)(ref_img)
        subject[self.seg_match_image_key] = ref_img_res
        # Resample other images like T1:
        for key in self.image_keys:
            if key not in subject:
                warnings.warn(f"Image key '{key}' not found in subject — skipping.")
                continue
            if key == self.seg_match_image_key:
                continue
            subject[key] = tio.Resample(target=ref_img_res)(subject[key])

        # Resample the segmentation to match the image
        segmentation = subject[self.seg_key]
        target_image = subject[self.seg_match_image_key]
        resampled_seg = tio.Resample(target=target_image, label_interpolation='nearest')(segmentation)
        if resampled_seg.shape[1:] != target_image.shape[1:]:  # [C, H, W, D]
            # Fallback to resize to match image shape
            resized_seg = tio.Resize(target_image.shape[1:], label_interpolation='nearest')(resampled_seg)
            # Force integer labels
            resized_seg_fixed = resized_seg.data.round().clamp_min(0).long()
            # seg.set_data(seg.data.round().clamp_min(0).long())
            subject[self.seg_key] = resized_seg_fixed
            # subject[self.seg_key] = resized_seg
        else:
            subject[self.seg_key] = resampled_seg

        return subject
# helper augmentation 2
class MultiLabelDilation(tio.Transform):
    def __init__(self, iterations=1, include_background=False):
        super().__init__()
        self.iterations = iterations
        self.include_background = include_background

    def apply_transform(self, subject):
        for name, image in subject.get_images_dict().items():
            if isinstance(image, tio.LabelMap):
                data = image.data.clone()
                # print((data>0).sum().item())
                dilated = torch.zeros_like(data)

                labels = torch.unique(data)
                # print(labels)
                if not self.include_background:
                    labels = labels[labels != 0]

                for lbl in labels:
                    mask = (data == lbl).cpu().numpy()
                    mask_dilated = binary_dilation(mask, iterations=self.iterations)
                    # dilated[0][torch.from_numpy(mask_dilated)] = lbl
                    add_region = mask_dilated & (dilated.cpu().numpy() == 0)
                    dilated[torch.from_numpy(add_region)] = int(lbl.item())
                image.set_data(dilated)
        return subject
# helper augmentation 3
class ChangeLabels(tio.Transform):
    """TorchIO transform to change labels in a LabelMap only."""

    def __init__(self, label_map, inplace=False):
        """
        Args:
            label_map (dict): mapping {old_label: new_label}, e.g., {2:0, 3:1}.
            inplace (bool): whether to modify the segmentation tensor in place.
        """
        super().__init__()
        self.label_map = label_map
        self.inplace = inplace

    def apply_transform(self, subject):
        # Only modify the 'seg' LabelMap
        if 't2_seg' not in subject:
            raise ValueError("Subject has no 'seg' key.")

        seg_tensor = subject['t2_seg'].data
        if not self.inplace:
            seg_tensor = seg_tensor.clone()

        for old_label, new_label in self.label_map.items():
            seg_tensor[seg_tensor == old_label] = new_label

        # subject.t2_seg.set_data(seg_tensor)
        subject['t2_seg'].set_data(seg_tensor)
        return subject

# prepare the images for the dataloader
def make_subjects(ids, labels, dir, T1_dir, if_add_t1 = False, if_T1_only = False):
    """
    Build a list of tio.Subjects from your ID/label lists.
    """
    subjects = []
    root = Path(dir)
    t1_root = Path(T1_dir)

    ids = np.atleast_1d(ids).tolist()
    labels = np.atleast_1d(labels).tolist()

    for pid, label in zip(ids, labels):
        short = str(pid[1:])  # drop the leading 'S'

        # 1) find image file
        img_folder = root / short
        img_files  = sorted(img_folder.glob('*.nii*'))
        if not img_files:
            print(f"[WARN] No image for {pid} in {img_folder}")
            continue
        img_path = str(img_files[0])

        # 2) find seg file
        seg_folder = root / short / 'seg'
        seg_files  = sorted(seg_folder.glob('*.nii*'))
        if not seg_files:
            print(f"[WARN] No seg for {pid} in {seg_folder}")
            continue
        seg_path = str(seg_files[0])

        # find T1 image file:
        if if_add_t1 or if_T1_only:
            T1_folder = t1_root/short
            t1_files = list(T1_folder.rglob("2.nii.gz"))
            if not t1_files: # if there isn't any 2.nii.gz file
                t1_files = list(T1_folder.rglob("0.nii.gz"))
            # print(short)
            t1_path = str(t1_files[0])

        # 3) build a Subject
        if if_add_t1:
            subject = tio.Subject(
                t2 = tio.ScalarImage(img_path),
                t2_seg = tio.LabelMap(seg_path),
                class_label = torch.tensor(label),
                ids = pid,
                t1 = tio.ScalarImage(t1_path),
            )
        elif if_T1_only:
            subject = tio.Subject(
                t2=tio.ScalarImage(t1_path),  # going to train just like T2 images
                t2_seg=tio.LabelMap(seg_path),
                class_label=torch.tensor(label),
                ids=pid,
            )
        else:
            subject = tio.Subject(
                t2=tio.ScalarImage(img_path),
                t2_seg=tio.LabelMap(seg_path),
                class_label=torch.tensor(label),
                ids=pid,
            )

        subjects.append(subject)
    return subjects


def pad_to_batch_max_collate(batch, ref_key='t2', multiples=(128, 128, 16)): # this is so that each image/segmentation matches the nnUNet in sizes
    """
    Pad/crop each Subject in the batch to:
      H -> next multiple of multiples[0]
      W -> next multiple of multiples[1]
      D -> next multiple of multiples[2]
    Then stack into a plain dict of tensors.
    """
    Hy, Wx, Dz = multiples

    # 1) current shapes from the reference image in each Subject
    shapes = [sub[ref_key].spatial_shape for sub in batch]  # each (D,H,W)
    # print(shapes)
    H_max = max(s[0] for s in shapes)
    W_max = max(s[1] for s in shapes)
    D_max = max(s[2] for s in shapes)

    roundup = lambda v, m: int(math.ceil(v / m) * m)
    H_t = roundup(H_max, Hy)
    W_t = roundup(W_max, Wx)
    D_t = roundup(D_max, Dz)
    # print(H_t,W_t,D_t)

    # 2) harmonize shapes for ALL image keys in each Subject
    crop_pad = tio.CropOrPad((H_t, W_t, D_t))
    padded = [crop_pad(sub) for sub in batch]

    # 3) sanity-check shapes after padding (helps catch surprises fast)
    for i, sub in enumerate(padded):
        for k, v in sub.items():
            if isinstance(v, tio.Image):
                assert v.spatial_shape == (H_t, W_t, D_t), \
                    f"{k} item {i} has {v.spatial_shape}, expected {(H_t,W_t,D_t)}"

    # 4) build batch dict (images -> [B,C,D,H,W], labels -> stacked, metadata -> list)
    out = {}
    for k in padded[0].keys():
        v0 = padded[0][k]
        if isinstance(v0, tio.Image):
            out[k] = torch.stack([sub[k].data for sub in padded], dim=0)
        elif torch.is_tensor(v0):
            out[k] = torch.stack([sub[k] for sub in padded], dim=0)
        else:
            out[k] = [sub[k] for sub in padded]
    return out

def seed_all(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass

def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

def make_optimizer(model_name, lr, model):
    if 't1_t2_classification' in model_name:
        param_groups = [
            {"params": model.nnunet_feature_maps1.parameters(), "lr": lr},
            {"params": model.nnunet_feature_maps2.parameters(), "lr": lr},
            {"params": model.GEM_head.parameters(), "lr": lr}
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": lr}]

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer

def plot_and_save_losses(train_losses, valid_losses=None, out_path="loss_curve.png"):
    """
    Plots training (and optional validation) loss curves and writes to `out_path`.
    """
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss")
    if valid_losses is not None:
        epoch_axis = np.arange(0,len(train_losses), len(train_losses)/len(valid_losses))
        plt.plot(epoch_axis, valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
