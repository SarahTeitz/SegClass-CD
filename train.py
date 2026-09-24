# training and validation for 5-fold cross validation

# imports:
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import logging
import torch
from torch import nn
import pickle
from utilities import plot_and_save_losses, img_tranformations, make_subjects, pad_to_batch_max_collate, seed_all, seed_worker, test_tranformations, make_optimizer
from models import GEM_multiple_classification, GEM_multiple_T1_T2_classification
from validation import validation
import torchio as tio
import numpy as np

# args:
print("ARGS:", sys.argv) # sanity check for arguments!

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='', help='experiment name_name') # used for snapshot
parser.add_argument('--model', type=str, default='t1_t2_classification', help='model_name') #  use this to determine if using multi-contrast or single contrast model
parser.add_argument('--frozen_decoder', type=bool, default=False, help= 'is the decoder frozen or not')
parser.add_argument('--max_epochs', type=int, default=12, help='maximum epochs to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size of data per gpu')
parser.add_argument('--base_lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--accum_steps', type=int, default=8, help='how many micro-batches to accumulate for gradient accum')
parser.add_argument('--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('--dilation', type=int, default=3, help= 'segmentation dilation')
parser.add_argument('--spacing', type=float, default= [1,1,5.5], nargs="+", help= '[H, W, D]')
parser.add_argument('--T1_only', type=bool, default=False, help='if training single contrast with T1 only')
parser.add_argument('--modality_type', type=str, default='attention', help='attention, non, linear or learned')

args = parser.parse_args()

# seeds:
seed = args.seed
seed_all(seed)
g = torch.Generator().manual_seed(seed)

# snapshot:
snapshot_path = "snapshot_path_here"
# paths:
img_dir = 'T2_images_path' #specifically with my dataset - the T2 images are together with the T2 segments. See make_subjects in utilities
T1_dir = 'T1_images_path'
seg_dir = 'segmenation_path' # extracted prior to training the classification model
pretrain_nnunet_path = "T2_path_to_nnUNetTrainer__nnUNetPlans__3d_fullers"
pretrain_t1_nnunet_path = "T1_path_to_nnUNetTrainer__nnUNetPlans__3d_fullers"

# load data split and paths:
# pickles have the splits between the different folds for 5-fold cross-validation, has the "patient ID"  and the labels
if args.labelnum == 1:
    with open("/argusdata4/sarah.teitz/data/classification/ileum_thick_splits_5folds.pkl", 'rb') as f:
        splits = pickle.load(f)
elif args.labelnum == 8:
    with open("/argusdata4/sarah.teitz/data/classification/t1_t2_splits_5folds.pkl", 'rb') as f:
        splits = pickle.load(f)


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

# type of model:
model_name = args.model
frozen_decoder = args.frozen_decoder
max_epochs = args.max_epochs
base_lr = args.base_lr
batch_size = args.batch_size
num_classes = args.labelnum
accum_steps =args.accum_steps
dilation = args.dilation
spacing = args.spacing  # H, W, D
label_map = {1:0, 3:0, 4:0, 5:0, 6:0, 7:2} # my dataset has extra labels
weighted_class = torch.tensor(1).to(device)
pos_weight = torch.tensor(1.364).to(device) if num_classes==1 else torch.tensor([2.04,14.52,19.01,3.40,7.06,1.37,1.54,4.99]).to(device) # the else is for label_num==8
eps = 0.1
permute_orientation =  (0,1,4,2,3) #to fit the pretrained nnUNet
in_channels =  800 if model_name=='GEM_multiple_classification' else 1600 #if model_name=='t1_t2_classification'
num_workers = 4
if_add_t1 = False # default is T2 single contrast model, if_add_t1 allows us to have a dual contrast (don't touch this - the model type will define if it is positive or not)
if_T1_only = args.T1_only # because the default is T2
modality_type = args.modality_type

if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(sys.argv[0])
    writer = SummaryWriter(snapshot_path + '/log')

    # fold scores:
    fold_scores = []

    # folds:
    for fold_num, fold in enumerate(splits["folds"]):
        fold_path = os.path.join(snapshot_path, f'fold_{fold_num}')
        #make a folder for the fold:
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

        # model:
        if model_name == 'GEM_multiple_classification': # the single contrast model
            if if_T1_only: # T1 single contrast model
                model = GEM_multiple_classification(pretrain_t1_nnunet_path, frozen_decoder, in_channels, num_classes).to(device)
            else: #T2 single contrast model
                model = GEM_multiple_classification(pretrain_nnunet_path, frozen_decoder, in_channels, num_classes).to(device)
            collate_multiples = (128, 128, 16)
        elif model_name == 't1_t2_classification': # multi-contrast model
            model = GEM_multiple_T1_T2_classification(pretrain_nnunet_path, pretrain_t1_nnunet_path, frozen_decoder, in_channels, num_classes, modality_type).to(device)
            collate_multiples = (128, 128, 16)
            if_add_t1 = True

        # transforms:
        train_transform = img_tranformations(
            spacing=spacing, label_map=label_map, dilation=dilation, if_add_t1=if_add_t1)
        valid_transform = test_tranformations(
            spacing=spacing, label_map=label_map, dilation=dilation, if_add_t1=if_add_t1)

        # Data
        train_ids, train_labels = fold['train']
        val_ids, val_labels = fold['val']

        # Dataset:
        train_subjects = make_subjects(train_ids, train_labels, seg_dir, T1_dir, if_add_t1, if_T1_only)
        val_subjects = make_subjects(val_ids, val_labels, seg_dir, T1_dir, if_add_t1, if_T1_only)

        train_data = tio.SubjectsDataset(train_subjects, transform=train_transform)
        val_data = tio.SubjectsDataset(val_subjects, transform=valid_transform)

        # Dataloader:
        train_loader = tio.SubjectsLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True,shuffle=True,collate_fn=lambda b: pad_to_batch_max_collate(b, ref_key='t2', multiples=collate_multiples), generator=g, worker_init_fn=seed_worker)
        val_loader = tio.SubjectsLoader(val_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True,shuffle=False, collate_fn=lambda b: pad_to_batch_max_collate(b, ref_key='t2', multiples=collate_multiples), generator=g, worker_init_fn=seed_worker)

        # optimizer+schedular+loss:
        optimizer = make_optimizer(model_name, base_lr, model)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        class_loss = nn.BCEWithLogitsLoss(weight=weighted_class, pos_weight=pos_weight)

        iterator = tqdm(range(args.max_epochs), ncols=70)
        epoch_num = 0
        best_score = 0
        train_loss = []
        val_loss = []
        print('start training')
        for epoch_num in iterator:
            running_loss = 0
            scaler = torch.amp.GradScaler("cuda", enabled=False) # to avoid small gradients
            optimizer.zero_grad(set_to_none=True)

            for batch_num, batch in enumerate(train_loader):
                # print(batch_num)
                with torch.amp.autocast("cuda", enabled=False, dtype=torch.float16):
                    # batch is a dict of torchio.Tensors
                    img_batch = batch['t2'].permute(*permute_orientation)  # shape [B, 1, D, H, W]
                    seg_batch = batch['t2_seg'].permute(*permute_orientation)  # shape [B, 1, D, H, W]
                    if if_add_t1:
                        t1_img_batch = batch ['t1'].permute(*permute_orientation) # shape [B, 1, D, H, W]
                        t1 = t1_img_batch.to(device)
                    y_batch = batch['class_label']  # shape [B, num_classes]

                    img, segment, y = img_batch.to(device),seg_batch.to(device), y_batch.to(device)
                    y_smooth = y * (1 - eps) + (1 - y) * eps
                    model.train()
                    if if_add_t1:
                        y_pred, _ = model(t1, img, segment)
                    else:
                        y_pred, _ = model(img, segment)
                    loss = class_loss(y_pred, y_smooth)/accum_steps

                    running_loss += loss.item()

                loss.backward()
                if (batch_num+1) % accum_steps == 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
                    print(float(total_norm))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                del loss, y_pred, y, img, segment, batch, img_batch, y_batch, seg_batch

            epoch_train_loss = running_loss / len(train_loader)
            logging.info('fold %d, epoch %d : loss : %03f' %
                         (fold_num, epoch_num, running_loss/len(train_loader)))

            writer.add_scalar('Labeled_loss/loss', running_loss/len(train_loader), epoch_num)
            train_loss.append(epoch_train_loss)
            scheduler.step()

            # validate!!!!!!
            if epoch_num % 1 == 0:
                model.eval()
                valid_loss, score = validation(model, val_loader, device, weighted_class, permute_orientation, if_add_t1) #
                val_loss.append(valid_loss)

                if score > best_score:
                    best_score = score
                    save_mode_path = os.path.join(fold_path, 'iter_{}_AP_{}.pth'.format(epoch_num, best_score))
                    save_best_path = os.path.join(fold_path, '{}_best_model.pth'.format(args.model))
                    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    torch.save(state_cpu, save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_AP/AP', score, epoch_num)
                writer.add_scalar('Var_AP/Best_AP', best_score, epoch_num)
                model.train()

                # loss graph updates every epoch instead of every fold
                plot_and_save_losses(train_loss, val_loss,
                                     os.path.join(fold_path, 'fold_{}_{}_loss_graph.png'.format(fold_num, args.model)))

        fold_scores.append(best_score)
        fold_scores_np = np.asarray(fold_scores)
        mean_score = fold_scores_np.mean()
        std_score = fold_scores_np.std()

    writer.close()

    with open(snapshot_path + '/total fold scores.txt', 'w') as f:
        f.write('fold scors for each fold: {} \n'.format(fold_scores))
        f.write(f"Mean fold score: {mean_score:.4f}\n")
        f.write(f"Std fold score:  {std_score:.4f}\n")



