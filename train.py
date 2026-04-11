import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import os

from data.pets_dataset import get_dataloaders
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss


def spliting_weights(model, save_dir="checkpoints"):  # we need 3 diff pth files 
    os.makedirs(save_dir, exist_ok=True)   

    torch.save({
        'backbone': model.backbone.state_dict(),
        'classifier_head': model.classifier.state_dict()
    }, os.path.join(save_dir, "classifier.pth"))

    torch.save(model.locator.state_dict(), os.path.join(save_dir, "localizer.pth"))
    torch.save(model.segmenter.state_dict(), os.path.join(save_dir, "unet.pth"))


def train_one_epoch(model, dataloader, optimizer, loss_fns, device, epoch):
    model.train()

    epoch_loss_accumulator = 0.0
    cls_loss_acc = 0.0
    bbox_loss_acc = 0.0
    seg_loss_acc = 0.0

    cls_loss_fn = loss_fns['cls']              # the 3 losses
    bbox_loss_fn = loss_fns['bbox']
    seg_loss_fn = loss_fns['seg']

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch in progress_bar:
        images, cls_targets, bbox_targets, seg_targets = batch

        images = images.to(device, non_blocking=True)
        cls_targets = cls_targets.to(device, non_blocking=True)
        bbox_targets = bbox_targets.to(device, non_blocking=True).float()
        seg_targets = seg_targets.to(device, non_blocking=True).long()

        optimizer.zero_grad()

        preds = model(images)
        cls_logits = preds['classification']
        bbox_preds = preds['localization']
        seg_masks = preds['segmentation']

        loss_cls = cls_loss_fn(cls_logits, cls_targets)
        loss_bbox = bbox_loss_fn(bbox_preds, bbox_targets)
        loss_seg = seg_loss_fn(seg_masks, seg_targets)

        # bbox loss tends to dominate early, so me reduce it after warmup
        if epoch < 10:
            w_cls, w_bbox, w_seg = 2.5, 5.0, 1.0
        else:
            w_cls, w_bbox, w_seg = 2.5, 3.0, 1.0

        total_loss = (w_cls * loss_cls) + (w_bbox * loss_bbox) + (w_seg * loss_seg)

        total_loss.backward()

        # gradient clipping helping to stabilize occasional spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        epoch_loss_accumulator += total_loss.item()
        cls_loss_acc += loss_cls.item()
        bbox_loss_acc += loss_bbox.item()
        seg_loss_acc += loss_seg.item()

        progress_bar.set_postfix({'loss': total_loss.item()})

    return {
        "train/total_loss": epoch_loss_accumulator / len(dataloader),
        "train/cls_loss": cls_loss_acc / len(dataloader),
        "train/bbox_loss": bbox_loss_acc / len(dataloader),
        "train/seg_loss": seg_loss_acc / len(dataloader),
    }


def calculate_batch_iou(preds, targets):          # calci for coords
    pred_x1 = preds[:, 0] - preds[:, 2] / 2
    pred_y1 = preds[:, 1] - preds[:, 3] / 2
    pred_x2 = preds[:, 0] + preds[:, 2] / 2
    pred_y2 = preds[:, 1] + preds[:, 3] / 2

    target_x1 = targets[:, 0] - targets[:, 2] / 2
    target_y1 = targets[:, 1] - targets[:, 3] / 2
    target_x2 = targets[:, 0] + targets[:, 2] / 2
    target_y2 = targets[:, 1] + targets[:, 3] / 2

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    pred_area = preds[:, 2] * preds[:, 3]
    target_area = targets[:, 2] * targets[:, 3]
    union_area = pred_area + target_area - inter_area

    return inter_area / (union_area + 1e-6)


@torch.no_grad()
def validate(model, dataloader, loss_fns, device, epoch):
    model.eval()

    total_loss_val = 0.0

    cls_loss_fn = loss_fns['cls']
    bbox_loss_fn = loss_fns['bbox']
    seg_loss_fn = loss_fns['seg']

    all_preds, all_targets = [], []

    total_iou, correct_50 = 0.0, 0
    total_dice = 0.0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Valid]")

    for batch in progress_bar:
        images, cls_targets, bbox_targets, seg_targets = batch

        images = images.to(device, non_blocking=True)
        cls_targets = cls_targets.to(device, non_blocking=True)
        bbox_targets = bbox_targets.to(device).float()
        seg_targets = seg_targets.to(device).long()

        preds = model(images)

        cls_logits = preds['classification']
        bbox_preds = preds['localization']
        seg_masks = preds['segmentation']

        loss_cls = cls_loss_fn(cls_logits, cls_targets)
        loss_bbox = bbox_loss_fn(bbox_preds, bbox_targets)
        loss_seg = seg_loss_fn(seg_masks, seg_targets)

        total_loss = loss_cls + loss_bbox + loss_seg
        total_loss_val += total_loss.item()

        _, pred_labels = torch.max(cls_logits, 1)
        all_preds.extend(pred_labels.cpu().numpy())
        all_targets.extend(cls_targets.cpu().numpy())

        ious = calculate_batch_iou(bbox_preds, bbox_targets)
        total_iou += ious.sum().item()
        correct_50 += (ious > 0.5).sum().item()

        _, seg_pred = torch.max(seg_masks, 1)
        smooth = 1e-6
        batch_dice = 0

        for c in range(3):
            pred_c = (seg_pred == c)
            target_c = (seg_targets == c)
            intersection = (pred_c & target_c).float().sum(dim=(1, 2))
            union = pred_c.float().sum(dim=(1, 2)) + target_c.float().sum(dim=(1, 2))
            batch_dice += ((2 * intersection + smooth) / (union + smooth)).mean().item()

        total_dice += batch_dice / 3
        total_samples += images.size(0)

    macro_f1 = f1_score(all_targets, all_preds, average='macro')

    return {
        "val/total_loss": total_loss_val / len(dataloader),
        "val/macro_f1": macro_f1,
        "val/bbox_mIoU": total_iou / total_samples,
        "val/bbox_mAP_50": correct_50 / total_samples,
        "val/seg_dice": total_dice / len(dataloader)
    }


def main():
    wandb.init(project="da6401-assignment-2", config={
        "learning_rate": 1e-4,
        "epochs": 50,
        "batch_size": 16
    })

    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = get_dataloaders("./data/oxford-iiit-pet", batch_size=config.batch_size)

    model = MultiTaskPerceptionModel(37, 3).to(device)

    loss_fns = {
        'cls': nn.CrossEntropyLoss(label_smoothing=0.1),  #  try slight smoothing for generalization
        'bbox': IoULoss(),
        'seg': nn.CrossEntropyLoss()
    }

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_score = 0.0

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fns, device, epoch)
        val_metrics = validate(model, val_loader, loss_fns, device, epoch)

        f1 = val_metrics["val/macro_f1"]
        mAP = val_metrics["val/bbox_mAP_50"]
        dice = val_metrics["val/seg_dice"]

        composite = (f1 + mAP + dice) / 3

        wandb.log({**train_metrics, **val_metrics, "val/composite": composite, "epoch": epoch})

        if composite > best_score:
            best_score = composite
            spliting_weights(model)

        scheduler.step()

    wandb.finish()


if __name__ == "__main__":
    main()




