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

# Import your custom modules
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss
# from data.pets_dataset import get_dataloaders  <-- You'll need to implement this

def train_one_epoch(model, dataloader, optimizer, criteria, device, epoch):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_bbox_loss = 0.0
    running_seg_loss = 0.0

    # criteria is a dictionary containing our 3 loss functions
    cls_criterion = criteria['cls']
    bbox_criterion = criteria['bbox']
    seg_criterion = criteria['seg']

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for images, cls_targets, bbox_targets, seg_targets in progress_bar:
        images = images.to(device)
        cls_targets = cls_targets.to(device)
        bbox_targets = bbox_targets.to(device).float()
        seg_targets = seg_targets.to(device).long() # Trimap classes

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: Single pass yielding all three outputs
        cls_logits, bbox_preds, seg_masks = model(images)

        # Calculate individual losses
        loss_cls = cls_criterion(cls_logits, cls_targets)
        loss_bbox = bbox_criterion(bbox_preds, bbox_targets)
        loss_seg = seg_criterion(seg_masks, seg_targets)

        # Multi-task Loss Weights (You can tune these as hyperparameters in W&B)
        # Often, regression losses are scaled up because their numerical values are smaller
        w_cls, w_bbox, w_seg = 1.0, 5.0, 1.0 

        # Unified Loss Formulation
        total_loss = (w_cls * loss_cls) + (w_bbox * loss_bbox) + (w_seg * loss_seg)

        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += total_loss.item()
        running_cls_loss += loss_cls.item()
        running_bbox_loss += loss_bbox.item()
        running_seg_loss += loss_seg.item()

        progress_bar.set_postfix({'Total Loss': total_loss.item()})

    # Averages for the epoch
    epoch_metrics = {
        "train/total_loss": running_loss / len(dataloader),
        "train/cls_loss": running_cls_loss / len(dataloader),
        "train/bbox_loss": running_bbox_loss / len(dataloader),
        "train/seg_loss": running_seg_loss / len(dataloader),
    }
    return epoch_metrics

def calculate_batch_iou(preds, targets):
    """Calculates IoU for a batch of bounding box predictions [cx, cy, w, h]"""
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
def validate(model, dataloader, criteria, device, epoch):
    model.eval()
    running_loss = 0.0
    
    cls_criterion = criteria['cls']
    bbox_criterion = criteria['bbox']
    seg_criterion = criteria['seg']

    # Metric accumulators
    all_cls_preds = []
    all_cls_targets = []
    
    total_dice_score = 0.0
    total_iou = 0.0
    correct_detections_50 = 0 # For single-object mAP@0.5 approximation
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Valid]")

    for images, cls_targets, bbox_targets, seg_targets in progress_bar:
        images = images.to(device)
        cls_targets = cls_targets.to(device)
        bbox_targets = bbox_targets.to(device).float()
        seg_targets = seg_targets.to(device).long()

        # Forward Pass
        cls_logits, bbox_preds, seg_masks = model(images)
        
        # ---------------- LOSS CALCULATIONS ----------------
        loss_cls = cls_criterion(cls_logits, cls_targets)
        loss_bbox = bbox_criterion(bbox_preds, bbox_targets)
        loss_seg = seg_criterion(seg_masks, seg_targets)

        w_cls, w_bbox, w_seg = 1.0, 5.0, 1.0 
        total_loss = (w_cls * loss_cls) + (w_bbox * loss_bbox) + (w_seg * loss_seg)
        running_loss += total_loss.item()
        
        # ---------------- METRIC CALCULATIONS ----------------
        
        # 1. Classification (Accumulate for F1)
        _, cls_pred_labels = torch.max(cls_logits, 1)
        all_cls_preds.extend(cls_pred_labels.cpu().numpy())
        all_cls_targets.extend(cls_targets.cpu().numpy())
        
        # 2. Bounding Box (mAP / Mean IoU)
        # For single object localization, mAP@0.5 is equivalent to accuracy at IoU > 0.5
        ious = calculate_batch_iou(bbox_preds, bbox_targets)
        total_iou += ious.sum().item()
        correct_detections_50 += (ious > 0.5).sum().item()
        
        # 3. Segmentation (Multi-Class Dice Score)
        _, seg_pred_masks = torch.max(seg_masks, 1)
        batch_dice = 0.0
        smooth = 1e-6
        # Calculate Dice per class (0, 1, 2) and average for the batch
        for c in range(3):
            pred_c = (seg_pred_masks == c)
            target_c = (seg_targets == c)
            intersection = (pred_c & target_c).float().sum(dim=(1, 2))
            union = pred_c.float().sum(dim=(1, 2)) + target_c.float().sum(dim=(1, 2))
            dice_c = (2. * intersection + smooth) / (union + smooth)
            batch_dice += dice_c.mean().item()
            
        total_dice_score += (batch_dice / 3.0) # Macro average across the 3 trimap classes
        total_samples += images.size(0)

        progress_bar.set_postfix({'Val Loss': total_loss.item()})

    # ---------------- FINALIZE METRICS ----------------
    
    # Macro F1 Score for 37 classes
    macro_f1 = f1_score(all_cls_targets, all_cls_preds, average='macro')
    
    # Bounding Box metrics
    mean_iou = total_iou / total_samples
    map_50 = correct_detections_50 / total_samples
    
    # Segmentation metric
    mean_dice = total_dice_score / len(dataloader)

    epoch_metrics = {
        "val/total_loss": running_loss / len(dataloader),
        "val/macro_f1": macro_f1,
        "val/bbox_mIoU": mean_iou,
        "val/bbox_mAP_50": map_50,
        "val/seg_dice": mean_dice
    }
    return epoch_metrics

def main():
    # 1. Initialize Weights & Biases
    wandb.init(
        project="da6401-assignment-2",
        config={
            "learning_rate": 1e-4,
            "epochs": 20,
            "batch_size": 16,
            "architecture": "VGG11-UNet-MultiTask"
        }
    )
    config = wandb.config

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") # Apple Silicon GPU!
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # 2. DataLoaders 
    # Point this to the folder you just created and extracted the images into!
    dataset_path = "./data/oxford-iiit-pet" 
    
    train_loader, val_loader = get_dataloaders(root_dir=dataset_path, batch_size=config.batch_size)
    
    # # --- PLACEHOLDER FOR TESTING SCRIPT WITHOUT DATA ---
    # # Delete this block once you have your dataset ready
    # class DummyLoader:
    #     def __iter__(self):
    #         for _ in range(10):
    #             yield (torch.randn(config.batch_size, 3, 224, 224), 
    #                    torch.randint(0, 37, (config.batch_size,)), 
    #                    torch.rand(config.batch_size, 4), 
    #                    torch.randint(0, 3, (config.batch_size, 224, 224)))
    #     def __len__(self): return 10
    # train_loader = val_loader = DummyLoader()
    # ---------------------------------------------------

    # 3. Model Setup
    model = MultiTaskPerceptionModel(num_classes=37, num_seg_classes=3).to(device)

    # 4. Loss Functions Dictionary
    criteria = {
        'cls': nn.CrossEntropyLoss(),
        'bbox': IoULoss(), # Your custom module
        'seg': nn.CrossEntropyLoss() # Standard for semantic segmentation (can swap for Dice)
    }

    # 5. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 6. Training Loop
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criteria, device, epoch)
        val_metrics = validate(model, val_loader, criteria, device, epoch)

        # Log everything to W&B
        wandb.log({**train_metrics, **val_metrics, "epoch": epoch})
        
        # Save checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/multitask_epoch_{epoch}.pth")

    wandb.finish()

if __name__ == "__main__":
    main()