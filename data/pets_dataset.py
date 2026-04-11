import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        # intializing all 
        self.root_dir = root_dir
        self.split = split
        self.tfms = transform   # renamed just for convenience
        
        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        self.xml_dir = os.path.join(root_dir, 'annotations', 'xmls')
        
        #  split file making for storing best till now
        split_path = os.path.join(root_dir, 'annotations', f'{split}.txt')
        
        self.data_list = []
        with open(split_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_id = parts[0]
                
                # label chganging
                cls_id = int(parts[1]) - 1 
                
                xml_p = os.path.join(self.xml_dir, f'{img_id}.xml')
                
                # only keeping samples jaha pe bbox actually exists
                if os.path.exists(xml_p):
                    self.data_list.append((img_id, cls_id))

    def _parse_xml_bbox(self, xml_path, W, H):
        """reading bbox from xml and converting to yolo style (normalized)"""
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        box = root.find('.//bndbox')
        
        xmin = float(box.find('xmin').text)
        ymin = float(box.find('ymin').text)
        xmax = float(box.find('xmax').text)
        ymax = float(box.find('ymax').text)
        
        # centering
        bw = xmax - xmin
        bh = ymax - ymin
        cx = xmin + bw / 2
        cy = ymin + bh / 2
        
        # normalizeing
        return [cx / W, cy / H, bw / W, bh / H]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_id, cls_id = self.data_list[idx]
        
        
        img_path = os.path.join(self.img_dir, f'{img_id}.jpg')
        img = np.array(Image.open(img_path).convert('RGB'))
        H, W = img.shape[:2]
        
        
        mask_path = os.path.join(self.mask_dir, f'{img_id}.png')
        
        
        mask = np.array(Image.open(mask_path)) - 1
        
        
        xml_p = os.path.join(self.xml_dir, f'{img_id}.xml')
        box = self._parse_xml_bbox(xml_p, W, H)
        
        if self.tfms:
            # albumentations expects list of bboxes
            aug = self.tfms(
                image=img,
                mask=mask,
                bboxes=[box],
                class_labels=[cls_id]
            )
            
            img = aug['image']
            mask = aug['mask']
            box = aug['bboxes'][0]  # only one bbox 
        
        #  tensors conversion
        cls_id = torch.tensor(cls_id, dtype=torch.long)
        box = torch.tensor(box, dtype=torch.float32)
        
        # mask already tensor from , just ensuring type
        mask = mask.clone().detach().long()
        
        return img, cls_id, box, mask


def get_dataloaders(root_dir, batch_size=16):
    """creates train + val loaders (simple split)"""
    
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    train_tfms = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),   # simple augmen
        A.ColorJitter(p=0.2),
        norm,
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    val_tfms = A.Compose([
        A.Resize(224, 224),
        norm,
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # using trainval split since test doesnt have bbox xmls
    train_ds = OxfordIIITPetDataset(root_dir=root_dir, split='trainval', transform=train_tfms)
    val_ds = OxfordIIITPetDataset(root_dir=root_dir, split='trainval', transform=val_tfms)
    
    
    N = len(train_ds.data_list)
    idxs = list(range(N))
    np.random.shuffle(idxs)
    
    cut = int(np.floor(0.2 * N))
    tr_idx, va_idx = idxs[cut:], idxs[:cut]
    
    train_ds.data_list = [train_ds.data_list[i] for i in tr_idx]
    val_ds.data_list = [val_ds.data_list[i] for i in va_idx]
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader


