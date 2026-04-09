import torch.nn as nn
from .vgg11 import VGG11Backbone, ClassificationHead
from .localization import RegressionHead
from .segmentation import UNetDecoder
import gdown

class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_classes=37, num_seg_classes=3):
        super().__init__()
        
        # 1. Define the network architecture first
        self.backbone = VGG11Backbone()
        # (Make sure your classification, bounding box, and segmentation 
        #  heads are initialized right here just like before!)

        # 2. Define the download paths
        classifier_path = "checkpoints/classifier.pth"
        localizer_path = "checkpoints/localizer.pth"
        unet_path = "checkpoints/unet.pth"

        # 3. Download the files via gdown 
        # ⚠️ PASTE YOUR BRAND NEW GOOGLE DRIVE IDS HERE!
        import gdown
        print("Downloading weights from Google Drive...")
        gdown.download(id="YOUR_NEW_CLASSIFIER_ID", output=classifier_path, quiet=False)
        gdown.download(id="YOUR_NEW_LOCALIZER_ID", output=localizer_path, quiet=False)
        gdown.download(id="YOUR_NEW_UNET_ID", output=unet_path, quiet=False)
            
        # 4. FORCE LOAD THE WEIGHTS (No silent skipping!)
        print("Loading weights into model...")
        
        cls_checkpoint = torch.load(classifier_path, map_location="cpu")
        self.backbone.load_state_dict(cls_checkpoint['backbone'])
        
        # NOTE: Make sure 'self.classifier' matches the actual name of your head!
        self.classifier.load_state_dict(cls_checkpoint['classifier_head'])
        
        # NOTE: Make sure 'self.locator' matches the actual name of your bbox head!
        self.locator.load_state_dict(torch.load(localizer_path, map_location="cpu"))
        
        # NOTE: Make sure 'self.segmenter' matches the actual name of your Unet head!
        self.segmenter.load_state_dict(torch.load(unet_path, map_location="cpu"))
            
        print("Successfully loaded all pretrained weights!")

    def forward(self, x):
        # Shared backbone
        bottleneck, skip_features = self.backbone(x)
        
        # 1. Breed Label [cite: 44]
        class_logits = self.classifier(bottleneck)
        
        # 2. Bounding Box [cite: 45]
        bbox_coords = self.locator(bottleneck)
        
        # 3. Segmentation Mask [cite: 46]
        seg_mask = self.segmenter(bottleneck, skip_features)
        
        # return class_logits, bbox_coords, seg_mask
        return {
            'classification': class_logits,
            'localization': bbox_coords,
            'segmentation': seg_mask
        }
