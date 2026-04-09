import torch.nn as nn
from .vgg11 import VGG11Backbone, ClassificationHead
from .localization import RegressionHead
from .segmentation import UNetDecoder
import gdown

class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_classes=37, num_seg_classes=3):
        super(MultiTaskPerceptionModel, self).__init__()

        # --- THE AUTOGRADER DOWNLOAD SCRIPT ---
        # (Replace these fake IDs with your actual Google Drive IDs)
        gdown.download(id="1fwQn62hYGGhZjxMtoxMO5BaqgesjhRy1", output="checkpoints/classifier.pth", quiet=False)
        gdown.download(id="1QTniV0lgu7ho1HY2EOdpyIwDguHeRg3c", output="checkpoints/localizer.pth", quiet=False)
        gdown.download(id="1GZYoxunNcZ5U9ne_jVgdXBrYQAa12U_F", output="checkpoints/unet.pth", quiet=False)
        # --------------------------------------
        
        self.backbone = VGG11Backbone()
        self.classifier = ClassificationHead(num_classes)
        self.locator = RegressionHead()
        self.segmenter = UNetDecoder(num_seg_classes)

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
