import torch
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
        
        # ---> THE FIX: Actually building the 3 task heads! <---
        self.classifier = ClassificationHead(num_classes=num_classes)
        self.locator = RegressionHead()
        self.segmenter = UNetDecoder(num_classes=num_seg_classes)

        # 2. Define the download paths
        classifier_path = "checkpoints/classifier.pth"
        localizer_path = "checkpoints/localizer.pth"
        unet_path = "checkpoints/unet.pth"

        # 3. Download the files via gdown 
        print("Downloading weights from Google Drive...")
        # gdown.download(id="1fwQn62hYGGhZjxMtoxMO5BaqgesjhRy1", output=classifier_path, quiet=False)
        # gdown.download(id="1QTniV0lgu7ho1HY2EOdpyIwDguHeRg3c", output=localizer_path, quiet=False)
        # gdown.download(id="1GZYoxunNcZ5U9ne_jVgdXBrYQAa12U_F", output=unet_path, quiet=False)

        # for 40 epochs
        gdown.download(id="16F5q7AGYELb09JIy4lgacICizM9PuVJc", output=classifier_path, quiet=False)
        gdown.download(id="1rrSHIr0Y8I-D0wY0VWV9FOLIZWjMlXnA", output=localizer_path, quiet=False)
        gdown.download(id="13ekfanq3G5B_mVLGGMWyTFyPQT0OMW9R", output=unet_path, quiet=False)
            
        # 4. FORCE LOAD THE WEIGHTS
        print("Loading weights into model...")
        
        cls_checkpoint = torch.load(classifier_path, map_location="cpu")
        self.backbone.load_state_dict(cls_checkpoint['backbone'])
        self.classifier.load_state_dict(cls_checkpoint['classifier_head'])
        self.locator.load_state_dict(torch.load(localizer_path, map_location="cpu"))
        self.segmenter.load_state_dict(torch.load(unet_path, map_location="cpu"))
            
        print("Successfully loaded all pretrained weights!")

    def forward(self, x):
        # Shared backbone
        bottleneck, skip_features = self.backbone(x)
        
        # 1. Breed Label
        class_logits = self.classifier(bottleneck)
        
        # 2. Bounding Box (Currently between 0.0 and 1.0)
        bbox_coords = self.locator(bottleneck)
        
        # ---> THE FIX: Scale to Image Space <---
        # x.shape is [Batch, Channels, Height, Width] (e.g., [10, 3, 224, 224])
        _, _, H, W = x.shape 
        
        # Create a scaling tensor: [Width, Height, Width, Height]
        scale_tensor = torch.tensor([W, H, W, H], device=bbox_coords.device)
        
        # Multiply the normalized coordinates by the image dimensions
        bbox_coords = bbox_coords * scale_tensor
        # ---------------------------------------
        
        # 3. Segmentation Mask
        seg_mask = self.segmenter(bottleneck, skip_features)
        
        return {
            'classification': class_logits,
            'localization': bbox_coords, # Now these are absolute pixels!
            'segmentation': seg_mask
        }






# import torch
# import torch.nn as nn
# from .vgg11 import VGG11Backbone, ClassificationHead
# from .localization import RegressionHead
# from .segmentation import UNetDecoder
# import gdown


# class MultiTaskPerceptionModel(nn.Module):
#     def __init__(self, num_classes=37, num_seg_classes=3):
#         super().__init__()
        
#         # backbone is sharing across all 3 tasks
#         self.bb = VGG11Backbone()
        
#         #   heads 
#         self.head_cls = ClassificationHead(num_classes=num_classes)
#         self.head_box = RegressionHead()
#         self.head_seg = UNetDecoder(num_classes=num_seg_classes)

#         # weight store
#         p_cls = "checkpoints/classifier.pth"
#         p_box = "checkpoints/localizer.pth"
#         p_seg = "checkpoints/unet.pth"

#        # gdrive weight loading
#         print("Downloading weights from Google Drive...")
#         # for 20 epochs
#         # gdown.download(id="1fwQn62hYGGhZjxMtoxMO5BaqgesjhRy1", output=p_cls, quiet=False)
#         # gdown.download(id="1QTniV0lgu7ho1HY2EOdpyIwDguHeRg3c", output=p_box, quiet=False)
#         # gdown.download(id="1GZYoxunNcZ5U9ne_jVgdXBrYQAa12U_F", output=p_seg, quiet=False)

#         # for 40 epochs
#         gdown.download(id="16F5q7AGYELb09JIy4lgacICizM9PuVJc", output=p_cls, quiet=False)
#         gdown.download(id="1rrSHIr0Y8I-D0wY0VWV9FOLIZWjMlXnA", output=p_box, quiet=False)
#         gdown.download(id="13ekfanq3G5B_mVLGGMWyTFyPQT0OMW9R", output=p_seg, quiet=False)
            
        
#         print("Loading weights into model...")
        
#         chk = torch.load(p_cls, map_location="cpu")
#         self.bb.load_state_dict(chk['backbone'])
#         self.head_cls.load_state_dict(chk['classifier_head'])

#         self.head_box.load_state_dict(torch.load(p_box, map_location="cpu"))
#         self.head_seg.load_state_dict(torch.load(p_seg, map_location="cpu"))
            
#         print("Successfully loaded all pretrained weights!")

#     def forward(self, x):
        
#         feat_bot, skips = self.bb(x)
#         out_cls = self.head_cls(feat_bot)
        
        
#         out_box = self.head_box(feat_bot)
        
#         _, _, h, w = x.shape 
        
#         # make tensor like [w, h, w, h]
#         scale_vec = torch.tensor([w, h, w, h], device=out_box.device)
        
     
#         out_box = out_box * scale_vec
        
#         # segmentation branch ( skip connections also)
#         out_seg = self.head_seg(feat_bot, skips)
        
#         return {
#             'classification': out_cls,
#             'localization': out_box,  # now in pixel coords, removing normailization
#             'segmentation': out_seg
#         }




