import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

from models.multitask import MultiTaskPerceptionModel
from data.pets_dataset import get_dataloaders

activations = {}

def get_activation(name):
    """Hook to intercept and save the feature map during the forward pass."""
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

def main():
    # save plots
    wandb.init(project="da6401-assignment-2", name="Task_2.4_Feature_Maps")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # dataloader
    dataset_path = "./data/oxford-iiit-pet"
    _, val_loader = get_dataloaders(root_dir=dataset_path, batch_size=1)
    
    
    image_tensor = None
    for images, cls_targets, _, _ in val_loader:
        if cls_targets[0].item() < 24: 
            image_tensor = images.to(device)
            print(f"Found a dog! (Class Index: {cls_targets[0].item()})")
            break

    # model laoding 
    print("Loading model and weights...")
    model = MultiTaskPerceptionModel(num_classes=37, num_seg_classes=3, use_bn=True).to(device)
    model.eval()

    # 
    backbone_convs = [m for m in model.backbone.modules() if isinstance(m, nn.Conv2d)]
    first_conv = backbone_convs[0]
    last_conv = backbone_convs[-1]

    print(f"🔗 Hooking First Layer: {first_conv}")
    print(f"🔗 Hooking Last Layer: {last_conv}")
    first_conv.register_forward_hook(get_activation("first_layer"))
    last_conv.register_forward_hook(get_activation("last_layer"))

    # image pass in model
    with torch.no_grad():
        _ = model(image_tensor)

    
    # reversing imagenet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    orig_img = image_tensor[0] * std + mean
    orig_img = orig_img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

    fig_orig = plt.figure(figsize=(6, 6))
    plt.imshow(orig_img)
    plt.title("Original Dog Image", fontsize=16)
    plt.axis('off')
    plt.tight_layout()

#plotting channels
    def plot_feature_maps(feature_map, title):
        channels = feature_map.squeeze(0) # Remove batch dimension
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle(title, fontsize=16, y=1.02)
        
        for i, ax in enumerate(axes.flatten()):
            if i < channels.size(0):
                # Normalize the channel for visualization
                fmap = channels[i].numpy()
                fmap -= fmap.min()
                fmap /= (fmap.max() + 1e-5)
                ax.imshow(fmap, cmap='viridis')
            ax.axis('off')
        plt.tight_layout()
        return fig

    # log plots 
    print("Generating plots...")
    fig_first = plot_feature_maps(activations["first_layer"], "First Conv Layer (Low-Level Features)")
    fig_last = plot_feature_maps(activations["last_layer"], "Last Conv Layer (High-Level Features)")

    def plot_mean_activation(feature_map, title):
        # avg across all channels
        mean_map = feature_map.squeeze(0).mean(dim=0).numpy()
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(mean_map, cmap='hot') # 
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        return fig

    fig_mean_first = plot_mean_activation(activations["first_layer"], "Mean Activation (First Layer)")
    fig_mean_last = plot_mean_activation(activations["last_layer"], "Mean Activation (Last Layer)")

    
    wandb.log({
        "Task 2.4/1_Original_Image": wandb.Image(fig_orig),
        "Task 2.4/2_First_Layer": wandb.Image(fig_first),
        "Task 2.4/3_Last_Layer": wandb.Image(fig_last),
        "Task 2.4/4_Mean_First_Layer": wandb.Image(fig_mean_first), # first
        "Task 2.4/5_Mean_Last_Layer": wandb.Image(fig_mean_last)    # last
    })
    
    print(" Feature maps saved to Weights & Biases!")
    plt.show()
    wandb.finish()

if __name__ == "__main__":
    main()
