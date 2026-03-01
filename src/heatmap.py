import matplotlib.pyplot as plt
import torch
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

def generate_heatmap(model, train_loader, device):

    model.eval()

    # 1) Take a batch
    Xb, yb = next(iter(train_loader))
    img_tensor = Xb[0:1].to(device)  # (1,3,224,224)
    img_tensor.requires_grad_()
    target_class = yb[0].item()

    # 2) Create extractor
    cam_extractor = SmoothGradCAMpp(model, target_layer='layer4')

    # Forward
    logits = model(img_tensor)
    pred_class = logits.argmax(1).item()

    # 3) Heatmap calculation
    activation_map = cam_extractor(pred_class, logits)
    heatmap = activation_map[0].squeeze(0)  # (H,W)

    # 4) Resize to 224x224
    heatmap_tensor = heatmap.unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(
        heatmap_tensor,
        size=(224,224),
        mode='bilinear',
        align_corners=False
    )

    heatmap_resized = heatmap_resized.squeeze().cpu().numpy()

    # 5) Convert image to PIL 
    img = img_tensor[0].cpu()
    img = (img - img.min()) / (img.max() - img.min())
    img_pil = to_pil_image(img).convert("RGB")

    # 6) Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12,6))

    # Original image
    axes[0].imshow(img_pil)
    axes[0].axis('off')
    axes[0].set_title("Original image")

    # Grad-CAM image
    axes[1].imshow(img_pil)
    axes[1].imshow(heatmap_resized, cmap='turbo', alpha=0.6)
    axes[1].axis('off')
    axes[1].set_title(f'Grad-CAM\nReal: {target_class} | Pred: {pred_class}')

    plt.tight_layout()
    plt.show()