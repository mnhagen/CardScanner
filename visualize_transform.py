import matplotlib.pyplot as plt
from PIL import Image
import torch
from model_definition import get_transforms

# ----------------------
# Load your transforms
# ----------------------

random_transform, _ = get_transforms()

# ----------------------
# Load local image
# ----------------------
img_path = "data\EOE\dataset\EOE_Adagia_ Windswept Bastion_250\EOE_Adagia_ Windswept Bastion_250.jpg"  # <-- replace with actual file
img = Image.open(img_path).convert("RGB")

# ----------------------
# Apply transform multiple times to see variation
# ----------------------

num_plots = 5
fig, axes = plt.subplots(1, num_plots, figsize=(20, 5))
axes[0].imshow(img)
axes[0].axis("off")
axes[0].set_title("unaltered card")
for i in range(num_plots - 1):
    t_img = random_transform(img)  # tensor (C,H,W)
    # Unnormalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    unnorm = t_img * std + mean
    unnorm = unnorm.clamp(0,1)  # keep values in [0,1]
    axes[i+1].imshow(unnorm.permute(1,2,0))  # convert CHW -> HWC
    axes[i+1].axis("off")
    axes[i+1].set_title(f"random transform {i+2}")
plt.show()
