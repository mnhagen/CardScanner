import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

def get_transforms():
    random_transform = transforms.Compose([
        transforms.RandomRotation(degrees=2),
        transforms.RandomPerspective(distortion_scale=0.03, p=0.5),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.75),
        #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    clean_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return random_transform, clean_transform

def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
