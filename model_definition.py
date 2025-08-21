import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

def get_transform():
    random_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(degrees=5),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    clean_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return random_transform, clean_transform

def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
