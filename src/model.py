import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(labels_list, device, lr):

    unique_labels = sorted(set(labels_list))
    NUM_CLASSES = len(unique_labels)

    # Load pre-trained ResNet18 in ImageNet
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    in_features = model.fc.in_features

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(in_features, NUM_CLASSES)

    model = model.to(device)

    fn_perdida = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(model.parameters(), lr=lr)

    return model, fn_perdida, optimizador