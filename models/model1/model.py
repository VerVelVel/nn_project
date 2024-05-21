import torch
from torch import nn
from torchvision.models import resnet152, ResNet152_Weights


class ResNet_1(nn.Module):
    def __init__(self):
        super().__init__()

        # подгружаем модель
        self.model = resnet152(weights=ResNet152_Weights.DEFAULT)
        # заменяем слой
        self.model.fc = nn.Linear(2048, 100)
        # замораживаем слои
        for i in self.model.parameters():
            i.requires_grad = False
        # размораживаем только последний, который будем обучать
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)

