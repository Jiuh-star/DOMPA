from __future__ import annotations

import torch
import torch.nn as nn
import torchvision as tv


class CnnModel(nn.Module):
    def __init__(self, in_channels=1, num_class=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


class RobustCnnModel(nn.Module):
    def __init__(self, in_channels=1, num_class=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True)
        )
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


class LstmModel(nn.Module):
    def __init__(self, embedding_dim=8, vocab_size=80, hidden_size=256):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output


class MobileNetV3(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.model = tv.models.mobilenet_v3_small(
            weights=tv.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
            # num_classes=num_class,
        )
        self.fc = nn.Linear(1000, num_class, bias=True)
        # self.model.classifier[3] = nn.Linear(1024, num_class, bias=True)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.model = tv.models.resnet18(
            # weights=tv.models.ResNet18_Weights.DEFAULT,
            num_classes=num_class,
        )
        # self.model.fc = nn.Linear(512, num_class, bias=True)

    def forward(self, x):
        return self.model(x)


class DnnModel(nn.Module):
    def __init__(self, in_features: int = 784, num_class: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_class),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class ResNet18GN(nn.Module):
    def __init__(self, num_class: int = 10):
        super().__init__()
        resnet18 = tv.models.resnet18()
        resnet18.fc = nn.Linear(512, num_class)
        resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

        resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

        resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        self.model = resnet18

    def forward(self, x):
        return self.model(x)
