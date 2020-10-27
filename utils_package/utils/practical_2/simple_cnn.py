import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, input_dim=3, internal_dim=64, output_dim=64, stride=2,
                 encode=True, device='cpu', data_type='image',
                 place_on_device=True):
        super(SimpleCNN, self).__init__()

        self.output_shape = 64  # 1024

        self.name = 'Simple CNN'
        self.encode = encode
        self.device = device
        self.data_type = data_type
        self.place_on_device = place_on_device

        self.encoder = nn.Sequential(
            conv_block(input_dim, internal_dim, stride),
            conv_block(internal_dim, internal_dim, stride),
            conv_block(internal_dim, internal_dim, stride),
            conv_block(internal_dim, output_dim, stride),
        )

        self.avgpool = nn.AvgPool2d(14, stride=1)
        self.classification = nn.Linear(1600, 64)

        if device is not None and self.place_on_device:
            self.to(device)

    def forward(self, x):
        if self.data_type == 'image':
            x = torch.stack(x)

        if self.place_on_device:
            x = x.to(self.device)

        x = self.encoder(x)
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        if not self.encode:
            x = self.classification(x)

        return x


def conv_block(in_channels, out_channels, stride):

    layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=stride)
        )

    return layers
