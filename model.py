"""
some modifications:
1. use instance normalization instead of batch normalization
2. use VGG-19 with batchnorm instead of VGG-16
"""

import torch.nn as nn
import torchvision.models as tvmodels


class TransformationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # down sampling
            ConvLayer(32, 64, 3, 2),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 5 res blocks
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),

            # up sampling
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1),
            nn.Tanh()

        )

    def forward(self, inp):
        out = self.model(inp)  # out is in range [-1, 1]
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            ConvLayer(in_channels, in_channels, 3, 1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(),
            ConvLayer(in_channels, in_channels, 3, 1),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, inp):
        out = self.model(inp) + inp
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        pad_size = kernel_size // 2
        self.padding = nn.ReflectionPad2d(pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, inp):
        x = self.padding(inp)
        out = self.conv(x)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = tvmodels.vgg19_bn(pretrained=True).features[:35]

        # freeze model
        for children in self.vgg.children():
            for param in children.parameters():
                param.requires_grad = False

        self.style_output_layers = {5: 'relu_1_2',
                                     12: 'relu_2_2',
                                     22: 'relu_3_3',
                                     35: 'relu_4_3'}

        self.content_output_layers = {22: 'relu_3_3'}

    def forward(self, inp, target='style'):
        assert target in ['style', 'content']

        if target == 'content':
            output_layer = self.content_output_layers
        else:
            output_layer = self.style_output_layers

        x = inp
        output = {}
        for idx in range(len(self.vgg)):
            module = self.vgg[idx]
            x = module(x)
            if idx in output_layer:
                output[output_layer[idx]] = x
                if idx == 22 and target == 'content':
                    break

        return output

