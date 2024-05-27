import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_Block_V1(nn.Module):  # handle different spatial sizes
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            )
        self.kernels = nn.ModuleList(kernels)  # register kernels by ModuleList
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[1, 2 * i + 3],
                    padding=[0, i + 1],
                )
            )
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[2 * i + 3, 1],
                    padding=[i + 1, 0],
                )
            )
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        # Compute the padding size required for causality
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x):
        # only left-side padding is required
        x = F.pad(x, (self.padding, 0))
        # Perform the convolution
        out = self.conv(x)
        return out


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = CausalConv(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.conv2 = CausalConv(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final
            else None
        )

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(
            *[
                DilatedConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    final=(i == len(channels) - 1),
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x):
        return self.net(x)


class TemporalSpatialConv(nn.Module):
    # Initialize EEGNet
    def __init__(self, f1, d, channels, kernel_size, dropout_rate):
        super(TemporalSpatialConv, self).__init__()
        # Number of spatial filters to learn within each temporal filter.
        self.f2 = f1 * d
        # Convolutional blocks
        # Block 1
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_size), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.Dropout(dropout_rate),
        )
        # Depthwise Conv2D
        self.depthwise = nn.Sequential(
            nn.Conv2d(f1, self.f2, (channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(self.f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )
        # Separable Conv2D
        self.separable = nn.Sequential(
            nn.Conv2d(
                self.f2, self.f2, (1, 16), padding="same", bias=False, groups=self.f2
            ),
            nn.BatchNorm2d(self.f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

    # Forward pass
    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)
        # Apply blocks
        x = self.temporal(x)
        # print(x.shape)
        x = self.depthwise(x)
        # print(x.shape)
        x = self.separable(x)
        # print(x.shape)
        x = x.squeeze()
        # print(x.shape)
        return x
