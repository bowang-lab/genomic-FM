from torch import nn
import torch
import math
from einops import rearrange
from torch.nn.init import trunc_normal_
import torch.nn.functional as F


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class MultiKernelConvBlock(nn.Module):
    def __init__(
        self,
        dim=10,
        dim_out=None,
        kernel_sizes=[1, 3, 5],
        gelu=True,
        norm_type="group",
        linear_head=False,
        dilation=1,
        transpose=False
    ):
        super().__init__()
        self.kernel_sizes = [
            self.toOdd(kernel_size) for kernel_size in kernel_sizes
        ]
        dim_out = default(dim_out, dim)
        original_dim_out = dim_out
        dim_out = dim_out if linear_head else dim_out // len(kernel_sizes)
        if (not linear_head) and original_dim_out % len(kernel_sizes) != 0:
            raise ValueError(
                "dim_out must be divisible by the number of kernel sizes"
            )
        if transpose is False:
            self.conv_layers = nn.ModuleList(
                [
                    nn.Conv1d(
                        dim,
                        dim_out,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2 if dilation == 1 else dilation,
                        groups=1,
                        dilation=dilation,
                    )
                    for kernel_size in self.kernel_sizes
                ]
            )
        else:
            self.conv_layers = nn.ModuleList(
                [
                    nn.ConvTranspose1d(
                        dim,
                        dim_out,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2 if dilation == 1 else dilation,
                        groups=1,
                        dilation=dilation,
                    )
                    for kernel_size in self.kernel_sizes
                ]
            )
        # self.norm is a nn.GroupNorm if norm_type is group
        # and self.norm is nn.batchnorm1d if norm_type is batch
        # and self.norm is None if norm_type is None
        self.norm = None
        if norm_type == "group":
            self.norm = nn.GroupNorm(dim, dim)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(dim)
        self.gelu = nn.GELU() if gelu else None
        self.linear_head = (
            nn.Linear(len(kernel_sizes) * dim_out, dim_out)
            if linear_head
            else None
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Sequential):
            for submodule in m.children():
                self._init_weights(submodule)
        elif isinstance(m, nn.ModuleList):
            for submodule in m:
                self._init_weights(submodule)

    def toOdd(self, num):
        if num % 2 == 0:
            return num + 1
        else:
            return num

    def forward(self, x):
        # fix: x.shape = (batch, seq_len, dim) if x.shape = (batch, 1, dim, seq_len)
        if x.dim() == 4:
            x = x.squeeze(1)

        x = rearrange(x, "b s d -> b d s")
        if exists(self.norm):
            x = self.norm(x)
        if exists(self.gelu):
            x = self.gelu(x)
        x = torch.cat([conv(x) for conv in self.conv_layers], dim=1)
        x = rearrange(x, "b d s -> b s d")
        if exists(self.linear_head):
            # rearrange x from (batch, dim, seq_len) to (batch, seq_len, dim)
            x = self.linear_head(x)
        return x


class ConvBlock(torch.nn.Module):
    """ConvBlock use maxPool+CNN to encode DNA sequence to a two
    dimension vector:
        1. reduce the length dimension of DNA sequence by half
        2. use CNN to encode DNA sequence to increase
           the dimension of DNA sequence
    Other types of encoder can be added by inheriting this class
    Args:
        filter_list: a list of intergers, specifying the number of input
                     and output channels
        kernel_size: a list of integers, specifying the kernel size, default
                     to [5]
    return:
        a 2D vector of DNA sequence
    """

    def __init__(self, filter_list, kernel_size):
        super().__init__()
        self.maxpool_layer = MaxPool()

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(
                torch.nn.Sequential(
                    MultiKernelConvBlock(
                        dim_in, dim_out, kernel_sizes=kernel_size, transpose=False # Conv Tranpose when upsampling
                    ),
                    Residual(MultiKernelConvBlock(dim_out, dim_out, [1])),
                )
            )
        self.conv_tower = torch.nn.ModuleList(conv_layers)

    def addPooling(self, x):
        for layer in self.conv_tower:
            x = layer(x)
            x = self.maxpool_layer(x)
        return x

    def forward(self, x):
        return self.addPooling(x)


class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class MaxPool(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = rearrange(x, "b l d-> b d l")
        x = F.max_pool1d(x, self.kernel_size, self.stride, return_indices=False)
        x = rearrange(x, "b d l -> b l d")
        return x

class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        hidden_size = 2*input_size
        super(FeedforwardNetwork, self).__init__()
        # Define the first layer (input to hidden)
        self.hidden = nn.Linear(input_size, hidden_size)
        # Define the second layer (hidden to output)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through the first layer, then apply ReLU activation
        x = F.relu(self.hidden(x))
        # Forward pass through the output layer
        x = self.output(x)
        return x
