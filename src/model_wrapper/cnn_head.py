from .base_model import BaseModel
import torch
import torch.nn as nn
from .common_blocks import ConvBlock
import math


class CNN_Head(BaseModel):
    def __init__(self, model_initiator_name, output_size, base_model_output_size=None, num_cnn_layers=5, kernel_sizes=[5]):
        super().__init__(model_initiator_name)

        # Freeze all parameters in the base model
        for param in self.model.model.parameters():
            param.requires_grad = False

        if base_model_output_size is None:

            # Create a sample input tensor of the specified shape
            # Ensure this is in the correct format for your base model
            sample_input = "ATATATATAG"  # Example string input

            # Pass the sample input through the base model to determine output size
            with torch.no_grad():  # Ensure no gradients are computed in this step
                base_model_output = self.model(sample_input)

            base_model_output_size = base_model_output.shape[-1]

        # Define the ConvBlock
        filter_list = exponential_linspace_int(
            base_model_output_size,
            base_model_output_size,
            num=(num_cnn_layers + 1),
            divisible_by=2,
        )
        self.conv_layers = ConvBlock(filter_list, kernel_sizes)
        self.linear = nn.Linear(base_model_output_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        return x


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

# count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
