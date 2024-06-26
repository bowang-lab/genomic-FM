from .base_model import BaseModel
import torch
import torch.nn as nn
from .common_blocks import ConvBlock, FeedforwardNetwork
import math
from ..dataloader.save_as_np import apply_pca

class CNN_Head(BaseModel):
    def __init__(self, model_initiator_name, output_size, base_model_output_size=None, num_cnn_layers=5, kernel_sizes=[5], ff=True, dropout_rate=0.5,full_size_file_tuning=False):
        super().__init__(model_initiator_name)

        self.full_size_file_tuning = full_size_file_tuning
        # Freeze all parameters in the base model
        if not full_size_file_tuning:
            for param in self.model.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.model.parameters():
                param.requires_grad = True

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
        self.linear = nn.Linear(base_model_output_size, output_size) if not ff else FeedforwardNetwork(base_model_output_size, output_size)
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.full_size_file_tuning:
            x = torch.tensor(self.model.embed(x[1],upsample_embeddings=True))-torch.tensor(self.model.embed(x[0],upsample_embeddings=True))
            x = x.squeeze(1)
        x = self.conv_layers(x)
        x = torch.mean(x, dim=1)
        # x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x
    def cache_embed_cnn_delta(self, data, pca_components=16):
        x = self.conv_layers(data)
        x = x.detach().numpy()
        x = apply_pca(x, n_components=pca_components)
        return x


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

# count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
