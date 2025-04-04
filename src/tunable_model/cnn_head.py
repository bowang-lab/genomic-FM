from .base_model import BaseModel
import torch
import torch.nn as nn
from ..model_wrapper.common_blocks import ConvBlock, FeedforwardNetwork
import math
from ..dataloader.save_as_np import apply_pca
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SUPPORTED_MODELS = [
    "zehui127/Omni-DNA-1B",
    "zehui127/Omni-DNA-116M",
    "zhihan1996/DNABERT-2-117M",
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
]



class CNN_Head(torch.nn.Module):
    def __init__(self, model_initiator_name, output_size, base_model_output_size=None, num_cnn_layers=5, kernel_sizes=[5], ff=False, dropout_rate=0.5,full_size_file_tuning=True):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_initiator_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_initiator_name, trust_remote_code=True)
        self.full_size_file_tuning = full_size_file_tuning
        # Freeze all parameters in the base model
        if not full_size_file_tuning:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        if base_model_output_size is None:

            # Create a sample input tensor of the specified shape
            # Ensure this is in the correct format for your base model
            sample_input = "ATATATATAG"  # Example string input

            # Pass the sample input through the base model to determine output size
            with torch.no_grad():  # Ensure no gradients are computed in this step
                sample_input = self.tokenizer(sample_input, return_tensors="pt")
                base_model_output = self.model(**sample_input,output_hidden_states=True).hidden_states[-1]

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
            x = torch.tensor(self.embed(x,upsample=True))- \
                torch.tensor(self.embed(x,upsample=True))
            x = x.squeeze(1)
        # x = self.conv_layers(x)
        x = torch.mean(x, dim=1)
        # x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def embed(self, x, upsample=True):
        # print(f"self.model.device is {self.model.device}")
        inputs = self.tokenizer(x, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        emb = self.model(**inputs,output_hidden_states=True).hidden_states[-1]
        if upsample:
            # print(f"shape after umsamping: {self.upsample(emb, inputs['input_ids'].shape[1]).shape}")
            return self.upsample(emb, inputs['input_ids'].shape[1]) # get the sequence length dimension
        return emb

    def upsample(self, emb, target_length):
        # repeat the emb for k*times such that it has the same length as initial input
        batch_size, seq_len, embedding_dim = emb.shape
        # Calculate the repeat factor
        repeat_factor = target_length // seq_len
        remainder = target_length % seq_len
        # Repeat the embeddings along the sequence dimension
        repeated_emb = emb.repeat_interleave(repeat_factor, dim=1)
        # Handle any remainder by slicing and appending
        if remainder > 0:
            extra_emb = emb[:, :remainder, :]
            repeated_emb = torch.cat([repeated_emb, extra_emb], dim=1)
        return repeated_emb
    def cache_embed_cnn_delta(self, data, pca_components=16):
        x = self.conv_layers(data)
        x = x.detach().numpy()
        x = apply_pca(x, n_components=pca_components)
        return x



class CNN_Head_OLD(BaseModel):
    def __init__(self, model_initiator_name, output_size, base_model_output_size=None, num_cnn_layers=5, kernel_sizes=[5], ff=True, dropout_rate=0.5,full_size_file_tuning=True):
        super().__init__(model_initiator_name)

        self.full_size_file_tuning = full_size_file_tuning
        # Freeze all parameters in the base model
        if not full_size_file_tuning:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        if base_model_output_size is None:

            # Create a sample input tensor of the specified shape
            # Ensure this is in the correct format for your base model
            sample_input = "ATATATATAG"  # Example string input

            # Pass the sample input through the base model to determine output size
            with torch.no_grad():  # Ensure no gradients are computed in this step
                sample_input = self.tokenizer(sample_input, return_tensors="pt")
                base_model_output = self.model(**sample_input,output_hidden_states=True).hidden_states[-1]

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
            x = torch.tensor(super().forward(x,upsample=True))- \
                torch.tensor(super().forward(x,upsample=True))
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
