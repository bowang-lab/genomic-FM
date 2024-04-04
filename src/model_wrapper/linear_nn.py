from .base_model import BaseModel
import torch
import torch.nn as nn


class LinearNN(BaseModel):
    def __init__(self, model_initiator_name, output_size):
        super().__init__(model_initiator_name)

        # Freeze all parameters in the base model
        for param in self.model.model.parameters():
            param.requires_grad = False

        # Create a sample input tensor of the specified shape
        # Ensure this is in the correct format for your base model
        sample_input = "ATATATATAG"  # Example string input

        # Pass the sample input through the base model to determine output size
        with torch.no_grad():  # Ensure no gradients are computed in this step
            base_model_output = self.model(sample_input)

        base_model_output_size = base_model_output.shape[-1]

        # Define the linear layer with the determined size
        self.head = nn.Linear(base_model_output_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        x = x.squeeze(1)
        sentence_embedding = torch.mean(x, dim=1)
        return self.head(sentence_embedding)
