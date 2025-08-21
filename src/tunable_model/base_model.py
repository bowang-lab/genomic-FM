import yaml
import torch
import importlib
import os
import sys
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SUPPORTED_MODELS = [
    "zehui127/Omni-DNA-1B",
    "zehui127/Omni-DNA-116M",
    "zhihan1996/DNABERT-2-117M",
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    "LucaGroup/LucaOne-default-step36M"
]

class BaseModel(torch.nn.Module):
    def __init__(self, model_initiator_name):
        super().__init__()
        assert model_initiator_name in SUPPORTED_MODELS, "model name not supported"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_initiator_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_initiator_name, trust_remote_code=True)
    def forward(self, x, upsample=True):
        inputs = self.tokenizer(x, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model(**inputs,output_hidden_states=True).hidden_states[-1][:,0,:]
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

# Now this should work if model name is in SUPPORTED_MODELS
# model = BaseModel('zehui127/Omni-DNA-116M')
