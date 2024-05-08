import yaml
import torch
import importlib
import os
import sys
from tqdm import tqdm
import numpy as np

SUPORTED_MODELS = ['dnabert2', 'dnabert6','gena-lm-bigbird-base-t2t',
                   'gena-lm-bert-large-t2', 'hyenadna-large-1m',
                   'hyenadna-tiny-1k',
                   'hyenadna-small-32k',
                   'hyenadna-medium-160k',
                   'hyenadna-medium-450k',
                   'nt_transformer_ms',
                   'nt_transformer_human_ref',
                   'nt_transformer_1000g',
                   'nt_transformer_v2_500m',
                   'grover']
#TODO evo, caduceus

class BaseModel(torch.nn.Module):
    def __init__(self, model_initiator_name):
        super().__init__()
        self.model_initiator_name = model_initiator_name
        # check model initiator name is supported
        if self.model_initiator_name not in SUPORTED_MODELS:
            raise ValueError(f"Model initiator name {self.model_initiator_name} is not supported. "
                             f"Please use one of the following: {SUPORTED_MODELS}")
        self.model = self._load_model()

    def _load_model(self):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(parent_dir)
        bend_dir = os.path.join(src_dir, 'BEND')
        sys.path.append(bend_dir)
        model_config = os.path.join(bend_dir, "conf/embedding/embed.yaml")
        if not os.path.exists(model_config):
            raise FileNotFoundError(f"Model config file not found at {model_config}"
                                        "If you have not included the submodule BEND, please"
                                        "Use 'git submodule update --init --recursive' to clone the submodule.")
        with open(model_config, 'r') as file:
            config = yaml.safe_load(file)
            model_info = config[self.model_initiator_name]

        target = model_info['_target_']
        module_name, class_name = target.rsplit('.', 1)

        # Dynamically import the module and class
        module = importlib.import_module(module_name)
        ModelClass = getattr(module, class_name)
        # Prepend the model directory to the model args
        model_args = {k: v for k, v in model_info.items() if k != '_target_'}
        embedders_dir = "./root/models"
        modified_path = model_args['model_path'].format(embedders_dir=embedders_dir)
        # remove dollar sign
        model_args['model_path'] = modified_path[1:]
        print(f"Model args: {model_args}")
        return ModelClass(**model_args)

    def forward(self, x):
        return self.model.embed(x)

    def cache_embed(self, data):
        new_data = []
        for i in tqdm(range(len(data)), desc="Caching embeddings"):
            x, y = data[i]
            seq1, seq2 = self.model(x[0]), self.model(x[1])
            new_data.append([[seq1,seq2,x[2]],y])
        return new_data

    def cache_embed_delta_with_annotation(self,data):
        new_data = []
        for x, y in tqdm(data, desc="Caching embeddings"):
            seq1, seq2 = self.model(x[0]), self.model(x[1])
            new_data.append([[seq1-seq2,x[2]],y])
        return new_data

    def cache_embed_delta(self, data, pca_components=16):
        # Step 1: Collect all differences
        differences = []
        labels = []
        for x, y in tqdm(data, desc="Caching embeddings"):
            seq1, seq2 = self.model(x[0]), self.model(x[1])
            differences.append(seq2 - seq1)
            labels.append(y)

        # Step 2: Convert list to tensor
        differences_tensor = np.stack(differences)

        # Step 3: Apply PCA to the collected differences
        reduced_data = apply_pca_torch(differences_tensor, n_components=pca_components)

        return reduced_data, labels


def apply_pca_torch(data, n_components=16):
    # Ensure data is a torch tensor and optionally move data to GPU
    data_tensor = torch.tensor(data).float()
    if torch.cuda.is_available():
        data_tensor = data_tensor.cuda()

    # Reshape the data
    reshaped_data = data_tensor.reshape(-1, data_tensor.shape[-1])  # Reshape to (num_samples * 1024, 128)

    # Center the data by subtracting the mean
    mean = torch.mean(reshaped_data, dim=0)
    data_centered = reshaped_data - mean

    # Compute SVD
    U, S, V = torch.svd(data_centered)

    # Compute PCA by projecting the data onto the principal components
    pca_transformed_data = torch.mm(data_centered, V[:, :n_components])

    # Reshape back to the original shape
    pca_transformed_data = pca_transformed_data.reshape(data_tensor.shape[:-1] + (-1,))  # Reshape back to (num_samples, 1024, n_components)

    # Optionally move data back to CPU
    pca_transformed_data = pca_transformed_data.cpu()

    return pca_transformed_data.numpy()  # Convert to numpy if needed
