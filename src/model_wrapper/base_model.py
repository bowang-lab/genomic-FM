import yaml
import torch
import importlib
import os
import sys
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
