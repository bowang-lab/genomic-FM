import torch
import pytorch_lightning as pl
from ..dataloader.iterable_dataset import IterableDataset
from ..dataloader.pl_data_module import MyDataModule
from ..model_wrapper.pl_model import MyLightningModule
from ..model_wrapper.linear_nn import LinearNN
import argparse
from torch.utils.data import random_split
import yaml
from ..dataloader import data_wrapper as data_wrapper


# Parsing command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lightning Training Script")
    parser.add_argument("--dataset", type=str, default="clinvar_CLNSIG_hyena-tiny", help="Task to perform (e.g. clivar, geneko, eqtl, stql, etc.)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use (0 for CPU)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml", help="Path to configuration file")
    parser.add_argument("--split_ratio", type=list, default=[0.8, 0.1, 0.1], help="Train, validation, test split ratio")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load configuration file
    with open(args.config, 'r') as f:
        info = yaml.load(f, Loader=yaml.FullLoader)
    info = info[args.dataset]

    # Create data module
    #TODO add cls
    cls = getattr(data_wrapper, info.pop('class'))
    DATA = cls()
    data = DATA.get_data(target=info['target'])
    iterable_dataset = IterableDataset(data=data,
                                       task=info['task'],
                                       transform=None)
    # split the data
    train_data, val_data, test_data = random_split(iterable_dataset,
                                                   args.split_ratio,
                                                   generator=torch.Generator().manual_seed(args.seed))

    data_module = MyDataModule(train_data=train_data, val_data=val_data,
                               test_data=test_data, batch_size=32,
                               num_workers=0, transform=None)
    # Initialize your Lightning Module

    model = LinearNN(model_initiator_name=info['model_initiator_name'],
                     output_size=info['output_size'])
    lightning_module = MyLightningModule(model=model, task=info['task'], learning_rate=args.lr)

    # Create the Trainer
    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator='ddp' if args.gpus > 1 else 'cpu')
    trainer.fit(lightning_module, data_module)

if __name__ == "__main__":
    main()
