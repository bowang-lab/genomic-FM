import torch
import pytorch_lightning as pl
from ..dataloader.iterable_dataset import IterableDataset
from ..dataloader.pl_data_module import MyDataModule
from ..dataloader.save_as_np import save_data_delta, get_cache_delta, map_to_class, has_cache
from ..dataloader.memmap_dataset_delta import MemMapDatasetDelta
from ..dataloader.memmap_dataset import MemMapDataset
from ..dataloader.data_wrapper import ClinVarDataWrapper

from .pl_module import MyLightningModuleHeart
from ..model_wrapper.linear_nn import LinearNN
from .cnn_head import CNN_Head
from .base_model import BaseModel
import argparse
from torch.utils.data import random_split
import yaml
from ..dataloader import data_wrapper as data_wrapper
from pytorch_lightning.loggers import WandbLogger
import psutil
import os
import time
import subprocess
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lightning Training Script")
    parser.add_argument("--dataset", type=str, default="heart_test_finetune", help="Task to perform (e.g. clivar, geneko, eqtl, stql, etc.)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (0 for CPU)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml", help="Path to configuration file")
    parser.add_argument("--split_ratio", type=list, default=[0.8, 0.1, 0.1], help="Train, validation, test split ratio")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of workers for data loading")
    parser.add_argument("--logger", type=str, default="wandb", help="Logger to use (e.g. wandb, tensorboard)")
    parser.add_argument("--disk_chunk", type=int, default=2500, help="Number of chunks to split the data into for saving to disk")
    parser.add_argument("--cache_dir", type=str, default="root/data/npy_output_delta", help="Directory to save the cached embeddings")
    parser.add_argument("--mode", type=str, default="train", help="Mode to run the script (train, test)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--project", type=str, default="Run-GFM", help="Name of the project in wandb")
    args = parser.parse_args()
    return args

def run_training(dataset, lr, epochs, gpus, seed, config_path, split_ratio, batch_size, num_workers, logger_name, disk_chunk, cache_dir="root/data/npy_output", cache_data_ram=True, mode="train", checkpoint=None, project=None):
    ### Get data ###
    ALL_RECORDS = True
    NUM_RECORDS = 30000
    SEQ_LEN = 256
    task = 'classification'
    data_loader = ClinVarDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
    data = data_loader.get_data(Seq_length=SEQ_LEN,target='CLNDN',disease_subset=True)
    ###############

    if logger_name == "wandb":
        run_name = f"Formal_{dataset}_lr={lr}_epochs={epochs}_gpus={gpus}_seed={seed}_Time={time.time()}"
        wandb_logger = WandbLogger(name=run_name, project=project)
    else:
        wandb_logger = None
    # Load configuration file
    with open(config_path, 'r') as f:
        info = yaml.load(f, Loader=yaml.FullLoader)
    # Create model
    info = info[dataset]
    iterable_dataset = IterableDataset(data=data,
                                       task=task,
                                       transform=None,
                                       skip_mapping=False)
    model = CNN_Head(model_initiator_name=info.pop('model_initiator_name'),
                     output_size=info.pop('output_size'))
    # return iterable_dataset,model
    # split the data
    print("split_ratio: ", split_ratio)
    train_data, val_data, test_data = random_split(iterable_dataset,
                                                   split_ratio,
                                                   generator=torch.Generator().manual_seed(seed))
    print("Train data size: ", len(train_data), "split ratio: ", split_ratio, "Total data size: ", len(iterable_dataset))
    data_module = MyDataModule(train_data=train_data, val_data=val_data,
                               test_data=test_data, batch_size=batch_size,
                               num_workers=num_workers, transform=None)
    # Initialize your Lightning Module
    lightning_module = MyLightningModuleHeart(model=model, task=task, learning_rate=lr)


    trainer_args = {
        'max_epochs': epochs,
        'logger': wandb_logger
    }
    if gpus >= 1:
        trainer_args['accelerator'] = 'gpu'
        trainer_args['devices'] = gpus
        trainer_args['strategy'] = 'ddp'
    else:
        trainer_args['accelerator'] = 'cpu'
    # Create the Trainer
    if mode == "train":
        trainer = pl.Trainer(**trainer_args)
        trainer.fit(lightning_module, data_module)
        trainer.test(lightning_module, data_module)
    elif mode == "test":
        lightning_module = MyLightningModuleDelta.load_from_checkpoint(model=model, checkpoint_path=checkpoint)
        trainer = pl.Trainer(**trainer_args)
        trainer.test(lightning_module, data_module)
    elif mode == "random":
        trainer = pl.Trainer(**trainer_args)
        trainer.test(lightning_module, data_module)


def main():
    args = parse_args()
    return run_training(args.dataset,
                 args.lr,
                 args.epochs,
                 args.gpus,
                 args.seed,
                 args.config,
                 args.split_ratio,
                 args.batch_size,
                 args.num_workers,
                 args.logger,
                 args.disk_chunk,
                 args.cache_dir,
                 mode=args.mode,
                 checkpoint=args.checkpoint,
                 project=args.project)

if __name__ == "__main__":
    main()
