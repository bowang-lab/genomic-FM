import torch
import pytorch_lightning as pl
from ..dataloader.iterable_dataset import IterableDataset
from ..dataloader.pl_data_module import MyDataModule
from ..dataloader.save_as_np import save_data_delta, get_cache_delta, map_to_class, has_cache
from ..dataloader.memmap_dataset_delta import MemMapDatasetDelta
from ..model_wrapper.pl_model_delta import MyLightningModuleDelta
from ..model_wrapper.linear_nn import LinearNN
from ..model_wrapper.cnn_head import CNN_Head
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


def get_memory_usage():
    # Get the memory details
    memory = psutil.virtual_memory()

    # Calculate the used memory as total - available
    total_memory_gb = memory.total / (1024 ** 3)
    available_memory_gb = memory.available / (1024 ** 3)
    used_memory_gb = total_memory_gb - available_memory_gb

    print(f"Total RAM: {total_memory_gb:.2f} GB")
    print(f"Available RAM: {available_memory_gb:.2f} GB")
    print(f"Used RAM: {used_memory_gb:.2f} GB")

def print_gpu_usage():
    """Prints the current GPU usage including memory details."""
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        total_memory_gb = total_memory / (1024**3)  # Convert total memory to GB
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        allocated_memory_gb = allocated_memory / (1024**3)  # Convert allocated memory to GB
        cached_memory = torch.cuda.memory_reserved(gpu_id)
        cached_memory_gb = cached_memory / (1024**3)  # Convert cached memory to GB

        print(f"GPU: {gpu_name}")
        print(f"Total Memory: {total_memory_gb:.2f} GB")
        print(f"Allocated Memory: {allocated_memory_gb:.2f} GB")
        print(f"Cached Memory: {cached_memory_gb:.2f} GB")
    else:
        print("CUDA is not available. Please check your setup.")


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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--logger", type=str, default="wandb", help="Logger to use (e.g. wandb, tensorboard)")
    parser.add_argument("--disk_chunk", type=int, default=2500, help="Number of chunks to split the data into for saving to disk")
    parser.add_argument("--cache_dir", type=str, default="root/data/npy_output_delta", help="Directory to save the cached embeddings")
    parser.add_argument("--mode", type=str, default="train", help="Mode to run the script (train, test)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    return args

def run_training(dataset, lr, epochs, gpus, seed, config_path, split_ratio, batch_size, num_workers, logger_name, disk_chunk, cache_dir="root/data/npy_output", cache_data_ram=True, mode="train", checkpoint=None):
    if logger_name == "wandb":
        run_name = f"Formal_{dataset}_lr={lr}_epochs={epochs}_gpus={gpus}_seed={seed}_Time={time.time()}"
        wandb_logger = WandbLogger(name=run_name, project="Run-GFM")
    else:
        wandb_logger = None
    # Load configuration file
    with open(config_path, 'r') as f:
        info = yaml.load(f, Loader=yaml.FullLoader)
    info = info[dataset]
    # Create model
    pca_components = info.pop('pca_components')
    model = CNN_Head(model_initiator_name=info.pop('model_initiator_name'),
                     output_size=info.pop('output_size'),
                     base_model_output_size=pca_components)
    task = info.pop('task')

    cache_dir = cache_dir + "_" +dataset
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    print(cache_dir)
    # save and cache the data in batches
    if not has_cache(cache_dir, dataset):
        # Create data module
        cls = getattr(data_wrapper, info.pop('class'))
        if 'num_records' in info and 'all_records' in info:
            DATA = cls(num_records=info.pop('num_records'), all_records=info.pop('all_records'))
        else:
            DATA = cls()
        data = DATA.get_data(**info)
        x_class, y_class = map_to_class(data, task, dataset,path=cache_dir)
        print(f"Mapped x_class: {x_class}")
        print(f"Mapped y_class: {y_class}")
        start_idx =  0
        end_idx = len(data)
        for i in range(0+start_idx, end_idx, disk_chunk):
            embeddings = model.cache_embed_delta_with_annotation(data[i:i+disk_chunk]) # Pre-compute embeddings for the data
            save_data_delta(embeddings, base_filename=dataset, base_index=i,pca_components=pca_components,
                            base_dir=cache_dir)
        print(">>>>End of caching")
    if cache_data_ram:
        destination_path = '/dev/shm/'
        # check is the /dev/shm/ is mounted
        if not os.path.exists(destination_path):
            print("RAM disk is not mounted, loading data to RAM at dataloader level,"
                  "this will be slower than loading to RAM disk!")
        else:
            # clear the cache directory first
            # subprocess.run(["rm", "-r", destination_path], check=True)
            subprocess.run(["cp", "-r", cache_dir, destination_path], check=True)
            cache_dir = destination_path + cache_dir.split('/')[-1]
            cache_data_ram = False
    seq1_path, annot_path, label_path = get_cache_delta(dataset, cache_dir)
    y_type = np.float32 if task == 'regression' else np.int64
    memmap_data = MemMapDatasetDelta(path_seq1=seq1_path,
                                seq_shape=(info['Seq_length'], pca_components),
                                chunk_size=info['Seq_length'],
                                annotation_paths=annot_path,
                                label_paths=label_path,
                                label_dtype=y_type)
    iterable_dataset = IterableDataset(data=memmap_data,
                                       task=task,
                                       transform=None,
                                       skip_mapping=True,
                                       load_to_ram=cache_data_ram)
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

    lightning_module = MyLightningModuleDelta(model=model, task=task, learning_rate=lr)


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
                 checkpoint=args.checkpoint)

if __name__ == "__main__":
    main()
