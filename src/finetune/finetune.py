import torch
import pytorch_lightning as pl
from ..dataloader.iterable_dataset import IterableDataset
from ..dataloader.pl_data_module import MyDataModule
from ..dataloader.save_as_np import save_data, get_cache, map_to_class, get_mapped_class, map_to_given_class
from ..dataloader.memmap_dataset import MemMapDataset
from ..dataloader.efficient_iteratable_dataset import EffIterableDataset
from ..model_wrapper.pl_model import MyLightningModule
from ..model_wrapper.linear_nn import LinearNN
import argparse
from torch.utils.data import random_split
import yaml
from ..dataloader import data_wrapper as data_wrapper
from pytorch_lightning.loggers import WandbLogger


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
    parser.add_argument("--disk_chunk", type=int, default=10000, help="Number of chunks to split the data into for saving to disk")
    args = parser.parse_args()
    return args

def run_training(dataset, lr, epochs, gpus, seed, config_path, split_ratio, batch_size, num_workers, logger_name, disk_chunk=10000):
    if logger_name == "wandb":
        run_name = f"{dataset}_lr={lr}_epochs={epochs}_gpus={gpus}_seed={seed}"
        wandb_logger = WandbLogger(name=run_name, project="Genomic-FM")
    else:
        wandb_logger = None
    # Load configuration file
    with open(config_path, 'r') as f:
        info = yaml.load(f, Loader=yaml.FullLoader)
    info = info[dataset]
    # Create model
    model = LinearNN(model_initiator_name=info.pop('model_initiator_name'),
                     output_size=info.pop('output_size'))
    task = info.pop('task')
    # Create data module
    cls = getattr(data_wrapper, info.pop('class'))
    DATA = cls()
    data = DATA.get_data(**info)

    x_class, y_class = get_mapped_class(data, task)
    # save and cache the data in batches
    for i in range(0, len(data), disk_chunk):
        embeddings = model.cache_embed(data[i:i+disk_chunk]) # Pre-compute embeddings for the data
        map_to_given_class(embeddings, x_class, y_class, task)
        save_data(embeddings, base_filename=dataset, base_index=i)

    seq1_path, seq2_path, annot_path, label_path = get_cache(dataset)
    memmap_data = MemMapDataset(path_seq1=seq1_path,
                                path_seq2=seq2_path,
                                seq_shape=(info['Seq_length'], 128),
                                chunk_size=info['Seq_length'],
                                annotation_paths=annot_path,
                                label_paths=label_path)
    iterable_dataset = IterableDataset(data=memmap_data,
                                       task=task,
                                       transform=None,
                                       skip_mapping=True)
    # split the data
    train_data, val_data, test_data = random_split(iterable_dataset,
                                                   split_ratio,
                                                   generator=torch.Generator().manual_seed(seed))
    print("Train data size: ", len(train_data), "split ratio: ", split_ratio, "Total data size: ", len(iterable_dataset))
    data_module = MyDataModule(train_data=train_data, val_data=val_data,
                               test_data=test_data, batch_size=batch_size,
                               num_workers=num_workers, transform=None)
    # Initialize your Lightning Module

    lightning_module = MyLightningModule(model=model, task=task, learning_rate=lr)

    trainer_args = {
        'max_epochs': epochs,
        'logger': wandb_logger
    }
    if gpus > 1:
        trainer_args['accelerator'] = 'ddp'
        trainer_args['devices'] = gpus
    else:
        trainer_args['accelerator'] = 'cpu'
    # Create the Trainer
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(lightning_module, data_module)


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
                 args.logger)

if __name__ == "__main__":
    main()
