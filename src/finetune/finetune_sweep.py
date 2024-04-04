from .finetune import run_training
import wandb
import argparse
import yaml

def run():
    run = wandb.init()
    run_training(dataset=wandb.config.dataset,
                 lr=wandb.config.lr,
                 epochs=wandb.config.epochs,
                 config_path=wandb.config.config_path,
                 batch_size=wandb.config.batch_size,
                 gpus=0,
                 seed=42,
                 split_ratio=[0.8, 0.1, 0.1],
                 num_workers=0,
                 logger_name="wandb")

def main():
    with open("configs/finetune_sweep.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    sweep_id = wandb.sweep(sweep=config, project="Genomic-FM")
    wandb.agent(sweep_id, function=run, count=5)

if __name__ == "__main__":
    main()
