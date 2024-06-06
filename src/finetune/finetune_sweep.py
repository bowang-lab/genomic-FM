from .finetune import run_training
import wandb
import argparse
import yaml

def run():
    run = wandb.init()
    run_training(
                 dataset="clinvar_CLNDN_hyena-tiny",
                #  dataset="clinvar_CLNDN_hyena-tiny",
                #  dataset="clinvar_CLNDN_hyena-tiny",
                #  dataset="clinvar_CLNDN_hyena-tiny",
                 lr=wandb.config.lr,
                 epochs=100,
                 gpus=1,
                 seed=42,
                 config_path="configs/finetune_hyena.yaml",
                #  config_path="configs/finetune_dnabert2.yaml",
                #  config_path="configs/finetune_nt.yaml",
                #  config_path="configs/finetune_ntv2.yaml",
                 split_ratio=[0.8, 0.1, 0.1],
                 batch_size=wandb.config.batch_size,
                 num_workers=8,
                 logger_name="wandb",
                 disk_chunk=2500,
                 cache_dir="root/data/npy_output_delta",
                 model='train',
                 checkpoint=None)


def main():
    with open("configs/finetune_sweep.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    sweep_id = wandb.sweep(sweep=config, project="Genomic-FM")
    wandb.agent(sweep_id, function=run, count=5)

if __name__ == "__main__":
    main()
