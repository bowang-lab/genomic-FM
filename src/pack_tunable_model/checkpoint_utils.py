import os
import json
import torch

def load_checkpoint_into_model(model, checkpoint_path):
    """
    Load checkpoint weights into a model. Automatically finds best checkpoint if available.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to model directory containing checkpoints or pytorch_model.bin
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return

    # Check if checkpoint_path has subdirectories with checkpoints
    checkpoint_dirs = [d for d in os.listdir(checkpoint_path)
                      if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_path, d))]

    if checkpoint_dirs:
        # Try to find best checkpoint from trainer_state.json
        best_step = None
        best_metric_value = float('-inf')

        # Look for best_model_checkpoint in trainer_state.json
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r') as f:
                state = json.load(f)
                best_checkpoint = state.get('best_model_checkpoint')
                if best_checkpoint:
                    best_step = int(os.path.basename(best_checkpoint).split('-')[1])
                    print(f"Using best checkpoint from trainer_state: step {best_step}")

        # If no best checkpoint found, use latest
        if not best_step:
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
            best_step = int(checkpoint_dirs[-1].split('-')[1])
            print(f"Using latest checkpoint: step {best_step}")

        weights_path = os.path.join(checkpoint_path, f"checkpoint-{best_step}", "pytorch_model.bin")
    elif os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
        weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        print(f"Loading weights from {weights_path}")
    else:
        print(f"Warning: No pytorch_model.bin found in {checkpoint_path}")
        return

    # Load weights
    checkpoint_state = torch.load(weights_path, map_location="cpu")
    try:
        if isinstance(checkpoint_state, dict) and 'model_state_dict' in checkpoint_state:
            model.load_state_dict(checkpoint_state['model_state_dict'], strict=False)
        elif isinstance(checkpoint_state, dict) and 'state_dict' in checkpoint_state:
            model.load_state_dict(checkpoint_state['state_dict'], strict=False)
        else:
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint_state.items()}
            model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded checkpoint weights")
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        print("Continuing with base model weights only")
