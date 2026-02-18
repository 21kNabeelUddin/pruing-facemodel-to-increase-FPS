import torch
import torch.nn as nn
from models.pruned_model import get_pruned_model
from data.dataset import get_dataloader
import os

def verify_checkpoint(checkpoint_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verifying checkpoint: {checkpoint_path} on {device}")
    
    # 1. Setup Model (Same config as train)
    skip_config = {
        'repeat_2': [6, 0, 4, 8, 2, 3, 9],
        'repeat_3': [0]
    }
    model = get_pruned_model(skip_config=skip_config)
    
    # 2. Setup Data
    val_loader = get_dataloader(data_dir, batch_size=64, shuffle=False)
    if val_loader is None:
        print(f"No validation data found in {data_dir}")
        return

    num_classes = len(val_loader.dataset.dataset.classes)
    classifier = nn.Linear(512, num_classes) # We don't have the saved classifier head, so we'll just check forward pass
    
    # Load state dict
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model backbone loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    # Without the classifier head (which wasn't saved in train.py), 
    # we can't get classification acc directly unless we retrain the head 
    # or rely on verification accuracy (similarity).
    # Since we can't retrain the head easily here, we'll do a quick consistency check.
    
    print("Running forward pass consistency check...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            feats = model(images)
            print(f"Batch {i} processed. Feature shape: {feats.shape}")
            if i >= 2: break # Just a check

    print("Consistency check complete. Model is functional and producing embeddings.")
    print("NOTE: train.py only saved the backbone, not the classifier head. "
          "To get classification accuracy, you should run train.py with the fixed reporting logic.")

if __name__ == "__main__":
    verify_checkpoint('checkpoints/pruned_epoch_10.pth', 'data/val')
