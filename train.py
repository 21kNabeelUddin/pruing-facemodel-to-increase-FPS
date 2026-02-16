import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from models.pruned_model import get_pruned_model
from data.dataset import get_dataloader
from utils.validation import evaluate_lfw
import time

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    # Determine which layers to skip
    skip_config = {}
    if args.prune_layer2:
        skip_config['layer2'] = [int(x) for x in args.prune_layer2.split(',')]
    if args.prune_layer3:
         skip_config['layer3'] = [int(x) for x in args.prune_layer3.split(',')]
         
    model = get_pruned_model(skip_config=skip_config)
    model.to(device)
    
    # 2. Data
    train_loader = get_dataloader(args.data_dir, batch_size=args.batch_size)
    
    if train_loader is None:
        print("No training data found. Exiting training loop.")
        return

    # 3. Loss & Optimizer
    # ArcFace loss usually requires a specific header (ArcFace/CosFace) which has learnable weights.
    # For fine-tuning with limited data, we might just use CrossEntropy on the features if we add a linear layer,
    # or just simple triplet loss. 
    # Here we assume we might want to fine-tune the backbone.
    
    # Setup simple CrossEntropy with a linear head for class-based training
    num_classes = len(train_loader.dataset.dataset.classes)
    classifier = nn.Linear(512, num_classes).to(device) # Assuming 512 embedding dim
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': model.parameters()},
        {'params': classifier.parameters()}
    ], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # 4. Loop
    for epoch in range(args.epochs):
        model.train()
        classifier.train()
        
        running_loss = 0.0
        start = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            embeddings = model(inputs)
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        epoch_time = time.time() - start
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s")
        
        # Validation
        acc = evaluate_lfw(model, train_loader, device=device) # Using train_loader as placeholder
        print(f"Validation Acc: {acc:.4f}")
        
    # Save
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/pruned_epoch_{args.epochs}.pth')
    print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--prune_layer2', type=str, default='', help='Indices to skip in layer2, comma separated')
    parser.add_argument('--prune_layer3', type=str, default='', help='Indices to skip in layer3, comma separated')
    
    args = parser.parse_args()
    train(args)
