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
    if args.prune_repeat_1:
         skip_config['repeat_1'] = [int(x) for x in args.prune_repeat_1.split(',')]
    if args.prune_repeat_2:
        skip_config['repeat_2'] = [int(x) for x in args.prune_repeat_2.split(',')]
    if args.prune_repeat_3:
         skip_config['repeat_3'] = [int(x) for x in args.prune_repeat_3.split(',')]
         
    model = get_pruned_model(skip_config=skip_config)
    model.to(device)
    
    # 2. Data
    train_loader = get_dataloader(args.data_dir, batch_size=args.batch_size)
    
    if train_loader is None:
        print("No training data found. Exiting training loop.")
        return

    # 3. Loss & Optimizer
    # For classification
    num_classes = len(train_loader.dataset.dataset.classes)
    
    # InceptionResnetV1 output dim is 512
    # We add a classifier head on top of the embeddings
    classifier = nn.Linear(512, num_classes).to(device) 
    
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
        if len(inputs) > 1: # Basic check
             acc = evaluate_lfw(model, train_loader, device=device)
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
    parser.add_argument('--prune_repeat_1', type=str, default='', help='Indices to skip in repeat_1, comma separated')
    parser.add_argument('--prune_repeat_2', type=str, default='', help='Indices to skip in repeat_2, comma separated')
    parser.add_argument('--prune_repeat_3', type=str, default='', help='Indices to skip in repeat_3, comma separated')
    
    args = parser.parse_args()
    train(args)
