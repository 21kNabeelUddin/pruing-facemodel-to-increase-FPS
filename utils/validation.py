import torch
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate

def evaluate_lfw(model, data_loader, device='cpu', classifier=None):
    """
    Evaluate validation accuracy.
    If classifier is provided, computes classification accuracy.
    Otherwise, if labels are pairs (len batch == 3), computes verification accuracy.
    """
    model.eval()
    if classifier: classifier.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    from tqdm import tqdm
    print("Running validation...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            if len(batch) == 2: # Class based (images, labels)
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                
                features = model(images)
                if classifier:
                    outputs = classifier(features)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                else:
                    # If no classifier, we can't do much with class labels 
                    # unless we do some clustering or nearest neighbor (too slow here)
                    total += labels.size(0)
                
            elif len(batch) == 3: # Pair based (img1, img2, same/diff)
                img1, img2, label = batch
                img1, img2 = img1.to(device), img2.to(device)
                
                feat1 = model(img1)
                feat2 = model(img2)
                
                sim = torch.cosine_similarity(feat1, feat2)
                # threshold check
                pred = (sim > 0.5).long()
                correct += (pred.cpu() == label).sum().item()
                total += len(label)
    
    if total == 0:
        return 0.0
        
    acc = correct / total
    return acc

if __name__ == "__main__":
    pass
