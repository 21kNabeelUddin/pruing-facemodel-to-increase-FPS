import os
import shutil
import random
import argparse

def split_subset_by_identity(src_dir, train_dir, val_dir, target_train_count=100000):
    """
    Samples images identity-wise to ensure validation accuracy is meaningful.
    """
    if os.path.exists(train_dir): shutil.rmtree(train_dir)
    if os.path.exists(val_dir): shutil.rmtree(val_dir)
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    random.shuffle(classes)
    
    current_train_count = 0
    current_val_count = 0
    
    print(f"Sampling from {len(classes)} classes...")
    
    # We want to pick enough classes to hit ~target_train_count
    # To have meaningful validation, each class should have at least 2 images.
    
    for cls in classes:
        cls_path = os.path.join(src_dir, cls)
        images = os.listdir(cls_path)
        
        if len(images) < 2:
            continue
            
        # Split this class
        random.shuffle(images)
        split_idx = max(1, int(len(images) * 0.9)) # At least 1 for train
        if split_idx == len(images): # Ensure at least 1 for val if possible
            split_idx = len(images) - 1
            
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        # Copy to train
        dest_train_path = os.path.join(train_dir, cls)
        os.makedirs(dest_train_path)
        for img in train_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(dest_train_path, img))
            
        # Copy to val
        dest_val_path = os.path.join(val_dir, cls)
        os.makedirs(dest_val_path)
        for img in val_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(dest_val_path, img))
            
        current_train_count += len(train_imgs)
        current_val_count += len(val_imgs)
        
        if current_train_count >= target_train_count:
            break
            
    print(f"Sampling complete.")
    print(f"Total Train images: {current_train_count}")
    print(f"Total Val images: {current_val_count}")
    print(f"Total Identities: {len(os.listdir(train_dir))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/val')
    parser.add_argument('--train_count', type=int, default=100000)
    args = parser.parse_args()
    
    split_subset_by_identity(args.src, args.train_dir, args.val_dir, args.train_count)
