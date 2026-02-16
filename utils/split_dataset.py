import os
import shutil
import random
import argparse

def split_dataset(src_dir, train_dir, val_dir, split_ratio=0.9):
    """
    Splits an ImageFolder structure into train and validation sets.
    """
    if not os.path.exists(train_dir): os.makedirs(train_dir)
    if not os.path.exists(val_dir): os.makedirs(val_dir)
    
    classes = os.listdir(src_dir)
    
    for cls in classes:
        cls_path = os.path.join(src_dir, cls)
        if not os.path.isdir(cls_path): continue
        
        train_cls_path = os.path.join(train_dir, cls)
        val_cls_path = os.path.join(val_dir, cls)
        
        os.makedirs(train_cls_path, exist_ok=True)
        os.makedirs(val_cls_path, exist_ok=True)
        
        images = os.listdir(cls_path)
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        for img in train_images:
            shutil.copy(os.path.join(cls_path, img), os.path.join(train_cls_path, img))
        for img in val_images:
            shutil.copy(os.path.join(cls_path, img), os.path.join(val_cls_path, img))
            
    print(f"Split complete. Train: {train_dir}, Val: {val_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Source ms1m_arcface folder')
    parser.add_argument('--train', type=str, default='data/train')
    parser.add_argument('--val', type=str, default='data/val')
    parser.add_argument('--ratio', type=float, default=0.9)
    args = parser.parse_args()
    
    split_dataset(args.src, args.train, args.val, args.ratio)
