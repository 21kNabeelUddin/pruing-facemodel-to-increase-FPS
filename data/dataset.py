import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, organized in subdirectories by class.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Using ImageFolder to handle class indexing automatically
        # Ensure the directory structure is root/class_name/image.jpg
        if os.path.exists(root_dir):
            self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
            self.classes = self.dataset.classes
        else:
            print(f"Warning: Dataset directory {root_dir} not found. Creating empty dataset.")
            self.dataset = []
            self.classes = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def get_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=4):
    # Standard preprocessing for ArcFace/AdaFace (112x112, normalized to [-1, 1])
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = FaceDataset(root_dir=data_dir, transform=transform)
    
    if len(dataset) == 0:
        return None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
