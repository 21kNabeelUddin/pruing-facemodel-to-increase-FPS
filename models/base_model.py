import torch
import torch.nn as nn
from insightface.recognition.arcface_torch.backbones import get_model

def get_base_model(name='r50', fp16=False):
    """
    Load the pre-trained AdaFace/ArcFace model.
    Defaults to ResNet50 (IR-50).
    """
    # Using insightface's backbone directly
    # 'r50' corresponds to IResNet50
    model = get_model(name, fp16=fp16)
    return model

if __name__ == "__main__":
    try:
        model = get_base_model()
        print(f"Successfully loaded {model.__class__.__name__}")
        print(model)
    except Exception as e:
        print(f"Error loading model: {e}")
