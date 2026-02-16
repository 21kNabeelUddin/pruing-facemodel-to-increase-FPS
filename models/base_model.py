import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

def get_base_model(pretrained='vggface2', fp16=False):
    """
    Load the pre-trained InceptionResnetV1 model.
    """
    print(f"Loading InceptionResnetV1 (pretrained={pretrained})...")
    model = InceptionResnetV1(pretrained=pretrained, classify=False)
    
    if fp16:
        model.half()
        
    return model

if __name__ == "__main__":
    try:
        model = get_base_model()
        print(f"Successfully loaded {model.__class__.__name__}")
        # print(model) # Uncomment to see full structure
    except Exception as e:
        print(f"Error loading model: {e}")
