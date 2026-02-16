import torch
from models.base_model import get_base_model
import sys

def inspect():
    try:
        model = get_base_model('r50')
        with open('model_structure.txt', 'w') as f:
            print(model, file=f)
            
        print("Model structure saved to model_structure.txt")
        
        # Also print layer names to console for quick verification
        print("Model Layers:")
        for name, module in model.named_children():
            print(f"- {name}: {module.__class__.__name__}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
