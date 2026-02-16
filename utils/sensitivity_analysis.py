import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.base_model import get_base_model
from models.pruned_model import get_pruned_model
from torchvision import transforms
from PIL import Image
import glob
import numpy as np

def load_images(image_folder, device):
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))
    image_paths.sort()
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return None
        
    print(f"Found {len(image_paths)} images for analysis.")
    
    # InceptionResnetV1 expects 160x160 usually, but can handle others. 
    # Facenet-pytorch typically uses fixed_image_standardization or similar.
    # standard normalization:
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    tensors = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            tensors.append(transform(img))
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    if not tensors:
        return None
        
    return torch.stack(tensors).to(device)

def analyze_sensitivity(image_folder='faces'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running sensitivity analysis on {device}...")
    
    # 1. Load Data
    images = load_images(image_folder, device)
    if images is None:
        return

    # 2. Get Baseline Embeddings
    print("Loading base model...")
    base_model = get_base_model()
    base_model.to(device)
    base_model.eval()
    
    with torch.no_grad():
        base_embeddings = base_model(images)

    # 3. Analyze Layers (InceptionResnetV1 structure)
    # repeat_1: 5 blocks
    # repeat_2: 10 blocks
    # repeat_3: 5 blocks
    
    layer_counts = {
        'repeat_1': len(base_model.repeat_1),
        'repeat_2': len(base_model.repeat_2),
        'repeat_3': len(base_model.repeat_3)
    }
    
    print("\nLayer Block Counts:", layer_counts)
    
    results = []
    
    print("Analyzing blocks...")
    for layer_name, num_blocks in layer_counts.items():
        for block_idx in range(num_blocks):
            
            skip_config = {layer_name: [block_idx]}
            pruned_model = get_pruned_model(skip_config=skip_config)
            pruned_model.to(device)
            pruned_model.eval()
            
            with torch.no_grad():
                pruned_embeddings = pruned_model(images)
            
            # Cosine Similarity
            cos_sim = torch.nn.functional.cosine_similarity(base_embeddings, pruned_embeddings)
            avg_sim = cos_sim.mean().item()
            
            results.append({
                'layer': layer_name,
                'block': block_idx,
                'similarity': avg_sim,
                'description': f"{layer_name} - Block {block_idx}"
            })
            
            print(f"Analyzed {layer_name} Block {block_idx}: Similarity = {avg_sim:.5f}")

    # 4. Generate Report
    report_path = "sensitivity_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Sensitivity Analysis Report\n\n")
        f.write("## Understanding the Score\n")
        f.write("- **Score ≈ 1.0**: The model's output changed very little when this block was removed. This means the block is **Redundant** and **Safe to Prune**.\n")
        f.write("- **Score < 0.95**: The model's output changed significantly. This means the block is **Important** and **Should NOT be Pruned**.\n\n")
        
        f.write("## Summary\n")
        f.write(f"- **Total Blocks Analyzed**: {len(results)}\n")
        f.write(f"- **Model Architecture**: InceptionResnetV1\n\n")
        
        f.write("## Full Ranking (Safest to Prune First)\n")
        f.write("| Rank | Layer | Block Index | Similarity Score (Sustainability) | Status |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|\n")
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        for i, res in enumerate(results):
            score = res['similarity']
            status = "Safe" if score > 0.985 else ("Caution" if score > 0.95 else "Critical")
            f.write(f"| {i+1} | `{res['layer']}` | {res['block']} | **{score:.5f}** | {status} |\n")
            
        f.write("\n## Top Recommendations\n")
        f.write("To improve FPS with minimal accuracy loss, consider pruning the following blocks:\n\n")
        f.write("```bash\n")
        # Generate command line args example
        prune_cmd = []
        for layer in ['repeat_2', 'repeat_3']:
            candidates = [str(r['block']) for r in results if r['layer'] == layer and r['similarity'] > 0.985]
            if candidates:
                prune_cmd.append(f"--prune_{layer} \"{','.join(candidates)}\"")
        
        f.write(f"python train.py {' '.join(prune_cmd)}\n")
        f.write("```\n")

    print(f"\nReport saved to {report_path}")
    return results

if __name__ == "__main__":
    analyze_sensitivity()
