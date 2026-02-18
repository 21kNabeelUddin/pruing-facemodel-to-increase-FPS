import torch
import torch.nn as nn
import sys
import os
import argparse
import glob
import numpy as np
from torchvision import transforms
from PIL import Image

# Ensure we can import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.base_model import get_base_model
from models.pruned_model import get_pruned_model

def load_images(image_folder, device):
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))
    image_paths.sort()
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return None
        
    print(f"Found {len(image_paths)} images for analysis.")
    
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

def analyze_sensitivity(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running sensitivity analysis on {device}...")
    
    # 1. Load Data
    images = load_images(args.image_folder, device)
    if images is None:
        return

    # 2. Get Baseline Embeddings
    print("Loading base model...")
    base_model = get_base_model()
    base_model.to(device).eval()
    
    with torch.no_grad():
        base_embeddings = base_model(images)

    # 3. Analyze Layers
    layer_counts = {
        'repeat_1': len(base_model.repeat_1),
        'repeat_2': len(base_model.repeat_2),
        'repeat_3': len(base_model.repeat_3)
    }
    
    results = []
    print("Analyzing blocks...")
    for layer_name, num_blocks in layer_counts.items():
        for block_idx in range(num_blocks):
            skip_config = {layer_name: [block_idx]}
            pruned_model = get_pruned_model(skip_config=skip_config)
            pruned_model.to(device).eval()
            
            with torch.no_grad():
                pruned_embeddings = pruned_model(images)
            
            cos_sim = torch.nn.functional.cosine_similarity(base_embeddings, pruned_embeddings)
            avg_sim = cos_sim.mean().item()
            
            results.append({
                'layer': layer_name,
                'block': block_idx,
                'similarity': avg_sim
            })
            print(f"Analyzed {layer_name} Block {block_idx}: Similarity = {avg_sim:.5f}")

    # 4. Generate Report
    report_path = args.output_report
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
        prune_cmd = []
        for layer in ['repeat_2', 'repeat_3']:
            candidates = [str(r['block']) for r in results if r['layer'] == layer and r['similarity'] > 0.985]
            if candidates:
                prune_cmd.append(f"--prune_{layer} \"{','.join(candidates)}\"")
        
        f.write(f"python train.py {' '.join(prune_cmd)}\n")
        f.write("```\n")

    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='faces', help='Folder with sample images')
    parser.add_argument('--output_report', type=str, default='sensitivity_report.md')
    args = parser.parse_args()
    analyze_sensitivity(args)
