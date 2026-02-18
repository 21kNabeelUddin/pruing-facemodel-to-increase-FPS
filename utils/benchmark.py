import torch
import time
import numpy as np
import sys
import os
import argparse

# Ensure we can import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def benchmark_fps(model, input_size=(1, 3, 160, 160), device='cpu', warmups=10, runs=100):
    model.to(device)
    model.eval()
    
    input_tensor = torch.randn(input_size).to(device)
    
    print(f"Benchmarking on {device}...")
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(input_tensor)
            
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device != 'cpu':
                torch.cuda.synchronize()
            start = time.time()
            
            _ = model(input_tensor)
            
            if device != 'cpu':
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
            
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"Average batch time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    
    return fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=160, help='Image size (160 for InceptionResnet)')
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--prune_repeat_2', type=str, default='6,0,4,8,2,3,9')
    parser.add_argument('--prune_repeat_3', type=str, default='0')
    
    args = parser.parse_args()
    
    from models.base_model import get_base_model
    from models.pruned_model import get_pruned_model
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size = (1, 3, args.input_size, args.input_size)
    
    print("--- Original InceptionResnetV1 ---")
    model = get_base_model()
    benchmark_fps(model, input_size=size, device=device, warmups=args.warmups, runs=args.runs)
    
    print("\n--- Pruned InceptionResnetV1 (Configured) ---")
    skip_config = {}
    if args.prune_repeat_2:
        skip_config['repeat_2'] = [int(x) for x in args.prune_repeat_2.split(',')]
    if args.prune_repeat_3:
        skip_config['repeat_3'] = [int(x) for x in args.prune_repeat_3.split(',')]
        
    pruned_model = get_pruned_model(skip_config=skip_config)
    benchmark_fps(pruned_model, input_size=size, device=device, warmups=args.warmups, runs=args.runs)
