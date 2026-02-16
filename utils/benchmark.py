import torch
import time
import numpy as np

def benchmark_fps(model, input_size=(1, 3, 112, 112), device='cpu', warmups=10, runs=100):
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
    from models.base_model import get_base_model
    from models.pruned_model import get_pruned_model
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("--- Original IR50 ---")
    model = get_base_model()
    benchmark_fps(model, device=device)
    
    print("\n--- Pruned IR50 (Example: Skipping some blocks) ---")
    # Example pruning: skipping 2 blocks in layer2 and layer3
    skip_config = {'layer2': [0, 2], 'layer3': [1, 3, 5]}
    pruned_model = get_pruned_model(skip_config=skip_config)
    benchmark_fps(pruned_model, device=device)
