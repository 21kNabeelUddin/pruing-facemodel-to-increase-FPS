# Face Recognition Model Optimization

This project optimizes a pre-trained Face Recognition model (AdaFace IR50) for real-time performance by pruning layers and fine-tuning.

## Structure
- `models/`: Contains `base_model.py` (loads pre-trained) and `pruned_model.py` (skips layers).
- `data/`: `dataset.py` for loading face images (ImageFolder structure).
- `utils/`: `benchmark.py` for FPS testing, `validation.py` for evaluation.
- `train.py`: Script for fine-tuning the pruned model.

## Setup
1. **Environment**:
   ```bash
   conda activate general_purpose
   pip install -r requirements.txt
   ```
   *Note: Requires Visual C++ Build Tools for `insightface` and `cython` on Windows.*

2. **Data**:
   Prepare your dataset in `data/your_dataset/` with subfolders for each identity.
   ```
   data/
       ms1mv2/
           id1/
               img1.jpg
           id2/
               img1.jpg
   ```

## Usage

### 1. Benchmark Original vs Pruned
Measure the inference speed difference.
```bash
python utils/benchmark.py
```
*Modify `utils/benchmark.py` to change pruning configuration.*

### 2. Fine-Tuning
Retrain the model after pruning (or skipping layers) to recover accuracy.
```bash
python train.py --data_dir "data/ms1mv2" --epochs 10 --prune_layer2 "0,2" --prune_layer3 "1,3"
```
- `--prune_layer2 "0,2"`: Skips block 0 and 2 in layer 2.

### 3. Validation
Validation runs automatically during training. To run separately (placeholder):
```bash
python -c "from utils.validation import evaluate_lfw; evaluate_lfw(model, loader)"
```

## Pruning Strategy
We use **Block Skipping**. Instead of pruning individual weights, whole residual blocks are skipped during forward pass. This reduces depth and increases FPS directly.
