# Face Recognition Model Optimization

This project optimizes an **InceptionResnetV1** model for real-time performance by pruning redundant blocks and fine-tuning to recover accuracy.

## 🧠 Pruning Theory: Residual Block Skipping

The model consists of three main stages: `repeat_1`, `repeat_2`, and `repeat_3`. 
- **The Concept**: We don't delete tiny weights; we **skip entire mathematical blocks** during the forward pass.
- **Why?**: Some layers in deep models are redundant. By skipping them, we avoid millions of multiplications, directly increasing **FPS** and reducing **Latency**.
- **How we pick**: We use **Sensitivity Analysis**. We compare the "Face Fingerprint" (embeddings) of the original model vs. the pruned model. If the fingerprints are >99% similar, the block is safe to prune.

---

## 🏃 Quick Start: Real-Time Demo
Compare the **Original vs. Pruned** models live using your webcam:
```bash
python utils/demo.py
```
- **Press 'p'**: Toggle Pruning (watch the FPS in the top-left corner).
- **Press 'q'**: Quit.

---

## 📂 Dataset Setup (MS1M-ArcFace)
To fine-tune the model on your PC or Kaggle, follow this structure:

1. **Download**: [MS1M-ArcFace Dataset](https://www.kaggle.com/datasets/yakhyokhuja/ms1m-arcface-dataset)
2. **Path**: Place it in `data/ms1m_arcface/`
3. **Structure**: 
   ```text
   data/ms1m_arcface/
   ├── 0/ (Images of Person 0)
   ├── 1/ (Images of Person 1)
   └── ...
   ```
*Note: The script handles the 112x112 images automatically by resizing them to 160x160 during loading.*

---

## 🛠️ Step-by-Step Optimization

### 1. Sensitivity Analysis 
Run this to see which blocks are safest for YOUR specific data:
```bash
python utils/sensitivity_analysis.py
```
Check `sensitivity_report.md` for the results.

### 2. Benchmarking
Verify the speed gain on your specific hardware (CPU/GPU):
```bash
python utils/benchmark.py
```

### 3. Fine-Tuning (Recovery)
After pruning, accuarcy drops slightly. Fine-tune to "heal" the model:
```bash
python train.py --prune_repeat_2 "6,0,4,8,2,3,9" --prune_repeat_3 "0" --data_dir "data/ms1m_arcface" --epochs 10
```---

## 📊 Evaluation & Accuracy

### Baseline Accuracy
The original InceptionResnetV1 model (pretrained on VGGFace2) achieves approximately **~99.0% accuracy** on the standard LFW (Labeled Faces in the Wild) benchmark.
- **Your Target**: Within 3-5% of original (**~94% to 96%**).

### How to Check Accuracy
After running `train.py`, the script will automatically run a validation pass at the end of each epoch and print the `Validation Acc`. 
- **Validation Dataset**: Use the "Testing Subset" described below.
- **Result**: If your validation accuracy is >95%, you have successfully "heal" the pruned model!

---

## ✂️ Preparing Your Dataset (Split)

Face recognition datasets are huge. For good results, we recommend:
- **Images**: Fine-tune on at least **50,000 to 100,000 images**.
- **Epochs**: **5 to 10 epochs** using a GPU.

To create training and testing subsets on your PC:
1. Run the splitting utility:
   ```bash
   python utils/split_dataset.py --src "data/ms1m_arcface" --train "data/train" --val "data/val" --ratio 0.9
   ```
   *This will put 90% of images in `data/train` and 10% in `data/val`.*

2. Use these folders for training:
   ```bash
   python train.py --data_dir "data/train" --epochs 10
   ```

---

## 📦 Project Structure
- `models/`: `base_model.py` (facenet-pytorch loader) and `pruned_model.py` (skipping logic).
- `data/`: `dataset.py` (auto-resizing loader).
- `utils/`: 
  - `benchmark.py` (speed test)
  - `demo.py` (webcam view)
  - `sensitivity_analysis.py` (safety ranking)
  - `split_dataset.py` (prepare train/val sets)
- `requirements.txt`: Environment dependencies.
