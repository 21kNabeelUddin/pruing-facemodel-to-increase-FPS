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
```

## 📦 Project Structure
- `models/`: `base_model.py` (facenet-pytorch loader) and `pruned_model.py` (skipping logic).
- `data/`: `dataset.py` (auto-resizing loader).
- `utils/`: `benchmark.py` (speed test), `demo.py` (webcam), `sensitivity_analysis.py` (safety ranking).
- `requirements.txt`: Environment dependencies.
