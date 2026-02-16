# Sensitivity Analysis Report

## Understanding the Score
- **Score ≈ 1.0**: The model's output changed very little when this block was removed. This means the block is **Redundant** and **Safe to Prune**.
- **Score < 0.95**: The model's output changed significantly. This means the block is **Important** and **Should NOT be Pruned**.

## Summary
- **Total Blocks Analyzed**: 20
- **Model Architecture**: InceptionResnetV1

## Full Ranking (Safest to Prune First)
| Rank | Layer | Block Index | Similarity Score (Sustainability) | Status |
|:---:|:---:|:---:|:---:|:---:|
| 1 | `repeat_3` | 0 | **0.99303** | Safe |
| 2 | `repeat_2` | 6 | **0.98959** | Safe |
| 3 | `repeat_2` | 0 | **0.98935** | Safe |
| 4 | `repeat_2` | 4 | **0.98782** | Safe |
| 5 | `repeat_2` | 8 | **0.98655** | Safe |
| 6 | `repeat_2` | 2 | **0.98646** | Safe |
| 7 | `repeat_2` | 3 | **0.98618** | Safe |
| 8 | `repeat_2` | 9 | **0.98567** | Safe |
| 9 | `repeat_3` | 2 | **0.98285** | Caution |
| 10 | `repeat_2` | 7 | **0.98106** | Caution |
| 11 | `repeat_2` | 5 | **0.98094** | Caution |
| 12 | `repeat_3` | 1 | **0.97678** | Caution |
| 13 | `repeat_2` | 1 | **0.97520** | Caution |
| 14 | `repeat_3` | 4 | **0.96886** | Caution |
| 15 | `repeat_3` | 3 | **0.95925** | Caution |
| 16 | `repeat_1` | 2 | **0.94155** | Critical |
| 17 | `repeat_1` | 4 | **0.92651** | Critical |
| 18 | `repeat_1` | 3 | **0.91553** | Critical |
| 19 | `repeat_1` | 0 | **0.89206** | Critical |
| 20 | `repeat_1` | 1 | **0.81330** | Critical |

## Top Recommendations
To improve FPS with minimal accuracy loss, consider pruning the following blocks:

```bash
python train.py --prune_repeat_2 "6,0,4,8,2,3,9" --prune_repeat_3 "0"
```
