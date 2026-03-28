# VGGT 3D Reconstruction — Evaluation & Benchmarking

A Jupyter notebook for evaluating **VGGT** (Visual Geometry Grounded Transformer), a pretrained 3D vision model, on the **CO3D Cup** dataset. The notebook measures depth prediction accuracy, geometric consistency, and attention mechanism performance.

---

## Overview

This project evaluates VGGT's ability to reconstruct 3D geometry from single images. Two key metrics are tracked:

- **Supervised Loss (Depth MSE)** — how accurately the model predicts depth compared to ground truth
- **Consistency Loss (Geometric Error)** — how internally consistent the model's 3D logic is (predicted world points vs. mathematically derived points from depth)

The notebook also includes a 3D point cloud visualizer and an attention benchmark comparing vanilla vs. memory-efficient attention.

---

## Requirements

### Hardware
- **GPU:** NVIDIA GPU with CUDA support (tested on RTX 3050 Laptop GPU, Compute Capability 8.6)

### Python
- Python 3.10+

### Dependencies

```bash
pip install torch torchvision
pip install einops
pip install Pillow numpy pandas tqdm
pip install plotly
pip install glob2
```

> **Note:** `flash-attn` installation was attempted but failed due to OS path length issues on Windows. The model falls back gracefully to `torch.scaled_dot_product_attention`.

### VGGT Model

Clone the VGGT repository and place it in your project directory:

```
your_project/
├── main.ipynb
├── vggt/               ← VGGT repo cloned here
│   └── vggt/
│       ├── models/
│       │   └── vggt.py
│       ├── layers/
│       │   └── attention.py
│       └── utils/
│           └── geometry.py
└── model_folder/
    └── model.pt        ← Pretrained weights
```

---

## Dataset Structure

Uses the **CO3D (Common Objects in 3D)** dataset, specifically the `cup` category.

```
co3d_data/
└── cup/
    └── <scene_id>/         ← e.g. 620_101304_203141
        ├── images/
        │   └── frame000001.jpg
        ├── depths/
        │   └── frame000001.jpg.geometric.png   ← uint16, divide by 10000 for metres
        ├── masks/
        │   └── frame000001.png
        └── depth_masks/
            └── frame000001.png
```

> **Depth Scale:** Raw depth PNG values are `uint16`. Divide by `10,000` to get metres. Valid range for tabletop cups: `0.1m – 5.0m`.

---

## Notebook Walkthrough

### Cell 1 — Install `einops`
Installs the `einops` library required by VGGT.

### Cell 2 — Load Model
Loads the VGGT model class and pretrained weights (`model_folder/model.pt`). Sets device to CUDA if available.

```python
model = VGGT().to(device)
model.load_state_dict(torch.load('model_folder/model.pt', map_location=device))
model.eval()
```

### Cell 3 — Load Geometry Utilities
Imports `get_derived_point_map_torch` from VGGT's geometry module.

### Cell 4 — Quick Single-Image Evaluation
Runs a single forward pass on one depth image and computes both losses to demonstrate the two-metric evaluation framework.

**Sample Output:**
```
SUPERVISED LOSS (Accuracy):   0.196512
CONSISTENCY LOSS (Logic):    0.063882
```

### Cell 5 — Dataset Definition & Depth Verification (`CO3DCupDataset`)
Defines a PyTorch `Dataset` for the CO3D cup scenes. Handles:
- Paired image + depth loading
- Optional mask and depth-mask loading
- Depth normalization (`uint16 / 10000 → metres`)
- Sanity clipping (removes values outside 0.1–5.0m)

**Sample Output:**
```
Found 24 scene folders
✅ Total valid samples: 4132

Depth verification:
  Sample 1: min=1.669m  max=2.216m  mean=1.872m
```

### Cell 6 — Batch Evaluation (100 Samples)
Runs evaluation over 100 randomly sampled images from the dataset.

**Sample Output:**
```
Total_Samples  Depth_MSE  Geo_Inconsistency  Geometric_Accuracy_%
          100    0.27261           0.074763                 92.52
```

### Cell 7 — 10-Epoch Consolidated Evaluation
Repeats the 200-sample evaluation loop 10 times (epochs) and reports per-epoch and aggregate metrics.

**Sample Output:**
```
 Epoch  Depth_MSE  Geo_Error  Accuracy_%
     1   0.240026   0.073838       92.62
    ...
    10   0.302804   0.076222       92.38

✅ Final Consolidated Geometric Accuracy: 92.50%
```

### Cell 8 — 3D Point Cloud Visualizer
Generates an interactive side-by-side HTML visualization:
- **Left panel:** Input RGB image
- **Right panel:** 3D point cloud colored with pixel colors

Saved to `cup_3d.html`. Open in any browser.

```python
fig = visualize_image_and_3d(dataset=dataset, device=device, image_idx=0, save_path="cup_3d.html")
```

Point cloud processing pipeline:
1. Confidence filter — keeps top 60% of points
2. Depth outlier removal (1st–99th percentile)
3. Subsampling to ~30,000 points for browser performance

### Cell 9 — Flash Attention Install (Optional)
Attempts `flash-attn` installation. Expected to fail on Windows due to path length limitations — the model falls back to PyTorch's built-in `scaled_dot_product_attention`.

### Cell 10 — GPU Info
Prints GPU name and CUDA compute capability.

### Cell 11 — Attention Benchmark
Benchmarks **Vanilla Attention** vs. **Memory-Efficient Attention** (`MemEffAttention`) from VGGT across sequence lengths `[1024, 2048, 4096, 8192]`.

Metrics reported:
- Latency (ms/step)
- Peak memory (MB)
- TFLOPS
- MSE loss mean ± std
- Accuracy (relative error < 10%)

Results are saved to `benchmark_results.csv` and plotted with matplotlib.

> **Note:** This cell was interrupted during execution (KeyboardInterrupt at seq_len=1024), likely due to long runtime on a laptop GPU.

---

## Key Metrics Explained

| Metric | Description |
|---|---|
| `Depth_MSE` | Mean Squared Error between predicted depth and ground truth depth map |
| `Geo_Inconsistency` | MSE between model's predicted world points and mathematically back-projected points from depth |
| `Geometric_Accuracy_%` | `max(0, 100 - Geo_Inconsistency × 100)` |

The **Consistency Loss** is the more revealing metric — it exposes whether the model's internal 3D representation is geometrically sound, independent of ground truth data quality.

---

## Results Summary

| Evaluation | Samples | Depth MSE | Geo Inconsistency | Accuracy |
|---|---|---|---|---|
| Single run | 100 | 0.2726 | 0.0748 | 92.52% |
| 10-epoch avg | 2000 | ~0.279 | ~0.0749 | **92.50%** |

---

## File Outputs

| File | Description |
|---|---|
| `cup_3d.html` | Interactive 3D point cloud visualization (open in browser) |
| `benchmark_results.csv` | Attention benchmark results across sequence lengths |

---

## Notes & Known Issues

- **FutureWarning:** `torch.cuda.amp.autocast` is deprecated in newer PyTorch. Replace with `torch.amp.autocast('cuda', ...)`.
- **Flash-Attn on Windows:** Fails due to Windows MAX_PATH limitation. No action needed — fallback is automatic.
- **`weights_only=False`:** The `torch.load` call uses the legacy default. Consider adding `weights_only=True` once model compatibility is confirmed.
- **tqdm + Jupyter:** Install `ipywidgets` and update Jupyter to get proper notebook progress bars.
