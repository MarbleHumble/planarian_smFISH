# Threshold Validation Guide

## Overview
This guide helps you decide where to run the threshold validation pipeline (18 images × 16 thresholds = 288 spot detections).

## Quick Performance Test

**First, run this to compare performance:**

```bash
python compare_gpu_cpu_performance.py --image_path /path/to/one/test/image.tif --n_tests 5
```

This will test:
- **MPS (Mac GPU)**: If you have an Apple Silicon Mac
- **CUDA (Server GPU)**: If available on your server
- **CPU (Big-FISH)**: Fallback option

The script will estimate total time for the full validation.

## Options

### Option 1: MPS (Mac GPU) - **RECOMMENDED IF < 2 hours**
**Pros:**
- No need to upload to server
- GPU acceleration (typically 5-10x faster than CPU)
- Easy to monitor progress locally

**Cons:**
- MPS may be slower than CUDA
- Uses your local machine resources

**Run:**
```bash
python test_thresholds_visualization_gpu.py \
    --data_dir /Users/eliasguan/Desktop/DL_210_data_analysis/ML_data_set_trainning \
    --output_dir ./threshold_validation_results \
    --use_gpu \
    --device mps \
    --n_thresholds 16
```

**Estimated time:** Run the performance test first to get accurate estimate (typically 30-120 minutes for 18 images).

---

### Option 2: Server (CUDA GPU) - **RECOMMENDED IF MPS IS SLOW**
**Pros:**
- Fastest option (CUDA is typically faster than MPS)
- Doesn't use local machine resources
- Can run in background on server

**Cons:**
- Need to upload code and data
- Need to monitor via SSH/logs

**Steps:**
1. Upload code to server:
   ```bash
   rsync -avz --exclude='.git' --exclude='__pycache__' \
       /Users/eliasguan/Desktop/planarian_smFISH/ \
       user@server:/path/to/planarian_smFISH/
   ```

2. Upload data to server:
   ```bash
   rsync -avz \
       /Users/eliasguan/Desktop/DL_210_data_analysis/ML_data_set_trainning/ \
       user@server:/path/to/ML_data_set_trainning/
   ```

3. Run on server (via SLURM or directly):
   ```bash
   # If using SLURM
   sbatch --gres=gpu:1 --time=2:00:00 run_threshold_validation.sh
   
   # Or directly
   python test_thresholds_visualization_gpu.py \
       --data_dir /path/to/ML_data_set_trainning \
       --output_dir ./threshold_validation_results \
       --use_gpu \
       --device cuda \
       --n_thresholds 16
   ```

**Estimated time:** Typically 15-60 minutes for 18 images (faster than MPS).

---

### Option 3: CPU (Big-FISH) - **FALLBACK ONLY**
**Pros:**
- Works anywhere
- No GPU needed

**Cons:**
- **Very slow** (typically 3-10 hours for 18 images)
- Not recommended unless GPU unavailable

**Run:**
```bash
python test_thresholds_visualization_gpu.py \
    --data_dir /Users/eliasguan/Desktop/DL_210_data_analysis/ML_data_set_trainning \
    --output_dir ./threshold_validation_results \
    --use_gpu False \
    --n_workers 4 \
    --n_thresholds 16
```

---

## Recommendation Workflow

1. **Run performance test first:**
   ```bash
   python compare_gpu_cpu_performance.py \
       --image_path /Users/eliasguan/Desktop/DL_210_data_analysis/ML_data_set_trainning/Original\ Images/one_image.tif \
       --n_tests 5
   ```

2. **Check estimated time:**
   - If MPS < 2 hours → Use MPS locally
   - If MPS > 2 hours → Use server CUDA
   - If no GPU available → Use CPU (but expect long wait)

3. **Run full validation** using the appropriate option above.

---

## Output

The validation will create:
- `threshold_validation_results/`: Directory with:
  - `*_threshold_validation.png`: Visualization plots for each image
  - `threshold_summary.json`: Best threshold for each image with metrics

---

## Notes

- **GPU acceleration**: The GPU version (`test_thresholds_visualization_gpu.py`) uses the same detection algorithm as the CPU version, just accelerated on GPU.
- **Parallelization**: GPU handles parallelism internally. CPU version can use multiple workers.
- **Memory**: GPU version requires GPU memory. For large images, you may need to reduce batch size or use CPU.

