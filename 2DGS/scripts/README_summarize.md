# Summarize Results Script

This script collects metrics from multiple sources and generates comprehensive Markdown tables:
- Test metrics (PSNR, SSIM, LPIPS) from `results.json` files
- Training metrics (Train/Test L1 loss and PSNR) from TensorBoard logs

## Installation

To use TensorBoard functionality, make sure tensorboard is installed:

```bash
pip install tensorboard
```

## Usage

### Basic Usage

```bash
# Print results to console
cd 2DGS
python scripts/summarize_results.py --output_dir output/mipnerf360

# Save to markdown file
python scripts/summarize_results.py --output_dir output/mipnerf360 --output_file mipnerf360_results.md
```

### Advanced Options

```bash
# Custom dataset name
python scripts/summarize_results.py --output_dir output/nerf_synthetic --dataset "NeRF Synthetic"

# Sort scenes by PSNR (default is by name)
python scripts/summarize_results.py --output_dir output/mipnerf360 --sort_by psnr

# Specify custom checkpoint iterations to report
python scripts/summarize_results.py --output_dir output/mipnerf360 --iterations 5000,10000,30000

# Skip TensorBoard data (only show test metrics)
python scripts/summarize_results.py --output_dir output/mipnerf360 --skip_tensorboard
```

## Output Format

### With TensorBoard Data

```markdown
## Checkpoint: ours_30000 (30000 iterations)

| Scene | Train L1↓ | Train PSNR↑ | Test L1↓ | Test PSNR↑ | Eval PSNR↑ | SSIM↑ | LPIPS↓ |
|-------|-----------|-------------|----------|------------|------------|-------|---------|
| bicycle | 0.0123 | 26.34 | 0.0145 | 25.89 | 24.7392 | 0.7328 | 0.2690 |
| garden | 0.0098 | 28.12 | 0.0112 | 27.45 | 26.6736 | 0.8423 | 0.1466 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| **Average** | **0.0115** | **27.23** | **0.0128** | **26.67** | **26.8079** | **0.7967** | **0.2522** |

## Summary: Average Metrics Across All Scenes

| Checkpoint | Train L1↓ | Train PSNR↑ | Test L1↓ | Test PSNR↑ | Eval PSNR↑ | SSIM↑ | LPIPS↓ |
|------------|-----------|-------------|----------|------------|------------|-------|---------|
| 5K | 0.0245 | 24.12 | 0.0289 | 23.45 | 25.1083 | 0.7291 | 0.3278 |
| 7K | 0.0198 | 25.34 | 0.0234 | 24.78 | 25.5969 | 0.7531 | 0.3008 |
| 10K | 0.0167 | 26.12 | 0.0201 | 25.56 | 25.7511 | 0.7670 | 0.2907 |
| 15K | 0.0143 | 26.78 | 0.0178 | 26.23 | 26.2412 | 0.7853 | 0.2692 |
| 30K | 0.0115 | 27.23 | 0.0128 | 26.67 | 26.8079 | 0.7967 | 0.2522 |
```

### Without TensorBoard Data

If TensorBoard is not available or `--skip_tensorboard` is used, only test metrics are shown:

```markdown
## Checkpoint: ours_30000 (30000 iterations)

| Scene | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|--------|--------|---------|
| bicycle | 24.7392 | 0.7328 | 0.2690 |
| garden | 26.6736 | 0.8423 | 0.1466 |
| ... | ... | ... | ... |
| **Average** | **26.8079** | **0.7967** | **0.2522** |
```

## Notes

- **Eval PSNR**: Computed from full rendering on test set (from `render.py` + `metrics.py`)
- **Test PSNR**: Computed during training on test views (from TensorBoard logs)
- **Train L1/PSNR**: Training metrics logged during optimization
- The script automatically finds all scenes in the output directory
- TensorBoard event files should be in the scene root directory (e.g., `output/mipnerf360/garden/events.out.tfevents.*`)

