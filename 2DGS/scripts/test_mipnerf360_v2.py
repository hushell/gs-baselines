#!/usr/bin/env python3
# Testing script for the MipNeRF360 dataset
# Renders and evaluates multiple checkpoints (5K, 7K, 10K, 15K, 30K) in parallel across available GPUs

import os
import argparse
import concurrent.futures
import time
import json
import torch
from argparse import Namespace, ArgumentParser
from gaussian_renderer import render
from scene import Scene
from utils.image_utils import psnr
from utils.loss_utils import ssim, l1_loss
from arguments import PipelineParams
from gaussian_renderer import GaussianModel
from lpipsPyTorch import lpips

# MipNeRF360 scenes
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]

parser = argparse.ArgumentParser(description="Parallel testing script for MipNeRF360 (multiple checkpoints)")
parser.add_argument("--data_dir", default="data/mipnerf360", type=str, help="Path to MipNeRF360 dataset")
parser.add_argument("--output_dir", default="output/mipnerf360", type=str, help="Output directory")
parser.add_argument("--num_workers", default=8, type=int, help="Number of parallel workers (GPUs)")
parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
parser.add_argument("--skip_render", action="store_true", help="Skip rendering")
parser.add_argument("--skip_metrics", action="store_true", help="Skip metrics computation")
parser.add_argument("--iterations", type=str, default="5000,7000,10000,15000,30000", 
                    help="Comma-separated list of checkpoint iterations to evaluate")
parser.add_argument("--excluded_gpus", type=str, default="", help="Comma-separated list of GPU IDs to exclude (e.g., '0,1')")
args = parser.parse_args()

excluded_gpus = set(int(gpu) for gpu in args.excluded_gpus.split(",") if gpu.strip())
iterations = [int(it) for it in args.iterations.split(",") if it.strip()]

# Create jobs list: (scene, image_folder)
jobs = []
for scene in mipnerf360_outdoor_scenes:
    jobs.append((scene, "images_4"))
for scene in mipnerf360_indoor_scenes:
    jobs.append((scene, "images_2"))


def test_scene(gpu, scene, image_folder, log_file):
    """Evaluate a single scene at multiple checkpoint iterations in-process (no subprocess calls)."""
    scene_output = f"{args.output_dir}/{scene}"
    scene_source = f"{args.data_dir}/{scene}"

    #os.makedirs(scene_output, exist_ok=True)
    #os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(msg):
        print(msg)
        try:
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    if args.dry_run:
        log(f"[GPU {gpu}] (dry-run) Would evaluate {scene} at iterations {iterations} using {image_folder}")
        return True

    # Bind computations to the target GPU for this worker process
    torch.cuda.set_device(gpu)

    # Load saved training config and build a dataset-like args object
    cfg_path = os.path.join(scene_output, "cfg_args")
    if not os.path.exists(cfg_path):
        log(f"[GPU {gpu}] Error: cfg_args not found at {cfg_path}. Did you train the scene first?")
        return False
    try:
        with open(cfg_path, "r") as f:
            cfg_args = eval(f.read())
    except Exception as e:
        log(f"[GPU {gpu}] Error reading cfg_args for {scene}: {e}")
        return False

    # Override fields for evaluation
    cfg_args.model_path = scene_output
    cfg_args.source_path = os.path.abspath(scene_source)
    cfg_args.images = image_folder
    cfg_args.eval = True

    # Build pipeline group from cfg (depth_ratio etc.)
    pipeline_group = PipelineParams(ArgumentParser())
    pipe = pipeline_group.extract(cfg_args)

    # Background color
    bg_color = [1, 1, 1] if getattr(cfg_args, "white_background", False) else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # LPIPS metric
    def compute_lpips(x, y):
        # x,y expected in [0,1], shape [B,C,H,W]
        return lpips(x, y, net_type='vgg')

    # results.json accumulation
    results_json_path = os.path.join(scene_output, "results.json")
    try:
        if os.path.exists(results_json_path):
            with open(results_json_path, "r") as f:
                results_data = json.load(f)
        else:
            results_data = {}
    except Exception:
        results_data = {}

    for iteration in iterations:
        log(f"[GPU {gpu}] Evaluating {scene} at iteration {iteration} ...")
        try:
            gaussians = GaussianModel(getattr(cfg_args, "sh_degree", 3))
            scene_obj = Scene(cfg_args, gaussians, load_iteration=iteration, shuffle=False)
            torch.cuda.empty_cache()

            # Validation configs: test set (all), small subset of train set
            validation_configs = (
                {'name': 'test', 'cameras': scene_obj.getTestCameras()},
                {'name': 'train', 'cameras': [scene_obj.getTrainCameras()[idx % len(scene_obj.getTrainCameras())] for idx in range(5, 30, 5)]}
            )

            # Accumulate metrics per split
            split_metrics = {}
            for config in validation_configs:
                if not config['cameras'] or len(config['cameras']) == 0:
                    continue

                l1_sum = 0.0
                psnr_sum = 0.0
                ssim_sum = 0.0
                lpips_sum = 0.0

                for _, viewpoint in enumerate(config['cameras']):
                    render_pkg = render(viewpoint, scene_obj.gaussians, pipe, background)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    # Ensure 4D for metrics that expect batches
                    if image.dim() == 3:
                        image_b = image.unsqueeze(0)
                        gt_image_b = gt_image.unsqueeze(0)
                    else:
                        image_b = image
                        gt_image_b = gt_image

                    l1_sum += l1_loss(image, gt_image).mean().double().item()
                    psnr_sum += psnr(image_b, gt_image_b).mean().double().item()
                    ssim_sum += ssim(image_b, gt_image_b).mean().double().item()
                    lpips_sum += compute_lpips(image_b, gt_image_b).mean().double().item()

                denom = float(len(config['cameras']))
                split_metrics[config['name']] = {
                    "L1": l1_sum / denom,
                    "PSNR": psnr_sum / denom,
                    "SSIM": ssim_sum / denom,
                    "LPIPS": lpips_sum / denom
                }
                log(f"[GPU {gpu}] [{scene}] iter {iteration} {config['name']}: "
                    f"L1 {split_metrics[config['name']]['L1']:.4f} "
                    f"PSNR {split_metrics[config['name']]['PSNR']:.4f} "
                    f"SSIM {split_metrics[config['name']]['SSIM']:.4f} "
                    f"LPIPS {split_metrics[config['name']]['LPIPS']:.4f}")

            # Persist metrics as separate keys per split for summarize_results compatibility
            wrote_any = False
            test_key = f"ours_{iteration}_test"
            train_key = f"ours_{iteration}_train"
            if "test" in split_metrics:
                results_data[test_key] = {
                    "PSNR": split_metrics["test"]["PSNR"],
                    "SSIM": split_metrics["test"]["SSIM"],
                    "LPIPS": split_metrics["test"]["LPIPS"]
                }
                wrote_any = True
            if "train" in split_metrics:
                results_data[train_key] = {
                    "PSNR": split_metrics["train"]["PSNR"],
                    "SSIM": split_metrics["train"]["SSIM"],
                    "LPIPS": split_metrics["train"]["LPIPS"]
                }
                wrote_any = True
            if wrote_any:
                with open(results_json_path, "w") as f:
                    json.dump(results_data, f, indent=2)
                log(f"[GPU {gpu}] Wrote results for {scene} iter {iteration} (test/train) to {results_json_path}")

            torch.cuda.empty_cache()
        except Exception as e:
            log(f"[GPU {gpu}] Error while evaluating {scene} at iter {iteration}: {e}")
            torch.cuda.empty_cache()
            continue

    return True


def worker(gpu, scene, image_folder, log_file):
    """Worker function that processes a single job"""
    print(f"\n{'=' * 60}")
    print(f"Starting job on GPU {gpu}: {scene} (using {image_folder})")
    print(f"Iterations to evaluate: {iterations}")
    print(f"Log file: {log_file}")
    print(f"{'=' * 60}\n")

    test_scene(gpu, scene, image_folder, log_file)

    print(f"\n{'=' * 60}")
    print(f"Finished job on GPU {gpu}: {scene}")
    print(f"{'=' * 60}\n")


def run_jobs_dynamically():
    """Run jobs with dynamic GPU scheduling (reuse GPUs as they finish)"""
    log_dir = f"{args.output_dir}/logs_test"
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nTotal jobs to process: {len(jobs)}")
    print(f"Scenes: {[job[0] for job in jobs]}")
    print(f"Checkpoint iterations: {iterations}")
    print(f"Logs directory: {log_dir}\n")

    available_gpus = [i for i in range(8) if i not in excluded_gpus]
    num_gpus = min(len(available_gpus), args.num_workers)

    print(f"Available GPUs: {available_gpus}")
    print(f"Running up to {num_gpus} jobs in parallel.\n")

    running = {}
    job_iter = iter(jobs)

    # Use process-based parallelism for safe multi-GPU execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
        # Start one job per GPU
        for gpu in available_gpus[:num_gpus]:
            try:
                scene, image_folder = next(job_iter)
            except StopIteration:
                break
            log_file = f"{log_dir}/{scene}_test.log"
            print(f"Launching {scene} on GPU {gpu} (log: {log_file})")
            future = executor.submit(worker, gpu, scene, image_folder, log_file)
            running[future] = gpu

        # Monitor completion and reuse GPUs as they free up
        while running:
            done, _ = concurrent.futures.wait(
                running, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                gpu = running.pop(future)
                try:
                    future.result()
                except Exception as e:
                    print(f"⚠️ Job on GPU {gpu} failed: {e}")
                else:
                    print(f"✅ Job on GPU {gpu} finished.")

                # Start next available job
                try:
                    scene, image_folder = next(job_iter)
                except StopIteration:
                    continue
                log_file = f"{log_dir}/{scene}_test.log"
                print(f"Launching {scene} on GPU {gpu} (log: {log_file})")
                new_future = executor.submit(worker, gpu, scene, image_folder, log_file)
                running[new_future] = gpu

    print("\nAll jobs have been processed.")
    print(f"Logs saved to: {log_dir}")
    print(f"\nResults saved to: {args.output_dir}/[scene_name]/results.json")


if __name__ == "__main__":
    run_jobs_dynamically()

