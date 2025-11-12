# Training script for the NeRF Synthetic dataset
# Runs all scenes in parallel across available GPUs

import os
import argparse
import concurrent.futures
import time

# NeRF Synthetic scenes
scenes = ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]

parser = argparse.ArgumentParser(description="Parallel training script for NeRF Synthetic")
parser.add_argument("--data_dir", default="data/nerf_synthetic", type=str, help="Path to NeRF Synthetic dataset")
parser.add_argument("--output_dir", default="output/nerf_synthetic", type=str, help="Output directory")
parser.add_argument("--num_workers", default=8, type=int, help="Number of parallel workers")
parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
parser.add_argument("--skip_train", action="store_true", help="Skip training")
parser.add_argument("--skip_render", action="store_true", help="Skip rendering")
parser.add_argument("--skip_metrics", action="store_true", help="Skip metrics computation")
parser.add_argument("--excluded_gpus", type=str, default="", help="Comma-separated list of GPU IDs to exclude (e.g., '0,1')")
parser.add_argument("--lambda_normal", default=0.0, type=float, help="Lambda for normal loss")
args = parser.parse_args()

excluded_gpus = set([int(gpu) for gpu in args.excluded_gpus.split(",") if gpu.strip()])

# Create jobs list
#jobs = [(scene,) for scene in scenes]
jobs = scenes


def train_scene(gpu, scene, log_file):
    """Train, render, and evaluate a single scene"""
    scene_output = f"{args.output_dir}/{scene}"
    scene_source = f"{args.data_dir}/{scene}"

    # Training
    if not args.skip_train:
        cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
               f"python train.py -s {scene_source} -m {scene_output} --eval "
               f"--white_background --lambda_normal {args.lambda_normal} "
               f"--port {6209+int(gpu)} >> {log_file} 2>&1")
        print(f"[GPU {gpu}] Training {scene}... (log: {log_file})")
        if not args.dry_run:
            os.system(cmd)

    # Rendering
    if not args.skip_render:
        cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
               f"python render.py -m {scene_output} --skip_train --skip_mesh "
               f">> {log_file} 2>&1")
        print(f"[GPU {gpu}] Rendering {scene}... (log: {log_file})")
        if not args.dry_run:
            os.system(cmd)

    # Metrics
    if not args.skip_metrics:
        cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
               f"python metrics.py -m {scene_output} >> {log_file} 2>&1")
        print(f"[GPU {gpu}] Computing metrics for {scene}... (log: {log_file})")
        if not args.dry_run:
            os.system(cmd)

    return True


def worker(gpu, scene, log_file):
    """Worker function that processes a single job"""
    print(f"\n{'=' * 60}")
    print(f"Starting job on GPU {gpu}: {scene}")
    print(f"Log file: {log_file}")
    print(f"{'=' * 60}\n")

    train_scene(gpu, scene, log_file)

    print(f"\n{'=' * 60}")
    print(f"Finished job on GPU {gpu}: {scene}")
    print(f"{'=' * 60}\n")


def run_jobs_dynamically():
    """Run jobs with dynamic GPU scheduling (reuse GPUs as they finish)"""
    log_dir = f"{args.output_dir}/logs"
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nTotal jobs to process: {len(jobs)}")
    print(f"Scenes: {[job[0] for job in jobs]}")
    print(f"Logs directory: {log_dir}\n")

    available_gpus = [i for i in range(8) if i not in excluded_gpus]
    num_gpus = min(len(available_gpus), args.num_workers)

    print(f"Available GPUs: {available_gpus}")
    print(f"Running up to {num_gpus} jobs in parallel.\n")

    running = {}
    job_iter = iter(jobs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        # Start one job per GPU
        for gpu in available_gpus[:num_gpus]:
            try:
                scene = next(job_iter)
            except StopIteration:
                break
            log_file = f"{log_dir}/{scene}.log"
            print(f"Launching {scene} on GPU {gpu} (log: {log_file})")
            future = executor.submit(worker, gpu, scene, log_file)
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
                    scene = next(job_iter)
                except StopIteration:
                    continue
                log_file = f"{log_dir}/{scene}.log"
                print(f"Launching {scene} on GPU {gpu} (log: {log_file})")
                new_future = executor.submit(worker, gpu, scene, log_file)
                running[new_future] = gpu

    print("\nAll jobs have been processed.")
    print(f"Logs saved to: {log_dir}")


if __name__ == "__main__":
    run_jobs_dynamically()
