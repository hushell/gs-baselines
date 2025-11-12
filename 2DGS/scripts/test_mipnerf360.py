#!/usr/bin/env python3
# Testing script for the MipNeRF360 dataset
# Renders and evaluates multiple checkpoints (5K, 7K, 10K, 15K, 30K) in parallel across available GPUs

import os
import argparse
import concurrent.futures
import time

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
    """Render and evaluate a single scene at multiple checkpoint iterations"""
    scene_output = f"{args.output_dir}/{scene}"
    scene_source = f"{args.data_dir}/{scene}"

    # Rendering at multiple iterations
    if not args.skip_render:
        for iteration in iterations:
            cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
                   f"python render.py --iteration {iteration} -s {scene_source} "
                   f"-m {scene_output} --eval --skip_train >> {log_file} 2>&1")
            print(f"[GPU {gpu}] Rendering {scene} at iteration {iteration}... (log: {log_file})")
            if not args.dry_run:
                os.system(cmd)

    # Metrics (computed once for all iterations)
    if not args.skip_metrics:
        cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
               f"python metrics.py -m {scene_output} >> {log_file} 2>&1")
        print(f"[GPU {gpu}] Computing metrics for {scene} (all iterations)... (log: {log_file})")
        if not args.dry_run:
            os.system(cmd)

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
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

