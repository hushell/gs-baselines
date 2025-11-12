#!/usr/bin/env python3
# Summary script to collect results.json files and TensorBoard logs, then generate Markdown tables

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    print("Will only show test metrics from results.json\n")

parser = argparse.ArgumentParser(description="Summarize results from multiple scenes into Markdown table")
parser.add_argument("--output_dir", default="output/mipnerf360", type=str, help="Output directory containing scene results")
parser.add_argument("--output_file", default=None, type=str, help="Output markdown file (default: print to console)")
parser.add_argument("--dataset", default="MipNeRF360", type=str, help="Dataset name for table header")
parser.add_argument("--sort_by", default="name", choices=["name", "psnr", "ssim"], help="Sort scenes by name or metric")
parser.add_argument("--iterations", type=str, default="5000,7000,10000,15000,30000", 
                    help="Comma-separated list of checkpoint iterations to report")
parser.add_argument("--skip_tensorboard", action="store_true", help="Skip reading TensorBoard logs")
args = parser.parse_args()

# Parse iterations
CHECKPOINT_ITERATIONS = [int(it) for it in args.iterations.split(",") if it.strip()]


def read_tensorboard_scalars(log_dir, tags, iterations):
    """Read specific scalar values from TensorBoard event files at given iterations"""
    if not TENSORBOARD_AVAILABLE or args.skip_tensorboard:
        return {}
    
    try:
        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()
        
        results = {}
        for tag in tags:
            if tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                # Create a dict mapping iteration to value
                scalar_dict = {int(e.step): e.value for e in events}
                
                # Get values at specific iterations
                for iteration in iterations:
                    if iteration in scalar_dict:
                        if tag not in results:
                            results[tag] = {}
                        results[tag][iteration] = scalar_dict[iteration]
        
        return results
    except Exception as e:
        print(f"Warning: Could not read TensorBoard logs from {log_dir}: {e}")
        return {}


def collect_tensorboard_metrics(output_dir, iterations):
    """Collect training/test PSNR and L1 loss from TensorBoard logs"""
    tensorboard_data = {}
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory '{output_dir}' does not exist")
        return tensorboard_data
    
    # Tags to extract from TensorBoard
    tags = [
        'train/loss_viewpoint - l1_loss',
        'train/loss_viewpoint - psnr',
        'test/loss_viewpoint - l1_loss',
        'test/loss_viewpoint - psnr'
    ]
    
    for scene_dir in output_path.iterdir():
        if not scene_dir.is_dir():
            continue
        
        # Look for TensorBoard event files in the scene directory
        tb_files = list(scene_dir.glob("events.out.tfevents.*"))
        
        if tb_files:
            print(f"Reading TensorBoard logs for: {scene_dir.name}")
            metrics = read_tensorboard_scalars(scene_dir, tags, iterations)
            if metrics:
                tensorboard_data[scene_dir.name] = metrics
    
    return tensorboard_data


def collect_results(output_dir):
    """Collect all results.json files from scene directories"""
    results = {}
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory '{output_dir}' does not exist")
        return results
    
    # Find all results.json files
    for scene_dir in output_path.iterdir():
        if not scene_dir.is_dir():
            continue
        
        results_file = scene_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    results[scene_dir.name] = data
                    print(f"Loaded results for: {scene_dir.name}")
            except Exception as e:
                print(f"Warning: Could not load {results_file}: {e}")
    
    return results


def format_markdown_table(results, tensorboard_data, dataset_name="MipNeRF360"):
    """Format results as a Markdown table with both test metrics and training stats"""
    if not results:
        return "No results found."
    
    # Get all unique checkpoint iterations across all scenes
    all_iterations = set()
    for scene_data in results.values():
        all_iterations.update(scene_data.keys())
    
    # Sort iterations (assuming format "ours_XXXXX")
    sorted_iterations = sorted(all_iterations, key=lambda x: int(x.split('_')[-1]) if '_' in x else 0)
    
    # Also check which iterations we have in the args
    checkpoint_iters = CHECKPOINT_ITERATIONS
    
    lines = []
    lines.append(f"# {dataset_name} Results\n")
    
    has_tb_data = bool(tensorboard_data)
    
    # Create a table for each checkpoint iteration
    for iteration in sorted_iterations:
        iteration_num = int(iteration.split('_')[-1]) if '_' in iteration else int(iteration)
        lines.append(f"## Checkpoint: {iteration} ({iteration_num} iterations)\n")
        
        # Table header - expanded if we have TensorBoard data
        if has_tb_data:
            lines.append("| Scene | Train L1↓ | Train PSNR↑ | Test L1↓ | Test PSNR↑ | Eval PSNR↑ | SSIM↑ | LPIPS↓ |")
            lines.append("|-------|-----------|-------------|----------|------------|------------|-------|---------|")
        else:
            lines.append("| Scene | PSNR ↑ | SSIM ↑ | LPIPS ↓ |")
            lines.append("|-------|--------|--------|---------|")
        
        # Collect data for this iteration
        scene_metrics = []
        for scene, scene_data in results.items():
            if iteration in scene_data:
                metrics = scene_data[iteration]
                psnr = metrics.get('PSNR', 0)
                ssim = metrics.get('SSIM', 0)
                lpips = metrics.get('LPIPS', 0)
                
                # Get TensorBoard data if available
                train_l1 = train_psnr = test_l1 = test_psnr = None
                if has_tb_data and scene in tensorboard_data:
                    tb_metrics = tensorboard_data[scene]
                    train_l1_tag = 'train/loss_viewpoint - l1_loss'
                    train_psnr_tag = 'train/loss_viewpoint - psnr'
                    test_l1_tag = 'test/loss_viewpoint - l1_loss'
                    test_psnr_tag = 'test/loss_viewpoint - psnr'
                    
                    if train_l1_tag in tb_metrics and iteration_num in tb_metrics[train_l1_tag]:
                        train_l1 = tb_metrics[train_l1_tag][iteration_num]
                    if train_psnr_tag in tb_metrics and iteration_num in tb_metrics[train_psnr_tag]:
                        train_psnr = tb_metrics[train_psnr_tag][iteration_num]
                    if test_l1_tag in tb_metrics and iteration_num in tb_metrics[test_l1_tag]:
                        test_l1 = tb_metrics[test_l1_tag][iteration_num]
                    if test_psnr_tag in tb_metrics and iteration_num in tb_metrics[test_psnr_tag]:
                        test_psnr = tb_metrics[test_psnr_tag][iteration_num]
                
                scene_metrics.append((scene, train_l1, train_psnr, test_l1, test_psnr, psnr, ssim, lpips))
        
        # Sort scenes
        if args.sort_by == "psnr":
            scene_metrics.sort(key=lambda x: x[5], reverse=True)  # eval PSNR is index 5
        elif args.sort_by == "ssim":
            scene_metrics.sort(key=lambda x: x[6], reverse=True)  # SSIM is index 6
        else:  # name
            scene_metrics.sort(key=lambda x: x[0])
        
        # Add rows
        if has_tb_data:
            train_l1_sum = train_psnr_sum = test_l1_sum = test_psnr_sum = 0
            train_l1_count = train_psnr_count = test_l1_count = test_psnr_count = 0
        psnr_sum = ssim_sum = lpips_sum = 0
        count = 0
        
        for scene, train_l1, train_psnr, test_l1, test_psnr, psnr, ssim, lpips in scene_metrics:
            if has_tb_data:
                train_l1_str = f"{train_l1:.4f}" if train_l1 is not None else "N/A"
                train_psnr_str = f"{train_psnr:.2f}" if train_psnr is not None else "N/A"
                test_l1_str = f"{test_l1:.4f}" if test_l1 is not None else "N/A"
                test_psnr_str = f"{test_psnr:.2f}" if test_psnr is not None else "N/A"
                
                lines.append(f"| {scene} | {train_l1_str} | {train_psnr_str} | {test_l1_str} | {test_psnr_str} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} |")
                
                if train_l1 is not None:
                    train_l1_sum += train_l1
                    train_l1_count += 1
                if train_psnr is not None:
                    train_psnr_sum += train_psnr
                    train_psnr_count += 1
                if test_l1 is not None:
                    test_l1_sum += test_l1
                    test_l1_count += 1
                if test_psnr is not None:
                    test_psnr_sum += test_psnr
                    test_psnr_count += 1
            else:
                lines.append(f"| {scene} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} |")
            
            psnr_sum += psnr
            ssim_sum += ssim
            lpips_sum += lpips
            count += 1
        
        # Add average row
        if count > 0:
            avg_psnr = psnr_sum / count
            avg_ssim = ssim_sum / count
            avg_lpips = lpips_sum / count
            
            if has_tb_data:
                avg_train_l1 = f"{train_l1_sum / train_l1_count:.4f}" if train_l1_count > 0 else "N/A"
                avg_train_psnr = f"{train_psnr_sum / train_psnr_count:.2f}" if train_psnr_count > 0 else "N/A"
                avg_test_l1 = f"{test_l1_sum / test_l1_count:.4f}" if test_l1_count > 0 else "N/A"
                avg_test_psnr = f"{test_psnr_sum / test_psnr_count:.2f}" if test_psnr_count > 0 else "N/A"
                
                lines.append(f"| **Average** | **{avg_train_l1}** | **{avg_train_psnr}** | **{avg_test_l1}** | **{avg_test_psnr}** | **{avg_psnr:.4f}** | **{avg_ssim:.4f}** | **{avg_lpips:.4f}** |")
            else:
                lines.append(f"| **Average** | **{avg_psnr:.4f}** | **{avg_ssim:.4f}** | **{avg_lpips:.4f}** |")
        
        lines.append("")  # Empty line between tables
    
    # Create a comparison table across all iterations
    if len(sorted_iterations) > 1:
        lines.append("## Summary: Average Metrics Across All Scenes\n")
        
        if has_tb_data:
            lines.append("| Checkpoint | Train L1↓ | Train PSNR↑ | Test L1↓ | Test PSNR↑ | Eval PSNR↑ | SSIM↑ | LPIPS↓ |")
            lines.append("|------------|-----------|-------------|----------|------------|------------|-------|---------|")
        else:
            lines.append("| Checkpoint | PSNR ↑ | SSIM ↑ | LPIPS ↓ |")
            lines.append("|------------|--------|--------|---------|")
        
        for iteration in sorted_iterations:
            iteration_num = int(iteration.split('_')[-1]) if '_' in iteration else int(iteration)
            
            if has_tb_data:
                train_l1_sum = train_psnr_sum = test_l1_sum = test_psnr_sum = 0
                train_l1_count = train_psnr_count = test_l1_count = test_psnr_count = 0
            psnr_sum = ssim_sum = lpips_sum = 0
            count = 0
            
            for scene, scene_data in results.items():
                if iteration in scene_data:
                    metrics = scene_data[iteration]
                    psnr_sum += metrics.get('PSNR', 0)
                    ssim_sum += metrics.get('SSIM', 0)
                    lpips_sum += metrics.get('LPIPS', 0)
                    count += 1
                    
                    # Aggregate TensorBoard data
                    if has_tb_data and scene in tensorboard_data:
                        tb_metrics = tensorboard_data[scene]
                        train_l1_tag = 'train/loss_viewpoint - l1_loss'
                        train_psnr_tag = 'train/loss_viewpoint - psnr'
                        test_l1_tag = 'test/loss_viewpoint - l1_loss'
                        test_psnr_tag = 'test/loss_viewpoint - psnr'
                        
                        if train_l1_tag in tb_metrics and iteration_num in tb_metrics[train_l1_tag]:
                            train_l1_sum += tb_metrics[train_l1_tag][iteration_num]
                            train_l1_count += 1
                        if train_psnr_tag in tb_metrics and iteration_num in tb_metrics[train_psnr_tag]:
                            train_psnr_sum += tb_metrics[train_psnr_tag][iteration_num]
                            train_psnr_count += 1
                        if test_l1_tag in tb_metrics and iteration_num in tb_metrics[test_l1_tag]:
                            test_l1_sum += tb_metrics[test_l1_tag][iteration_num]
                            test_l1_count += 1
                        if test_psnr_tag in tb_metrics and iteration_num in tb_metrics[test_psnr_tag]:
                            test_psnr_sum += tb_metrics[test_psnr_tag][iteration_num]
                            test_psnr_count += 1
            
            if count > 0:
                avg_psnr = psnr_sum / count
                avg_ssim = ssim_sum / count
                avg_lpips = lpips_sum / count
                iteration_display = f"{iteration_num // 1000}K"
                
                if has_tb_data:
                    avg_train_l1 = f"{train_l1_sum / train_l1_count:.4f}" if train_l1_count > 0 else "N/A"
                    avg_train_psnr = f"{train_psnr_sum / train_psnr_count:.2f}" if train_psnr_count > 0 else "N/A"
                    avg_test_l1 = f"{test_l1_sum / test_l1_count:.4f}" if test_l1_count > 0 else "N/A"
                    avg_test_psnr = f"{test_psnr_sum / test_psnr_count:.2f}" if test_psnr_count > 0 else "N/A"
                    
                    lines.append(f"| {iteration_display} | {avg_train_l1} | {avg_train_psnr} | {avg_test_l1} | {avg_test_psnr} | {avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |")
                else:
                    lines.append(f"| {iteration_display} | {avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |")
        
        lines.append("")
    
    return "\n".join(lines)


def main():
    print(f"Collecting results from: {args.output_dir}\n")
    results = collect_results(args.output_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"\nFound results for {len(results)} scenes\n")
    
    # Collect TensorBoard data
    tensorboard_data = {}
    if TENSORBOARD_AVAILABLE and not args.skip_tensorboard:
        print("Collecting TensorBoard metrics...\n")
        tensorboard_data = collect_tensorboard_metrics(args.output_dir, CHECKPOINT_ITERATIONS)
        if tensorboard_data:
            print(f"Found TensorBoard data for {len(tensorboard_data)} scenes\n")
        else:
            print("No TensorBoard data found\n")
    
    # Generate markdown table
    markdown = format_markdown_table(results, tensorboard_data, args.dataset)
    
    # Output
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(markdown)
        print(f"Results saved to: {args.output_file}")
    else:
        print("\n" + "="*80)
        print(markdown)
        print("="*80)


if __name__ == "__main__":
    main()

