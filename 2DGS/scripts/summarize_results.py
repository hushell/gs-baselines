#!/usr/bin/env python3
# Summary script to collect results.json files and generate Markdown tables (TensorBoard removed)

import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Summarize results from multiple scenes into Markdown table")
parser.add_argument("--output_dir", default="output/mipnerf360", type=str, help="Output directory containing scene results")
parser.add_argument("--output_file", default=None, type=str, help="Output markdown file (default: print to console)")
parser.add_argument("--dataset", default="MipNeRF360", type=str, help="Dataset name for table header")
parser.add_argument("--sort_by", default="name", choices=["name", "psnr", "ssim"], help="Sort scenes by name or metric")
args = parser.parse_args()

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


def format_markdown_table(results, dataset_name="MipNeRF360"):
    """Format results as a Markdown table using only results.json metrics"""
    if not results:
        return "No results found."
    
    # Parse keys of form 'ours_<iter>_test' or 'ours_<iter>_train'
    def parse_key(key):
        # Expected: ours_<number>_<split>
        parts = key.split('_')
        if len(parts) >= 3 and parts[0] == "ours":
            try:
                base_iter = int(parts[1])
            except ValueError:
                return None
            split = parts[2]
            if split not in ("test", "train"):
                return None
            return base_iter, split
        return None

    # Collect base iterations
    base_iterations = set()
    for scene_data in results.values():
        for k in scene_data.keys():
            parsed = parse_key(k)
            if parsed:
                base_iterations.add(parsed[0])
    sorted_base_iterations = sorted(base_iterations)
    
    lines = []
    lines.append(f"# {dataset_name} Results\n")
    
    # Create tables for each checkpoint iteration and split
    for base_iter in sorted_base_iterations:
        iteration_display = f"ours_{base_iter}"
        lines.append(f"## Checkpoint: {iteration_display} ({base_iter} iterations)\n")

        for split in ["test", "train"]:
            split_key = f"ours_{base_iter}_{split}"
            # Table header
            lines.append(f"### {split.title()} set\n")
            lines.append("| Scene | PSNR ↑ | SSIM ↑ | LPIPS ↓ |")
            lines.append("|-------|--------|--------|---------|")
            
            # Collect data for this iteration/split
            scene_metrics = []
            for scene, scene_data in results.items():
                if split_key in scene_data:
                    metrics = scene_data[split_key]
                    psnr = metrics.get('PSNR', 0)
                    ssim = metrics.get('SSIM', 0)
                    lpips = metrics.get('LPIPS', 0)
                    scene_metrics.append((scene, psnr, ssim, lpips))
            
            # Sort scenes
            if args.sort_by == "psnr":
                scene_metrics.sort(key=lambda x: x[1], reverse=True)
            elif args.sort_by == "ssim":
                scene_metrics.sort(key=lambda x: x[2], reverse=True)
            else:
                scene_metrics.sort(key=lambda x: x[0])
            
            # Add rows
            psnr_sum = ssim_sum = lpips_sum = 0
            count = 0
            for scene, psnr_v, ssim_v, lpips_v in scene_metrics:
                lines.append(f"| {scene} | {psnr_v:.4f} | {ssim_v:.4f} | {lpips_v:.4f} |")
                psnr_sum += psnr_v
                ssim_sum += ssim_v
                lpips_sum += lpips_v
                count += 1
            
            # Add average row
            if count > 0:
                avg_psnr = psnr_sum / count
                avg_ssim = ssim_sum / count
                avg_lpips = lpips_sum / count
                lines.append(f"| **Average** | **{avg_psnr:.4f}** | **{avg_ssim:.4f}** | **{avg_lpips:.4f}** |")
            lines.append("")  # Empty line between subtables
        lines.append("")  # Empty line between iterations
    
    # Create comparison tables across all iterations for each split
    if len(sorted_base_iterations) > 1:
        for split in ["test", "train"]:
            lines.append(f"## Summary: Average {split.title()} Metrics Across All Scenes\n")
            lines.append("| Checkpoint | PSNR ↑ | SSIM ↑ | LPIPS ↓ |")
            lines.append("|------------|--------|--------|---------|")
            for base_iter in sorted_base_iterations:
                split_key = f"ours_{base_iter}_{split}"
                psnr_sum = ssim_sum = lpips_sum = 0
                count = 0
                for scene, scene_data in results.items():
                    if split_key in scene_data:
                        metrics = scene_data[split_key]
                        psnr_sum += metrics.get('PSNR', 0)
                        ssim_sum += metrics.get('SSIM', 0)
                        lpips_sum += metrics.get('LPIPS', 0)
                        count += 1
                if count > 0:
                    avg_psnr = psnr_sum / count
                    avg_ssim = ssim_sum / count
                    avg_lpips = lpips_sum / count
                    iteration_display = f"{base_iter // 1000}K"
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
    
    # Generate markdown table
    markdown = format_markdown_table(results, args.dataset)
    
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

