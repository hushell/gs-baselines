#!/usr/bin/env python3
# Summary script to collect results.json files and generate Markdown tables

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

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
    """Format results as a Markdown table"""
    if not results:
        return "No results found."
    
    # Get all unique checkpoint iterations across all scenes
    all_iterations = set()
    for scene_data in results.values():
        all_iterations.update(scene_data.keys())
    
    # Sort iterations (assuming format "ours_XXXXX")
    sorted_iterations = sorted(all_iterations, key=lambda x: int(x.split('_')[-1]) if '_' in x else 0)
    
    lines = []
    lines.append(f"# {dataset_name} Results\n")
    
    # Create a table for each checkpoint iteration
    for iteration in sorted_iterations:
        iteration_num = iteration.split('_')[-1] if '_' in iteration else iteration
        lines.append(f"## Checkpoint: {iteration} ({iteration_num} iterations)\n")
        
        # Table header
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
                scene_metrics.append((scene, psnr, ssim, lpips))
        
        # Sort scenes
        if args.sort_by == "psnr":
            scene_metrics.sort(key=lambda x: x[1], reverse=True)
        elif args.sort_by == "ssim":
            scene_metrics.sort(key=lambda x: x[2], reverse=True)
        else:  # name
            scene_metrics.sort(key=lambda x: x[0])
        
        # Add rows
        psnr_sum = ssim_sum = lpips_sum = 0
        count = 0
        for scene, psnr, ssim, lpips in scene_metrics:
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
            lines.append(f"| **Average** | **{avg_psnr:.4f}** | **{avg_ssim:.4f}** | **{avg_lpips:.4f}** |")
        
        lines.append("")  # Empty line between tables
    
    # Create a comparison table across all iterations
    if len(sorted_iterations) > 1:
        lines.append("## Summary: Average Metrics Across All Scenes\n")
        lines.append("| Checkpoint | PSNR ↑ | SSIM ↑ | LPIPS ↓ |")
        lines.append("|------------|--------|--------|---------|")
        
        for iteration in sorted_iterations:
            psnr_sum = ssim_sum = lpips_sum = 0
            count = 0
            
            for scene_data in results.values():
                if iteration in scene_data:
                    metrics = scene_data[iteration]
                    psnr_sum += metrics.get('PSNR', 0)
                    ssim_sum += metrics.get('SSIM', 0)
                    lpips_sum += metrics.get('LPIPS', 0)
                    count += 1
            
            if count > 0:
                avg_psnr = psnr_sum / count
                avg_ssim = ssim_sum / count
                avg_lpips = lpips_sum / count
                iteration_display = iteration.split('_')[-1] if '_' in iteration else iteration
                lines.append(f"| {iteration_display}K | {avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |")
        
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

