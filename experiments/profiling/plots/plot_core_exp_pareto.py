"""
Generate latency vs accuracy Pareto frontiers for core_exp combined_profiling_results.

Input directory structure:
    results/core_exp_h100/1run_12samples/
        <task>/
            combined_profiling_results_*.json

For each task, this script:
    1) Loads all combined_profiling_results_*.json files
    2) Extracts config (max_crops, top_k, num_active_blocks), accuracy, and latencies
    3) Computes Pareto frontiers for prefill, decode, and total latency (using P50)
    4) Saves separate PNG files for each latency type

Usage:
    python3 experiments/profiling/plots/plot_core_exp_pareto.py \
        --results-dir /home/x-pwang1/ai_project/molmo_hf/results/core_exp_h100/1run_12samples \
        --output-dir /home/x-pwang1/ai_project/molmo_hf/experiments/profiling/plots
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import glob

import matplotlib.pyplot as plt  # type: ignore


ConfigKey = Tuple[int, int, int]  # (max_crops, top_k, num_active_blocks)


def load_task_results(task_dir: Path) -> Tuple[List[Dict], float]:
    """Load all combined_profiling_results_*.json files for a task."""
    result_files = sorted(task_dir.glob("combined_profiling_results_*.json"))
    
    if not result_files:
        raise FileNotFoundError(f"No combined_profiling_results_*.json files found in {task_dir}")
    
    rows: List[Dict] = []
    total_output_tokens = 0
    total_samples = 0
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract config
            max_crops = data.get("max_crops", 0)
            top_k = data.get("top_k", 0)
            num_active_blocks = data.get("num_active_blocks", 0)
            accuracy = data.get("accuracy", 0.0)
            
            # Extract latencies from aggregate_stats
            agg_stats = data.get("aggregate_stats", {})
            latency_prefill = agg_stats.get("T_LLM_prefill_p50", 0.0)
            latency_decode = agg_stats.get("T_LLM_decode_p50", 0.0)
            latency_total = agg_stats.get("T_total_p50", 0.0)
            
            rows.append({
                "max_crops": max_crops,
                "top_k": top_k,
                "num_active_blocks": num_active_blocks,
                "accuracy": accuracy,
                "latency_prefill": latency_prefill,
                "latency_decode": latency_decode,
                "latency_total": latency_total,
            })
            
            # Calculate average decode length from per_sample_results
            per_sample = data.get("per_sample_results", [])
            for sample in per_sample:
                output_tokens = sample.get("output_tokens", 0)
                total_output_tokens += output_tokens
                total_samples += 1
                
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}")
            continue
    
    avg_decode_len = total_output_tokens / total_samples if total_samples > 0 else 0.0
    
    return rows, avg_decode_len


def pareto_frontier(rows: List[Dict], latency_key: str) -> List[Dict]:
    """Compute accuracy-maximizing frontier for given latency metric."""
    sorted_rows = sorted(rows, key=lambda r: r[latency_key])
    frontier: List[Dict] = []
    best_acc = -1.0
    for row in sorted_rows:
        if row["accuracy"] >= best_acc - 1e-9:
            frontier.append(row)
            best_acc = max(best_acc, row["accuracy"])
    return frontier


def plot_task(
    task_name: str,
    rows: List[Dict],
    avg_decode_len: float,
    output_dir: Path,
) -> List[Path]:
    """Create separate plots for prefill/decode/total Pareto lines."""
    metrics = [
        ("latency_prefill", "Prefill Latency", "Prefill latency (ms, P50)"),
        ("latency_decode", "Decode Latency", "Decode latency (ms, P50)"),
        ("latency_total", "Total Latency", "Total latency (ms, P50)"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for lat_key, metric_name, xlabel in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        x_vals = [r[lat_key] for r in rows]
        y_vals = [r["accuracy"] for r in rows]
        
        # Plot all configs
        ax.scatter(x_vals, y_vals, color="gray", alpha=0.5, s=50, label="All Configurations", zorder=1)

        # Compute and plot Pareto frontier
        frontier = pareto_frontier(rows, lat_key)
        fx = [r[lat_key] for r in frontier]
        fy = [r["accuracy"] for r in frontier]
        ax.plot(fx, fy, color="red", marker="o", linewidth=2, markersize=8, 
                label="Pareto Frontier", zorder=3)

        # Calculate ranges for smart annotation
        x_range = max(x_vals) - min(x_vals) if x_vals else 1.0
        y_range = max(y_vals) - min(y_vals) if y_vals else 1.0
        
        # Estimate text box size
        text_width_data = x_range * 0.08
        text_height_data = y_range * 0.03
        
        # Base offset
        base_offset_x = x_range * 0.04
        base_offset_y = y_range * 0.04
        
        # Store annotation positions to detect overlaps
        annotation_positions = []
        
        # Annotate frontier points with config details - use collision detection
        for i, r in enumerate(frontier):
            # Create compact label as tuple: (crops, top_k, blocks)
            label = f"({r['max_crops']}, {r['top_k']}, {r['num_active_blocks']})"
            
            x_pos = r[lat_key]
            y_pos = r["accuracy"]
            
            # Try 8 candidate positions around the point
            candidates = [
                # (text_x, text_y, ha, va, description)
                (x_pos + base_offset_x, y_pos + base_offset_y, "left", "bottom", "top-right"),
                (x_pos + base_offset_x, y_pos - base_offset_y, "left", "top", "bottom-right"),
                (x_pos - base_offset_x, y_pos + base_offset_y, "right", "bottom", "top-left"),
                (x_pos - base_offset_x, y_pos - base_offset_y, "right", "top", "bottom-left"),
                (x_pos, y_pos + base_offset_y * 1.5, "center", "bottom", "top"),
                (x_pos, y_pos - base_offset_y * 1.5, "center", "top", "bottom"),
                (x_pos + base_offset_x * 1.5, y_pos, "left", "center", "right"),
                (x_pos - base_offset_x * 1.5, y_pos, "right", "center", "left"),
            ]
            
            # Find the best position (least overlap)
            best_candidate = None
            min_overlap = float('inf')
            
            for text_x, text_y, ha, va, _ in candidates:
                # Calculate text box bounds based on alignment
                if ha == "left":
                    box_left = text_x
                    box_right = text_x + text_width_data
                elif ha == "right":
                    box_left = text_x - text_width_data
                    box_right = text_x
                else:  # center
                    box_left = text_x - text_width_data / 2
                    box_right = text_x + text_width_data / 2
                
                if va == "bottom":
                    box_bottom = text_y
                    box_top = text_y + text_height_data
                elif va == "top":
                    box_bottom = text_y - text_height_data
                    box_top = text_y
                else:  # center
                    box_bottom = text_y - text_height_data / 2
                    box_top = text_y + text_height_data / 2
                
                # Check overlap with existing annotations
                overlap_count = 0
                for existing_box in annotation_positions:
                    ex_left, ex_right, ex_bottom, ex_top = existing_box
                    # Check if boxes overlap
                    if not (box_right < ex_left or box_left > ex_right or 
                           box_top < ex_bottom or box_bottom > ex_top):
                        overlap_count += 1
                
                # Also check if too close to data points (avoid covering points)
                min_dist_to_points = float('inf')
                for other_r in frontier:
                    if other_r != r:
                        dist = ((text_x - other_r[lat_key])**2 + (text_y - other_r["accuracy"])**2)**0.5
                        min_dist_to_points = min(min_dist_to_points, dist)
                
                # Prefer positions with less overlap and reasonable distance from points
                score = overlap_count * 1000 + max(0, text_width_data * 0.5 - min_dist_to_points) * 100
                
                if score < min_overlap:
                    min_overlap = score
                    best_candidate = (text_x, text_y, ha, va)
            
            # If still overlapping, try increasing offset
            if min_overlap > 0:
                # Try with larger offsets
                for scale in [1.5, 2.0, 2.5, 3.0]:
                    for text_x, text_y, ha, va, _ in candidates:
                        text_x_scaled = x_pos + (text_x - x_pos) * scale
                        text_y_scaled = y_pos + (text_y - y_pos) * scale
                        
                        # Recalculate box bounds
                        if ha == "left":
                            box_left = text_x_scaled
                            box_right = text_x_scaled + text_width_data
                        elif ha == "right":
                            box_left = text_x_scaled - text_width_data
                            box_right = text_x_scaled
                        else:
                            box_left = text_x_scaled - text_width_data / 2
                            box_right = text_x_scaled + text_width_data / 2
                        
                        if va == "bottom":
                            box_bottom = text_y_scaled
                            box_top = text_y_scaled + text_height_data
                        elif va == "top":
                            box_bottom = text_y_scaled - text_height_data
                            box_top = text_y_scaled
                        else:
                            box_bottom = text_y_scaled - text_height_data / 2
                            box_top = text_y_scaled + text_height_data / 2
                        
                        # Check overlap
                        overlap_count = 0
                        for existing_box in annotation_positions:
                            ex_left, ex_right, ex_bottom, ex_top = existing_box
                            if not (box_right < ex_left or box_left > ex_right or 
                                   box_top < ex_bottom or box_bottom > ex_top):
                                overlap_count += 1
                        
                        if overlap_count == 0:
                            best_candidate = (text_x_scaled, text_y_scaled, ha, va)
                            min_overlap = 0
                            break
                    
                    if min_overlap == 0:
                        break
            
            # Use best candidate
            if best_candidate:
                text_x, text_y, ha, va = best_candidate
            else:
                # Fallback to first candidate
                text_x, text_y, ha, va, _ = candidates[0]
            
            # Store this annotation's position
            if ha == "left":
                box_left = text_x
                box_right = text_x + text_width_data
            elif ha == "right":
                box_left = text_x - text_width_data
                box_right = text_x
            else:
                box_left = text_x - text_width_data / 2
                box_right = text_x + text_width_data / 2
            
            if va == "bottom":
                box_bottom = text_y
                box_top = text_y + text_height_data
            elif va == "top":
                box_bottom = text_y - text_height_data
                box_top = text_y
            else:
                box_bottom = text_y - text_height_data / 2
                box_top = text_y + text_height_data / 2
            
            annotation_positions.append((box_left, box_right, box_bottom, box_top))
            
            ax.annotate(
                label,
                xy=(x_pos, y_pos),
                xytext=(text_x, text_y),
                fontsize=8,
                ha=ha,
                va=va,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", 
                         alpha=0.85, edgecolor="darkred", linewidth=1.2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", 
                              lw=1.2, color="darkred", alpha=0.7),
                zorder=4,
            )

        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
        
        # Enhanced legend with more details - place in bottom right
        legend_text = (
            f"Task: {task_name}\n"
            f"Avg decode length: {avg_decode_len:.1f} tokens\n"
            f"Config format: crops=vision_tokens, top_k=MoE_experts, blocks=transformer_layers"
        )
        ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8, 
                         edgecolor="gray", linewidth=1),
                zorder=5)
        
        # Move plot legend to upper left
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.set_title(f"{task_name} - {metric_name} vs Accuracy", fontsize=12, fontweight="bold", pad=10)
        
        fig.tight_layout()
        out_path = output_dir / f"core_exp_pareto_{task_name}_{metric_name.lower().replace(' ', '_')}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_path)

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Plot core_exp Pareto frontiers.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/core_exp_h100/1run_12samples"),
        help="Directory that contains task subdirectories with combined_profiling_results_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/profiling/plots"),
        help="Directory to write PNG plots.",
    )
    args = parser.parse_args()

    # Find all task directories
    task_dirs = [d for d in args.results_dir.iterdir() if d.is_dir()]
    
    if not task_dirs:
        raise RuntimeError(f"No task directories found under {args.results_dir}")

    for task_dir in sorted(task_dirs):
        task_name = task_dir.name
        print(f"Processing task: {task_name}")
        
        try:
            rows, avg_decode_len = load_task_results(task_dir)
            if not rows:
                print(f"  Warning: No valid results found for {task_name}")
                continue
            
            print(f"  Loaded {len(rows)} configurations")
            print(f"  Avg decode length: {avg_decode_len:.1f} tokens")
            
            out_paths = plot_task(task_name, rows, avg_decode_len, args.output_dir)
            for out_path in out_paths:
                print(f"[OK] {task_name}: wrote {out_path}")
        except Exception as e:
            print(f"  Error processing {task_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

