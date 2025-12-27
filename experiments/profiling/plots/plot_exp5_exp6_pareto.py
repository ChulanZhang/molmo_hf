"""
Generate latency vs accuracy Pareto frontiers for Exp5/Exp6 results.

Input directory structure (produced by run_all_experiments.sh):
    results/profiling/exp5_exp6_2run/
        exp5_accuracy_<benchmark>/exp5_accuracy_results.json
        exp6_latency_<benchmark>/exp6_latency_results.json

For each benchmark, this script:
    1) Joins accuracy (Exp5) and latency (Exp6) summaries by config
       (max_crops, top_k, num_active_blocks).
    2) Computes Pareto frontiers for prefill, decode, and total latency
       (using P50 latency).
    3) Annotates the figure with average decode length
       (mean num_output_tokens from latency all_samples).
    4) Saves one PNG per benchmark under experiments/profiling/plots/.

Usage:
    python3 experiments/profiling/plots/plot_exp5_exp6_pareto.py \
        --results-dir /home/x-pwang1/ai_project/molmo_hf/results/profiling/exp5_exp6_2run \
        --output-dir  /home/x-pwang1/ai_project/molmo_hf/experiments/profiling/plots
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt  # type: ignore


ConfigKey = Tuple[int, int, int]  # (max_crops, top_k, num_active_blocks)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _load_latency_entries(latency_dir: Path) -> Tuple[List[Dict], Iterable[Dict]]:
    """Load latency summaries + per-sample rows (supports aggregated or rank files)."""
    aggregated = latency_dir / "exp6_latency_results.json"
    if aggregated.exists():
        data = _load_json(aggregated)
        return data["summary"], data.get("all_samples", [])

    per_config_files = sorted(latency_dir.glob("exp6_latency_crops*_topk*_blocks*_rank*.json"))
    if not per_config_files:
        raise FileNotFoundError(f"No latency files found in {latency_dir}")

    summaries: List[Dict] = []
    per_sample: List[Dict] = []
    for path in per_config_files:
        data = _load_json(path)
        summaries.extend(data.get("summary", []))
        per_sample.extend(data.get("per_sample_latencies", []))
    return summaries, per_sample


def _load_accuracy_entries(acc_dir: Path) -> List[Dict]:
    aggregated = acc_dir / "exp5_accuracy_results.json"
    if aggregated.exists():
        return _load_json(aggregated)["summary"]

    per_config_files = sorted(acc_dir.glob("exp5_accuracy_results_crops*_topk*_blocks*_rank*.json"))
    if not per_config_files:
        raise FileNotFoundError(f"No accuracy files found in {acc_dir}")

    summaries: List[Dict] = []
    for path in per_config_files:
        data = _load_json(path)
        summaries.extend(data.get("summary", []))
    return summaries


def load_benchmark(
    base_dir: Path, benchmark: str
) -> Tuple[List[Dict], float]:
    """Load accuracy + latency results and return merged rows + avg decode length."""
    acc_dir = base_dir / f"exp5_accuracy_{benchmark}"
    lat_dir = base_dir / f"exp6_latency_{benchmark}"

    if not acc_dir.exists() or not lat_dir.exists():
        raise FileNotFoundError(f"Missing results for {benchmark}")

    acc_entries = _load_accuracy_entries(acc_dir)
    lat_entries, all_samples = _load_latency_entries(lat_dir)

    acc_map = {
        (e["max_crops"], e["top_k"], e["num_active_blocks"]): e for e in acc_entries
    }
    lat_map = {
        (e["max_crops"], e["top_k"], e["num_active_blocks"]): e for e in lat_entries
    }

    common = sorted(acc_map.keys() & lat_map.keys())
    if not common:
        raise RuntimeError(f"No overlapping configs between accuracy and latency for {benchmark}")

    rows: List[Dict] = []
    for key in common:
        acc_entry = acc_map[key]
        lat_entry = lat_map[key]
        rows.append(
            {
                "max_crops": key[0],
                "top_k": key[1],
                "num_active_blocks": key[2],
                "accuracy": acc_entry["accuracy"],
                "latency_prefill": lat_entry["latency_prefill_ms"]["P50"],
                "latency_decode": lat_entry["latency_decode_ms"]["P50"],
                "latency_total": lat_entry["latency_total_ms"]["P50"],
            }
        )

    if all_samples:
        avg_decode_len = sum(s.get("num_output_tokens", 0) for s in all_samples) / len(
            all_samples
        )
    else:
        avg_decode_len = 0.0

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


def plot_benchmark(
    benchmark: str,
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
        
        # Estimate text box size (approximate)
        # For fontsize=8, typical tuple like "(2, 4, 12)" is about 60-80 pixels wide, 20-25 pixels tall
        # Convert to data coordinates
        fig_width_px = fig.get_figwidth() * fig.dpi
        fig_height_px = fig.get_figheight() * fig.dpi
        ax_width_data = x_range
        ax_height_data = y_range
        text_width_data = ax_width_data * 0.08  # Approximate text width in data coordinates
        text_height_data = ax_height_data * 0.03  # Approximate text height in data coordinates
        
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
            f"Benchmark: {benchmark}\n"
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
        ax.set_title(f"{benchmark} - {metric_name} vs Accuracy", fontsize=12, fontweight="bold", pad=10)
        
        fig.tight_layout()
        out_path = output_dir / f"exp5_exp6_pareto_{benchmark}_{metric_name.lower().replace(' ', '_')}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_path)

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Plot Exp5/Exp6 Pareto frontiers.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/profiling/exp5_exp6_2run"),
        help="Directory that contains exp5_accuracy_* and exp6_latency_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/profiling/plots"),
        help="Directory to write PNG plots.",
    )
    args = parser.parse_args()

    accuracy_dirs = sorted(args.results_dir.glob("exp5_accuracy_*"))
    benchmarks = [d.name.replace("exp5_accuracy_", "") for d in accuracy_dirs]

    if not benchmarks:
        raise RuntimeError(f"No exp5_accuracy_* directories found under {args.results_dir}")

    for benchmark in benchmarks:
        rows, avg_decode_len = load_benchmark(args.results_dir, benchmark)
        out_paths = plot_benchmark(benchmark, rows, avg_decode_len, args.output_dir)
        for out_path in out_paths:
            print(f"[OK] {benchmark}: wrote {out_path}")


if __name__ == "__main__":
    main()


