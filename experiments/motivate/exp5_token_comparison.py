
import json
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def analyze_comparison(phase2_path, phase3_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        p2_data = load_json(phase2_path)
        p3_data = load_json(phase3_path)
    except FileNotFoundError as e:
        log.error(f"Could not load data: {e}")
        return

    # Process Phase 2 (Vision Scaling)
    # We want: Delta Vision Tokens -> Delta Latency (Prefill)
    p2_results = p2_data.get("results", [])
    vision_points = []
    
    # Group by resolution/tokens
    p2_groups = {}
    for r in p2_results:
        vt = r["num_vision_tokens"]
        if vt not in p2_groups:
            p2_groups[vt] = []
        p2_groups[vt].append(r)
    
    # Calculate means
    base_vision_tokens = min(p2_groups.keys())
    base_vision_latency = np.mean([r["T_total"] for r in p2_groups[base_vision_tokens]]) # Use Total or Prefill?
    # For vision scaling, T_total is dominated by prefill changes, but let's look at T_total to be fair comparison with Decode
    # Actually, let's look at T_LLM_prefill + T_vision + T_projector for Vision cost
    
    vision_x = []
    vision_y = []
    
    sorted_v_tokens = sorted(p2_groups.keys())
    base_tokens = sorted_v_tokens[0]
    base_lat = np.mean([r["T_total"] for r in p2_groups[base_tokens]])
    
    for vt in sorted_v_tokens:
        mean_lat = np.mean([r["T_total"] for r in p2_groups[vt]])
        vision_x.append(vt - base_tokens)
        vision_y.append(mean_lat - base_lat)

    # Process Phase 3 (Language Scaling)
    # We want: Delta Output Tokens -> Delta Latency (Decode)
    p3_results = p3_data.get("results", [])
    
    p3_groups = {}
    for r in p3_results:
        ot = r["max_new_tokens"] # Use max_new_tokens as the controlled variable
        if ot not in p3_groups:
            p3_groups[ot] = []
        p3_groups[ot].append(r)
        
    language_x = []
    language_y = []
    
    sorted_l_tokens = sorted(p3_groups.keys())
    base_l_tokens = sorted_l_tokens[0]
    base_l_lat = np.mean([r["T_total"] for r in p3_groups[base_l_tokens]])
    
    for ot in sorted_l_tokens:
        mean_lat = np.mean([r["T_total"] for r in p3_groups[ot]])
        language_x.append(ot - base_l_tokens)
        language_y.append(mean_lat - base_l_lat)
        
    # Plot 1: Vision Scaling
    plt.figure(figsize=(8, 6))
    color = 'tab:blue'
    plt.plot(vision_x, vision_y, 'o-', label='Vision Tokens', color=color, linewidth=2)
    plt.xlabel('Added Vision Tokens', fontsize=12)
    plt.ylabel('Increase in Latency (ms)', fontsize=12)
    plt.title('Vision Scaling (Prefill)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    slope_v, _, _, _, _ = linregress(vision_x, vision_y)
    plt.text(0.05, 0.95, f"Cost: {slope_v:.3f} ms/token", 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_dir / "exp5_vision_scaling.png", dpi=300)
    log.info(f"Saved plot to {output_dir / 'exp5_vision_scaling.png'}")
    plt.close()

    # Plot 2: Language Scaling
    plt.figure(figsize=(8, 6))
    color = 'tab:red'
    plt.plot(language_x, language_y, 's-', label='Language Tokens', color=color, linewidth=2)
    plt.xlabel('Added Output Tokens', fontsize=12)
    plt.ylabel('Increase in Latency (ms)', fontsize=12)
    plt.title('Language Scaling (Decode)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    slope_l, _, _, _, _ = linregress(language_x, language_y)
    plt.text(0.05, 0.95, f"Cost: {slope_l:.3f} ms/token", 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_dir / "exp5_language_scaling.png", dpi=300)
    log.info(f"Saved plot to {output_dir / 'exp5_language_scaling.png'}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2_results", type=str, required=True, help="Path to Phase 2 JSON")
    parser.add_argument("--phase3_results", type=str, required=True, help="Path to Phase 3 JSON")
    parser.add_argument("--output_dir", type=str, default="results/exp5")
    args = parser.parse_args()
    
    analyze_comparison(args.phase2_results, args.phase3_results, args.output_dir)
