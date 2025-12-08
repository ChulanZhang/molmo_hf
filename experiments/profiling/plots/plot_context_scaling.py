
import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_results(json_file, output_file):
    if not os.path.exists(json_file):
        print(f"File {json_file} not found.")
        return

    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract data
    tokens = [d['num_input_tokens'] for d in data]
    prefill_latency = [d['P50'] for d in data] # Using P50 (median)
    
    plt.figure(figsize=(10, 6))
    plt.plot(tokens, prefill_latency, marker='o', linestyle='-', color='b', label='LLM Prefill Latency')
    
    plt.xlabel('Number of Input Tokens')
    plt.ylabel('Latency (ms)')
    plt.title('LLM Prefill Latency vs. Context Length')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/context_scaling/exp1_context_scaling_results.json")
    parser.add_argument("--output", type=str, default="results/context_scaling/context_scaling_plot.png")
    args = parser.parse_args()
    
    plot_results(args.input, args.output)
