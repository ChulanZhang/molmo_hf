
import os
import sys
import torch
import numpy as np
from PIL import Image

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

print("Importing transformers...", flush=True)
from transformers import AutoProcessor, AutoTokenizer

def analyze_tokens():
    print("Starting analysis...", flush=True)
    
    model_path = "allenai/MolmoE-1B-0924"
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"Model Path: {model_path}", flush=True)
    print(f"Cache Dir: {cache_dir}", flush=True)
    
    print("Loading processor (this may take time if downloading)...", flush=True)
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True,
            device_map="auto"
        )
        print("Processor loaded successfully.", flush=True)
    except Exception as e:
        print(f"Error loading processor: {e}", flush=True)
        return

    tokenizer = processor.tokenizer

    # Create dummy input
    text = "Describe this image."
    # Use large image to trigger max crops (e.g. 2000x2000)
    image = Image.new('RGB', (2000, 2000), color='red')

    print("Processing input (Large Image + Padding)...", flush=True)
    # Simulate DataConfig padding behavior
    inputs = processor.process(
        text=text, 
        images=image, 
        padding="max_length", 
        max_length=1536, 
        return_tensors="pt"
    )
    input_ids = inputs['input_ids']
    images = inputs['images']

    print(f"\nInput IDs shape: {input_ids.shape}", flush=True)
    if images is not None:
        print(f"Images shape: {images.shape}", flush=True)
        num_crops = images.shape[1]
        print(f"Num Crops: {num_crops}", flush=True)
        print(f"Vision Tokens: {num_crops * 576}", flush=True)

    
    # Decode tokens
    print("\n--- Token Analysis ---", flush=True)
    if input_ids.dim() == 2:
        tokens = input_ids[0].tolist()
    else:
        tokens = input_ids.tolist()
    
    decoded_text = tokenizer.decode(tokens, skip_special_tokens=False)
    
    print(f"Total Tokens: {len(tokens)}", flush=True)
    print(f"Decoded Text (Full):\n{decoded_text}", flush=True)
    
    # Analyze composition
    special_tokens = set(tokenizer.all_special_ids)
    
    print("\n--- Detailed Breakdown (All tokens) ---", flush=True)
    for i, token_id in enumerate(tokens):
        token_str = tokenizer.decode([token_id])
        is_special = token_id in special_tokens
        special_str = " (SPECIAL)" if is_special else ""
        print(f"{i}: {token_id} -> '{token_str}'{special_str}", flush=True)
    
    # Check for image tokens
    unique, counts = np.unique(tokens, return_counts=True)
    print("\n--- Most Frequent Tokens ---", flush=True)
    sorted_indices = np.argsort(-counts)
    for i in range(min(10, len(unique))):
        idx = sorted_indices[i]
        token_id = unique[idx]
        count = counts[idx]
        token_str = tokenizer.decode([token_id])
        print(f"Token {token_id} ('{token_str}'): {count} times", flush=True)

if __name__ == "__main__":
    analyze_tokens()
