from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

# load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/MolmoE-1B-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# process the image and text
inputs = processor.process(
    images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
    text="Describe this image."
)

# move inputs to the correct device and make a batch of size 1
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

# === DEBUG: Add ipdb for interactive debugging ===
import ipdb

print("\n=== Model Structure Info ===")
print(f"Model type: {type(model)}")
print(f"Config block_type: {model.config.block_type}")
print(f"Config moe_top_k: {model.config.moe_top_k}")

# Check first transformer block
if hasattr(model.model, 'transformer'):
    first_block = model.model.transformer["blocks"][0]
    print(f"\nFirst block type: {type(first_block)}")
    if hasattr(first_block, 'mlp'):
        print(f"Block has 'mlp': {type(first_block.mlp)}")
        if hasattr(first_block.mlp, 'top_k'):
            print(f"  mlp.top_k = {first_block.mlp.top_k}")
    if hasattr(first_block, 'ffn'):
        print(f"Block has 'ffn': {type(first_block.ffn)}")

print("\n" + "="*80)
print("🔍 IPDB BREAKPOINT - Inspect model structure and modify top_k if needed")
print("="*80)
print("Useful commands:")
print("  - print(type(model.model.transformer['blocks'][0].mlp))  # Check block structure")
print("  - model.model.transformer['blocks'][0].mlp.top_k = 2     # Modify top_k")
print("  - c  # Continue execution")
print("="*80)

# SET BREAKPOINT HERE - you can inspect and modify the model
ipdb.set_trace()

# generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

# only get generated tokens; decode them to text
generated_tokens = output[0,inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# print the generated text
print(generated_text)

# >>> This photograph captures a small black puppy, likely a Labrador or a similar breed,
#     sitting attentively on a weathered wooden deck. The deck, composed of three...
