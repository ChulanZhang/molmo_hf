
from transformers import AutoProcessor
model_path = "allenai/MolmoE-1B-0924"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
config = processor.image_processor.model_config
print(f"Pooling H: {config.image_pooling_h}")
print(f"Pooling W: {config.image_pooling_w}")
print(f"Patch Size: {config.vision_backbone.image_patch_size}")
print(f"Default Input Size: {config.image_default_input_size}")
