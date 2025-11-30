
import torch
import logging
from PIL import Image
from transformers import AutoProcessor
from olmo import Molmo
from olmo.config import ModelConfig
import types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("MolmoFlow")

def print_separator(title):
    log.info(f"\n{'='*20} {title} {'='*20}")

def inspect_molmo_flow():
    model_path_hf = "allenai/MolmoE-1B-0924"
    model_path_molmo = "hf:allenai/MolmoE-1B-0924"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print_separator("1. Loading Model & Config")
    # Load processor first to get config
    processor = AutoProcessor.from_pretrained(model_path_hf, trust_remote_code=True)
    model = Molmo.from_checkpoint(model_path_molmo, device=device)
    
    # Debug: print attributes
    log.info(f"Image Processor Attributes: {dir(processor.image_processor)}")
    
    # Try to get config from image_processor directly if possible, or fallback
    if hasattr(processor.image_processor, 'image_default_input_size'):
         h, w = processor.image_processor.image_default_input_size
    else:
         # Hardcode fallback for MolmoE-1B if not found, or try to read from v_cfg if available
         # v_cfg usually has it?
         # Let's try to print v_cfg attributes too
         log.info(f"Vision Config Attributes: {dir(v_cfg)}")
         h, w = (336, 336) # Default fallback
         
    log.info(f"Model Config:")
    log.info(f"  - Image Patch Size: {v_cfg.image_patch_size}")
    log.info(f"  - Image Default Input Size: {(h, w)}")
    log.info(f"  - Image Emb Dim: {v_cfg.image_emb_dim}")
    log.info(f"  - Pooling Type: {config.image_pooling_2d}")
    log.info(f"  - Pooling H/W: {config.image_pooling_h}x{config.image_pooling_w}")
    
    # Calculate expected tokens
    h, w = proc_config.image_default_input_size
    patch_size = v_cfg.image_patch_size
    grid_h = h // patch_size
    grid_w = w // patch_size
    tokens_per_crop = grid_h * grid_w
    log.info(f"  - Calculated Grid: {grid_h}x{grid_w} = {tokens_per_crop} tokens per crop")

    # Monkey-patch MolmoVisionBackbone.forward to probe internals
    original_vision_forward = model.vision_backbone.forward
    
    def debug_vision_forward(self, images, image_masks):
        print_separator("3. Vision Backbone Forward (Monkey-Patched)")
        log.info(f"[Input] images shape: {images.shape} (B, Num_Crops, H, W, C)? No, check dim")
        # images shape is usually (B, T, N, D) or similar depending on stage. 
        # Actually encode_image takes (B, T, N, D) where N is num_patches? 
        # Let's check encode_image input in original code.
        # Wait, processor outputs pixel values. 
        
        # Let's call original encode_image first to see what it does
        # But we want to see INSIDE forward.
        
        # We will copy the logic of forward roughly or just wrap sections?
        # Wrapping is safer.
        
        # 1. Check Input
        log.info(f"Step 3.1: Input Images: {images.shape}")
        
        # 2. Run encode_image (ViT)
        # We can't easily probe inside encode_image without patching it too, 
        # but let's assume encode_image returns raw features.
        image_features = self.encode_image(images)
        log.info(f"Step 3.2: After ViT (encode_image): {image_features.shape}")
        log.info(f"         -> Expecting (B, T, N, D) where T=Crops, N=Tokens/Crop")
        
        # 3. Continue with original forward logic for the rest... 
        # Actually, to probe the pooling, we need to replicate the forward logic 
        # OR patch the pooling layer itself.
        return original_vision_forward(images, image_masks)

    # Better approach: Patch the pooling layer's forward
    original_pooling_forward = model.vision_backbone.image_pooling_2d.forward
    
    def debug_pooling_forward(query, key_value):
        print_separator("4. Pooling Layer (Attention-MeanQ)")
        log.info(f"[Input] Query shape: {query.shape}")
        log.info(f"[Input] Key/Value shape: {key_value.shape}")
        
        output = original_pooling_forward(query, key_value)
        
        log.info(f"[Output] Pooled shape: {output.shape}")
        return output
    
    # Apply patches
    # model.vision_backbone.forward = types.MethodType(debug_vision_forward, model.vision_backbone)
    # We won't patch top-level forward because it's complex. 
    # We will patch `encode_image` and `image_pooling_2d`.
    
    original_encode_image = model.vision_backbone.encode_image
    def debug_encode_image(self, images):
        print_separator("3. Vision Encoder (ViT)")
        log.info(f"[Input] images: {images.shape}")
        output = original_encode_image(images)
        log.info(f"[Output] Raw Features: {output.shape}")
        return output
        
    model.vision_backbone.encode_image = types.MethodType(debug_encode_image, model.vision_backbone)
    model.vision_backbone.image_pooling_2d.forward = debug_pooling_forward

    # Create Dummy Input
    print_separator("2. Processing Input")
    text = "Describe this image."
    # 2000x2000 to trigger max crops
    image = Image.new('RGB', (2000, 2000), color='blue')
    
    inputs = processor.process(text=text, images=image, return_tensors="pt")
    inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}
    
    log.info(f"Processor Output:")
    log.info(f"  - input_ids: {inputs['input_ids'].shape}")
    log.info(f"  - images: {inputs['images'].shape}")
    log.info(f"  - image_masks: {inputs['image_masks'].shape}")

    # Run Model
    print_separator("5. Full Model Forward")
    with torch.no_grad():
        # We only need to run the embedding/vision part, but running full model is easiest
        model(
            input_ids=inputs['input_ids'],
            images=inputs['images'],
            image_masks=inputs['image_masks'],
            image_input_idx=inputs['image_input_idx']
        )

if __name__ == "__main__":
    inspect_molmo_flow()
