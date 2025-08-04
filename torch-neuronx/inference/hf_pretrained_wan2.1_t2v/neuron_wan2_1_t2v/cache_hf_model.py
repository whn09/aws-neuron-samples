import torch
from diffusers import AutoencoderKLWan, WanPipeline

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DTYPE = torch.bfloat16
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="wan2.1_t2v_hf_cache_dir")
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE, cache_dir="wan2.1_t2v_hf_cache_dir")
