# imports
from diffusers import AutoencoderKLWan, WanPipeline

import neuronx_distributed
import numpy as npy
import time
import torch
import torch_neuronx

from neuron_wan2_1_t2v.neuron_commons import InferenceTextEncoderWrapper
from neuron_wan2_1_t2v.neuron_commons import InferenceTransformerWrapper
from neuron_wan2_1_t2v.neuron_commons import SimpleWrapper

COMPILED_MODELS_DIR = "compile_workdir_latency_optimized"
HUGGINGFACE_CACHE_DIR = "wan2.1_t2v_hf_cache_dir"

if __name__ == "__main__":
    DTYPE=torch.bfloat16
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="wan2.1_t2v_hf_cache_dir")
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE, cache_dir="wan2.1_t2v_hf_cache_dir")
    
    text_encoder_model_path = f"{COMPILED_MODELS_DIR}/text_encoder"
    transformer_model_path = f"{COMPILED_MODELS_DIR}/transformer" 
    decoder_model_path = f"{COMPILED_MODELS_DIR}/decoder/model.pt"
    post_quant_conv_model_path = f"{COMPILED_MODELS_DIR}/post_quant_conv/model.pt"

    seqlen=300
    text_encoder_wrapper = InferenceTextEncoderWrapper(
        torch.bfloat16, pipe.text_encoder, seqlen
    )
    
    text_encoder_wrapper.t = neuronx_distributed.trace.parallel_model_load(
        text_encoder_model_path
    )

    transformer_wrapper = InferenceTransformerWrapper(pipe.transformer)
    transformer_wrapper.transformer = neuronx_distributed.trace.parallel_model_load(
        transformer_model_path
    )

    vae_decoder_wrapper = SimpleWrapper(pipe.vae.decoder)
    vae_decoder_wrapper.model = torch_neuronx.DataParallel( 
        # torch.jit.load(decoder_model_path), [0, 1, 2, 3], False  # Use for trn2
        torch.jit.load(decoder_model_path), [0, 1, 2, 3, 4, 5, 6, 7], False # Use for trn1/inf2
    )
    
    vae_post_quant_conv_wrapper = SimpleWrapper(pipe.vae.post_quant_conv)
    vae_post_quant_conv_wrapper.model = torch_neuronx.DataParallel(
        # torch.jit.load(post_quant_conv_model_path), [0, 1, 2, 3], False # Use for trn2
        torch.jit.load(post_quant_conv_model_path), [0, 1, 2, 3, 4, 5, 6, 7], False # Use for trn1/inf2
    )
    
    pipe.text_encoder = text_encoder_wrapper
    pipe.transformer = transformer_wrapper
    pipe.vae.decoder = vae_decoder_wrapper
    pipe.vae.post_quant_conv = vae_post_quant_conv_wrapper
    
    # Run pipeline
    prompt = "a photo of an astronaut riding a horse on mars"
    negative_prompt = "mountains"
    
    # First do a warmup run so all the asynchronous loads can finish
    image_warmup = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        num_images_per_prompt=1, 
        height=480,
        width=640,
        num_inference_steps=25
    ).images[0]
    

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        height=480,
        width=640,
        num_inference_steps=25
    ).images
    
    for idx, img in enumerate(images): 
        img.save(f"image_{idx}.png")