# imports
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

import neuronx_distributed
import numpy as npy
import os
import time
import torch
import torch_neuronx

from neuron_wan2_1_t2v.neuron_commons import InferenceTextEncoderWrapper
from neuron_wan2_1_t2v.neuron_commons import InferenceTransformerWrapper
from neuron_wan2_1_t2v.neuron_commons import SimpleWrapper

COMPILED_MODELS_DIR = "compile_workdir_latency_optimized"
HUGGINGFACE_CACHE_DIR = "wan2.1_t2v_hf_cache_dir"

if __name__ == "__main__":
    # os.environ["LOCAL_WORLD_SIZE"] = "8"
    
    DTYPE=torch.bfloat16
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="wan2.1_t2v_hf_cache_dir")
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE, cache_dir="wan2.1_t2v_hf_cache_dir")
    
    text_encoder_model_path = f"{COMPILED_MODELS_DIR}/text_encoder"
    transformer_model_path = f"{COMPILED_MODELS_DIR}/transformer" 
    decoder_model_path = f"{COMPILED_MODELS_DIR}/decoder/model.pt"
    post_quant_conv_model_path = f"{COMPILED_MODELS_DIR}/post_quant_conv/model.pt"
    
    seqlen=77  # default: 300
    text_encoder_wrapper = InferenceTextEncoderWrapper(
        torch.bfloat16, pipe.text_encoder, seqlen
    )
    
    print('text_encoder_wrapper.t start ****************')
    text_encoder_wrapper.t = neuronx_distributed.trace.parallel_model_load(
        text_encoder_model_path
    )
    # text_encoder_wrapper.t = torch_neuronx.DataParallel( 
    #     # torch.jit.load(os.path.join(text_encoder_model_path, 'model.pt')), [0, 1, 2, 3], False  # Use for trn2
    #     # torch.jit.load(os.path.join(text_encoder_model_path, 'model.pt')), [0, 1, 2, 3, 4, 5, 6, 7], False # Use for trn1/inf2
    #     torch.jit.load(os.path.join(text_encoder_model_path, 'model.pt'), [1])  # model.pt
    # )
    print('text_encoder_wrapper.t end ****************')

    transformer_wrapper = InferenceTransformerWrapper(pipe.transformer)
    print('transformer_wrapper.transformer start ****************')
    transformer_wrapper.transformer = neuronx_distributed.trace.parallel_model_load(
        transformer_model_path
    )
    # transformer_wrapper.transformer = torch_neuronx.DataParallel( 
    #     # torch.jit.load(os.path.join(transformer_model_path, 'model.pt')), [0, 1, 2, 3], False  # Use for trn2
    #     # torch.jit.load(os.path.join(transformer_model_path, 'model.pt')), [0, 1, 2, 3, 4, 5, 6, 7], False # Use for trn1/inf2
    #     torch.jit.load(os.path.join(transformer_model_path, 'model.pt'))
    # )
    print('transformer_wrapper.transformer end ****************')

    vae_decoder_wrapper = SimpleWrapper(pipe.vae.decoder)
    print('vae_decoder_wrapper.model start ****************')
    vae_decoder_wrapper.model = torch_neuronx.DataParallel( 
        torch.jit.load(decoder_model_path), [0, 1, 2, 3], False  # Use for trn2
        # torch.jit.load(decoder_model_path), [0, 1, 2, 3, 4, 5, 6, 7], False # Use for trn1/inf2
        # torch.jit.load(decoder_model_path),
    )
    print('vae_decoder_wrapper.model end ****************')
    
    vae_post_quant_conv_wrapper = SimpleWrapper(pipe.vae.post_quant_conv)
    print('vae_post_quant_conv_wrapper.model start ****************')
    vae_post_quant_conv_wrapper.model = torch_neuronx.DataParallel(
        torch.jit.load(post_quant_conv_model_path), [0, 1, 2, 3], False # Use for trn2
        # torch.jit.load(post_quant_conv_model_path), [0, 1, 2, 3, 4, 5, 6, 7], False # Use for trn1/inf2
        # torch.jit.load(post_quant_conv_model_path),
    )
    print('vae_post_quant_conv_wrapper.model end ****************')
    
    pipe.text_encoder = text_encoder_wrapper
    pipe.transformer = transformer_wrapper
    pipe.vae.decoder = vae_decoder_wrapper
    pipe.vae.post_quant_conv = vae_post_quant_conv_wrapper
    
    # os.environ["LOCAL_WORLD_SIZE"] = "8"
    
    prompt = "A cat walks on the grass, realistic"
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    start = time.time()
    output_warmup = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=256,  # default: 480
        width=256,  # default: 832
        num_frames=81,
        guidance_scale=5.0,
        max_sequence_length=seqlen  # default: 512
    ).frames[0]
    end = time.time()
    print('warmup time:', end-start)

    start = time.time()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=256,  # default: 480
        width=256,  # default: 832
        num_frames=81,
        guidance_scale=5.0,
        max_sequence_length=seqlen  # default: 512
    ).frames[0]
    end = time.time()
    print('time:', end-start)
    export_to_video(output, "output.mp4", fps=15)
