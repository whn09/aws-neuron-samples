# source /opt/aws_neuronx_venv_pytorch_2_7/bin/activate
# pip install diffusers transformers sentencepiece matplotlib accelerate ftfy

import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"

compiler_flags = """ --verbose=INFO --target=trn1 --model-type=transformer --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn1/inf2
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import torch.nn as nn
import torch_neuronx
import numpy as np

import time
import copy

from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.models.autoencoders.autoencoder_kl_wan import WanUpsample
from diffusers.utils import export_to_video
from diffusers.models.autoencoders.vae import Decoder

# Compatibility for diffusers<0.18.0
from packaging import version
import diffusers
diffusers_version = version.parse(diffusers.__version__)
use_new_diffusers = diffusers_version >= version.parse('0.18.0')
if use_new_diffusers:
    from diffusers.models.attention_processor import Attention
else:
    from diffusers.models.cross_attention import CrossAttention

# Define datatype
DTYPE = torch.bfloat16

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel  # noqa: E402
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel  # noqa: E402
from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402
import math
import torch.nn.functional as F
from typing import Optional

_flash_fwd_call = nki_jit()(attention_isa_kernel)
def attention_wrapper_without_swap(query, key, value):
    bs, n_head, q_len, d_head = query.shape  # my change
    k_len = key.shape[2]
    v_len = value.shape[2]
    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, k_len))
    v = value.clone().reshape((bs * n_head, v_len, d_head))
    attn_output = torch.zeros((bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)

    scale = 1 / math.sqrt(d_head)
    _flash_fwd_call(q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")

    attn_output = attn_output.reshape((bs, n_head, q_len, d_head))

    return attn_output

# XLA兼容的rotary embedding实现
def apply_rotary_emb_xla_compatible(hidden_states: torch.Tensor, freqs: torch.Tensor):
    """XLA兼容的rotary embedding实现，避免使用view_as_complex"""
    # hidden_states: [batch, heads, seq_len, head_dim]
    # freqs: [1, 1, seq_len, head_dim//2] (complex frequencies)
    
    # 将hidden_states重塑为实数和虚数部分
    # 假设head_dim是偶数
    head_dim = hidden_states.shape[-1]
    
    # 将hidden_states按照奇偶索引分成实部和虚部
    x_real = hidden_states[..., 0::2]  # 偶数索引作为实部
    x_imag = hidden_states[..., 1::2]  # 奇数索引作为虚部
    
    # 如果freqs是复数格式，需要提取实部和虚部
    if freqs.dtype == torch.complex64 or freqs.dtype == torch.complex128:
        cos_freq = freqs.real
        sin_freq = freqs.imag
    else:
        # 如果freqs已经是实数格式，假设它包含cos和sin值
        freq_half_dim = freqs.shape[-1] // 2
        cos_freq = freqs[..., :freq_half_dim]
        sin_freq = freqs[..., freq_half_dim:]
    
    # 应用旋转：(x_real + i*x_imag) * (cos + i*sin) = (x_real*cos - x_imag*sin) + i*(x_real*sin + x_imag*cos)
    rotated_real = x_real * cos_freq - x_imag * sin_freq
    rotated_imag = x_real * sin_freq + x_imag * cos_freq
    
    # 重新交错实部和虚部
    batch_size, num_heads, seq_len = hidden_states.shape[:3]
    result = torch.zeros_like(hidden_states)
    result[..., 0::2] = rotated_real
    result[..., 1::2] = rotated_imag
    
    return result.type_as(hidden_states)

class KernelizedWanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("KernelizedWanAttnProcessor2_0 requires PyTorch 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # # TODO: 使用XLA兼容的rotary embedding
        # if rotary_emb is not None:
        #     query = apply_rotary_emb_xla_compatible(query, rotary_emb)
        #     key = apply_rotary_emb_xla_compatible(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            # 使用Neuron优化的注意力或回退到标准实现
            if (attention_mask is None and query.shape[3] <= 128 and 
                query.shape[3] <= query.shape[2] and value_img.shape[2] != 77):
                hidden_states_img = attention_wrapper_without_swap(query, key_img, value_img)
            else:
                hidden_states_img = F.scaled_dot_product_attention(
                    query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
                )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # 使用Neuron优化的注意力或回退到标准实现
        if (attention_mask is None and query.shape[3] <= 128 and 
            query.shape[3] <= query.shape[2] and value.shape[2] != 77):
            hidden_states = attention_wrapper_without_swap(query, key, value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

class TransformerWrap(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image=None):
        # WanTransformer3DModel的forward方法签名
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=False
        )

class NeuronTransformer(nn.Module):
    def __init__(self, transformer_wrap):
        super().__init__()
        self.transformer_wrap = transformer_wrap
        self.config = transformer_wrap.transformer.config
        if hasattr(transformer_wrap.transformer, 'in_channels'):
            self.in_channels = transformer_wrap.transformer.in_channels
        self.device = transformer_wrap.transformer.device
        
        # 替换Wan的注意力处理器为优化版本
        self._replace_attention_processors()

    def _replace_attention_processors(self):
        """替换所有WanAttnProcessor2_0为优化版本"""
        def replace_processor(module):
            if hasattr(module, 'processor') and hasattr(module.processor, '__class__'):
                if 'WanAttnProcessor2_0' in str(type(module.processor)):
                    module.processor = KernelizedWanAttnProcessor2_0()
            for child in module.children():
                replace_processor(child)
        
        replace_processor(self.transformer_wrap.transformer)

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image=None, 
                timestep_cond=None, added_cond_kwargs=None, cross_attention_kwargs=None, return_dict=False):
        sample = self.transformer_wrap(
            hidden_states, 
            timestep.to(dtype=DTYPE).expand((hidden_states.shape[0],)), 
            encoder_hidden_states,
            encoder_hidden_states_image
        )[0]
        
        if return_dict:
            return type('TransformerOutput', (), {'sample': sample})()
        return sample
    
class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]

# Optimized attention
def get_attention_scores(self, query, key, attn_mask):       
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    # Check for square matmuls
    if(query.size() == key.size()):
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)
        
    return attention_probs

# In the original badbmm the bias is all zeros, so only apply scale
def custom_badbmm(a, b):
    bmm = torch.bmm(a, b)
    scaled = bmm * 0.125
    return scaled


# For saving compiler artifacts
COMPILER_WORKDIR_ROOT = 'wan2.1_t2v_compile_dir'

# Model ID for SD version pipeline
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# These dimensions need to be adjusted based on Wan2.1's expected input format
batch_size = 1
frames = 16  # default: 16  # typical frame count for video generation
height, width = 32, 32  # default: 96, 96  # spatial dimensions
in_channels = 16  # 根据配置，Wan使用16个输入通道


# # --- Compile Transformer and save [PASS]---

# vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
# pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE)

# # Replace original cross-attention module with custom cross-attention module for better performance
# if use_new_diffusers:
#     Attention.get_attention_scores = get_attention_scores
# else:
#     CrossAttention.get_attention_scores = get_attention_scores

# # Apply double wrapper to deal with custom return type
# pipe.transformer = NeuronTransformer(TransformerWrap(pipe.transformer))

# # Only keep the model being compiled in RAM to minimze memory pressure
# transformer = copy.deepcopy(pipe.transformer.transformer_wrap)
# del pipe

# # Compile transformer - adjust input shapes for 3D video

# # 3D input for video: (batch, channels, frames, height, width)
# hidden_states_1b = torch.randn([batch_size, in_channels, frames, height, width], dtype=DTYPE)
# timestep_1b = torch.tensor(999, dtype=DTYPE).expand((batch_size,))
# # Text encoder output dimension for Wan (might be different from SD)
# encoder_hidden_states_1b = torch.randn([batch_size, 77, 4096], dtype=DTYPE)  # Wan uses 4096 dim

# example_inputs = hidden_states_1b, timestep_1b, encoder_hidden_states_1b

# transformer_neuron = torch_neuronx.trace(
#     transformer,
#     example_inputs,
#     compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'transformer'),
#     compiler_args=compiler_flags
# )

# # Enable asynchronous and lazy loading to speed up model load
# torch_neuronx.async_load(transformer_neuron)
# torch_neuronx.lazy_load(transformer_neuron)

# # save compiled transformer
# transformer_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'transformer/model.pt')
# torch.jit.save(transformer_neuron, transformer_filename)

# # delete unused objects
# del transformer
# del transformer_neuron


# # --- Compile CLIP text encoder and save [PASS] ---

# # Only keep the model being compiled in RAM to minimze memory pressure
# vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
# pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE)
# text_encoder = copy.deepcopy(pipe.text_encoder)
# del pipe

# # Apply the wrapper to deal with custom return type
# text_encoder = NeuronTextEncoder(text_encoder)

# # Compile text encoder
# # This is used for indexing a lookup table in torch.nn.Embedding,
# # so using random numbers may give errors (out of range).
# emb = torch.tensor([[49406, 18376,   525,  7496, 49407,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0]])
# text_encoder_neuron = torch_neuronx.trace(
#         text_encoder.neuron_text_encoder, 
#         emb, 
#         compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
#         compiler_args=compiler_flags
#         )

# # Enable asynchronous loading to speed up model load
# torch_neuronx.async_load(text_encoder_neuron)

# # Save the compiled text encoder
# text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
# torch.jit.save(text_encoder_neuron, text_encoder_filename)

# # delete unused objects
# del text_encoder
# del text_encoder_neuron


# # --- Compile VAE decoder and save [PASS] ---

# class f32Wrapper(nn.Module):
#     def __init__(self, original):
#         super().__init__()
#         self.original = original
#     def forward(self, x):
#         t = x.dtype
#         y = x.to(torch.float32)
#         output = self.original(y)
#         return output.type(t)
    
# def upcast_norms_to_f32(decoder: Decoder):
#     for upblock in decoder.up_blocks:
#         for resnet in upblock.resnets:
#             orig_resnet_norm1 = resnet.norm1
#             orig_resnet_norm2 = resnet.norm2
#             resnet.norm1 = f32Wrapper(orig_resnet_norm1)
#             resnet.norm2 = f32Wrapper(orig_resnet_norm2)
#     # for attn in decoder.mid_block.attentions:
#     #     orig_group_norm = attn.group_norm
#     #     attn.group_norm = f32Wrapper(orig_group_norm)
#     for resnet in decoder.mid_block.resnets:
#         orig_resnet_norm1 = resnet.norm1
#         orig_resnet_norm2 = resnet.norm2
#         resnet.norm1 = f32Wrapper(orig_resnet_norm1)
#         resnet.norm2 = f32Wrapper(orig_resnet_norm2)
#     orig_norm_out = decoder.norm_out
#     decoder.norm_out = f32Wrapper(orig_norm_out)

# # Only keep the model being compiled in RAM to minimze memory pressure
# vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
# decoder = copy.deepcopy(vae.decoder)
# decoder.eval()
# del vae

# upcast_norms_to_f32(decoder)

# compiler_flags = """ --verbose=INFO --target=trn1 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn1/inf2
# os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

# # Compile vae decoder
# decoder_in = torch.randn([batch_size, 16, frames, height, width], dtype=torch.float32)
# decoder_neuron = torch_neuronx.trace(
#     decoder, 
#     decoder_in, 
#     compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
#     compiler_args=compiler_flags,
#     inline_weights_to_neff=False
# )

# # Enable asynchronous loading to speed up model load
# torch_neuronx.async_load(decoder_neuron)

# # Save the compiled vae decoder
# decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
# torch.jit.save(decoder_neuron, decoder_filename)

# # delete unused objects
# del decoder
# del decoder_neuron


# # --- Compile VAE post_quant_conv and save [PASS] ---

# # Only keep the model being compiled in RAM to minimze memory pressure
# vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
# pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE)
# post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
# del pipe

# # # Compile vae post_quant_conv
# post_quant_conv_in = torch.randn([batch_size, 16, frames, height, width], dtype=torch.float32)
# post_quant_conv_neuron = torch_neuronx.trace(
#     post_quant_conv, 
#     post_quant_conv_in,
#     compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
# )

# # Enable asynchronous loading to speed up model load
# torch_neuronx.async_load(post_quant_conv_neuron)

# # # Save the compiled vae post_quant_conv
# post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
# torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

# # delete unused objects
# del post_quant_conv
# del post_quant_conv_neuron


# --- Load all compiled models and run pipeline ---
COMPILER_WORKDIR_ROOT = 'wan2.1_t2v_compile_dir'
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
transformer_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'transformer/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Load the compiled Transformer onto two neuron cores.
pipe.transformer = NeuronTransformer(TransformerWrap(pipe.transformer))
device_ids = [0,1]
pipe.transformer.transformer_wrap = torch_neuronx.DataParallel(torch.jit.load(transformer_filename), device_ids, set_dynamic_batching=False)

# Load other compiled models onto a single neuron core.
pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
pipe.vae.decoder = torch.jit.load(decoder_filename)
pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)

 # Run pipeline
prompt = ["A cat walks on the grass, realistic"]
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# First do a warmup run so all the asynchronous loads can finish.
# output_warmup = pipe(prompt[0], negative_prompt=negative_prompt, height=480, width=832, num_frames=81, guidance_scale=5.0).frames[0]
output_warmup = pipe(prompt[0], negative_prompt=negative_prompt, height=256, width=256, num_frames=81, guidance_scale=5.0).frames[0]

# total_time = 0
# for x in prompt:
#     start_time = time.time()
#     video = pipe(x, negative_prompt=negative_prompt, height=480, width=832, num_frames=81, guidance_scale=5.0).frames[0]  # 或其他适当的属性
#     total_time = total_time + (time.time()-start_time)
#     export_to_video(video, f"wan_video_{x}.mp4", fps=15)
# print("Average time: ", np.round((total_time/len(prompt)), 2), "seconds")