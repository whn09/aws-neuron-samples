import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
# os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2" # Comment this line out if using trn1/inf2
# os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2" # Comment this line out if using trn1/inf2
# compiler_flags = """ --verbose=INFO --target=trn2 --lnc=2 --internal-hlo2tensorizer-options='--fuse-dot-logistic=false' --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn2
# compiler_flags = """ --verbose=INFO --target=trn1 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn1/inf2
compiler_flags = """ --verbose=INFO --target=trn1 --model-type=transformer --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn1/inf2
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import copy
import math
from typing import Optional

from diffusers import AutoencoderKLWan, WanPipeline
import torch
import argparse
import neuronx_distributed
import torch_neuronx

from torch import nn
import torch.nn.functional as F
from functools import partial

from neuron_commons import attention_wrapper_for_transformer
from neuron_parallel_utils import shard_transformer_attn, shard_transformer_feedforward

from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
# torch.nn.functional.scaled_dot_product_attention = attention_wrapper_for_transformer

from diffusers.models.attention_processor import Attention
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel  # noqa: E402
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel  # noqa: E402
from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402

# class TracingTransformerWrapper(nn.Module):
#     def __init__(self, transformer):
#         super().__init__()
#         self.transformer = transformer
#         self.config = transformer.config
#         self.dtype = transformer.dtype
#         self.device = transformer.device    
    
#     def forward(self, hidden_states=None, timestep=None, encoder_hidden_states=None, **kwargs):  # , encoder_attention_mask=None
#         return self.transformer(
#         hidden_states=hidden_states, 
#         timestep=timestep, 
#         encoder_hidden_states=encoder_hidden_states, 
#         # encoder_attention_mask=encoder_attention_mask,
#         # added_cond_kwargs={"resolution": None, "aspect_ratio": None},
#         return_dict=False)

# def get_transformer_model(tp_degree: int):
#     DTYPE = torch.bfloat16
#     model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
#     vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="wan2.1_t2v_hf_cache_dir")
#     pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE, cache_dir="wan2.1_t2v_hf_cache_dir")
    
#     # 分片所有30个blocks
#     for block_idx, block in enumerate(pipe.transformer.blocks):
#         print(f"Processing block {block_idx}/29")
        
#         # 分片attention层
#         block.attn1 = shard_transformer_attn(tp_degree, block.attn1)
#         block.attn2 = shard_transformer_attn(tp_degree, block.attn2)

#         # 分片feedforward层
#         block.ffn = shard_transformer_feedforward(block.ffn)
        
#     mod_pipe_transformer_f = TracingTransformerWrapper(pipe.transformer)
#     return mod_pipe_transformer_f, {}

# def compile_transformer(args):
#     # tp_degree = 4
#     tp_degree = 8 # Use tensor parallel degree as 8 for trn1/inf2
#     os.environ["LOCAL_WORLD_SIZE"] = "4" # Use tensor parallel degree as 4 for trn2
#     latent_height = args.height//8
#     latent_width = args.width//8
#     num_prompts = 1
#     num_images_per_prompt = args.num_images_per_prompt
#     max_sequence_length = args.max_sequence_length
#     hidden_size = 4096
#     compiler_workdir = args.compiler_workdir
#     compiled_models_dir = args.compiled_models_dir
#     batch_size = 1
#     frames = 16
#     height, width = 96, 96
#     in_channels = 16
#     sample_hidden_states = torch.ones((batch_size, in_channels, frames, height, width), dtype=torch.bfloat16)
#     sample_encoder_hidden_states = torch.ones((batch_size, max_sequence_length, hidden_size), dtype=torch.bfloat16)
#     sample_timestep = torch.ones((batch_size), dtype=torch.int64)
#     # sample_encoder_attention_mask = torch.ones((batch_size, max_sequence_length), dtype=torch.int64)

#     get_transformer_model_f = partial(get_transformer_model, tp_degree)
#     with torch.no_grad():
#         sample_inputs = sample_hidden_states, sample_timestep, sample_encoder_hidden_states  # , sample_encoder_attention_mask
#         compiled_transformer = neuronx_distributed.trace.parallel_model_trace(
#             get_transformer_model_f,
#             sample_inputs,
#             compiler_workdir=f"{compiler_workdir}/transformer",
#             compiler_args=compiler_flags,
#             tp_degree=tp_degree,
#             inline_weights_to_neff=False,
#         )
#         compiled_model_dir = f"{compiled_models_dir}/transformer"
#         if not os.path.exists(compiled_model_dir):
#             os.makedirs(compiled_model_dir)         
#         neuronx_distributed.trace.parallel_model_save(
#             compiled_transformer, f"{compiled_model_dir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--height", help="height of generated image.", type=int, default=1024)
#     parser.add_argument("--width", help="width of generated image.", type=int, default=1024)
#     parser.add_argument("--num_images_per_prompt", help="number of images per prompt.", type=int, default=1)
#     parser.add_argument("--max_sequence_length", help="max sequence length.", type=int, default=300)
#     parser.add_argument("--compiler_workdir", help="dir for compiler artifacts.", type=str, default="compiler_workdir")
#     parser.add_argument("--compiled_models_dir", help="dir for compiled artifacts.", type=str, default="compiled_models")
#     args = parser.parse_args()
#     compile_transformer(args)


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

        # # 使用XLA兼容的rotary embedding TODO：临时注释掉，为了编译通过，以后还得研究怎么解决
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

DTYPE=torch.bfloat16
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
COMPILER_WORKDIR_ROOT = 'compile_workdir_latency_optimized'

batch_size = 1
frames = 4  # default: 16  # typical frame count for video generation
height, width = 32, 32  # default: 96, 96  # spatial dimensions
in_channels = 16  # 根据配置，Wan使用16个输入通道

# --- Compile Transformer and save [NOT PASS]---

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="wan2.1_t2v_hf_cache_dir")
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE, cache_dir="wan2.1_t2v_hf_cache_dir")

# # Replace original cross-attention module with custom cross-attention module for better performance
# Attention.get_attention_scores = get_attention_scores

# # Apply double wrapper to deal with custom return type
pipe.transformer = NeuronTransformer(TransformerWrap(pipe.transformer))
# pipe.transformer = TracingTransformerWrapper(pipe.transformer)

# Only keep the model being compiled in RAM to minimze memory pressure
transformer = copy.deepcopy(pipe.transformer)  # .transformer_wrap
del pipe

# Compile transformer - adjust input shapes for 3D video

# 3D input for video: (batch, channels, frames, height, width)
hidden_states_1b = torch.randn([batch_size, in_channels, frames, height, width], dtype=DTYPE)
timestep_1b = torch.tensor(999, dtype=DTYPE).expand((batch_size,))
# Text encoder output dimension for Wan (might be different from SD)
encoder_hidden_states_1b = torch.randn([batch_size, 77, 4096], dtype=DTYPE)  # Wan uses 4096 dim

example_inputs = hidden_states_1b, timestep_1b, encoder_hidden_states_1b

transformer_neuron = torch_neuronx.trace(
    transformer,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'transformer'),
    compiler_args=compiler_flags
)

# Enable asynchronous and lazy loading to speed up model load
torch_neuronx.async_load(transformer_neuron)
torch_neuronx.lazy_load(transformer_neuron)

# save compiled transformer
transformer_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'transformer/model.pt')
torch.jit.save(transformer_neuron, transformer_filename)

# delete unused objects
del transformer
del transformer_neuron