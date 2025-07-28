# source /opt/aws_neuronx_venv_pytorch_2_7/bin/activate
# pip install diffusers transformers sentencepiece matplotlib

import os

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"

import copy
import diffusers
import math
import numpy as npy
import time
import torch
import torch_neuronx
import torch.nn as nn
import torch.nn.functional as F

from diffusers import PixArtAlphaPipeline
# from IPython.display import clear_output
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from torch import nn

# Define datatype
DTYPE = torch.bfloat16

# clear_output(wait=False)

import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5EncoderModel
from diffusers import Transformer2DModel
from diffusers.models.autoencoders.vae import Decoder

import math

def neuron_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
  orig_shape = None
  if len(query.shape) == 4:
    orig_shape = query.shape
    def to3d(x):
      return x.reshape(-1, x.shape[2], x.shape[3])
    query, key, value = map(to3d, [query, key, value])
  if query.size() == key.size():
    attention_scores = torch.bmm(key, query.transpose(-1, -2)) * (
      1 / math.sqrt(query.size(-1))
    )
    attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
  else:
    attention_scores = torch.bmm(query, key.transpose(-1, -2)) * (
      1 / math.sqrt(query.size(-1))
    )
    attention_probs = attention_scores.softmax(dim=-1)
  attn_out = torch.bmm(attention_probs, value)
  if orig_shape:
    attn_out = attn_out.reshape(
      orig_shape[0], orig_shape[1], attn_out.shape[1], attn_out.shape[2]
    )
  return attn_out

class TracingT5Wrapper(nn.Module):
  def __init__(self, t: T5EncoderModel, seqlen: int):
    super().__init__()
    self.t = t
    self.device = t.device
    precomputed_bias = self.t.encoder.block[0].layer[0].SelfAttention.compute_bias(seqlen, seqlen)
    self.t.encoder.block[0].layer[0].SelfAttention.compute_bias = lambda *args, **kwargs: precomputed_bias
  def forward(self, text_input_ids, prompt_attention_mask):
    return self.t(
      text_input_ids, 
      attention_mask=prompt_attention_mask
    )

class InferenceTextEncoderWrapper(nn.Module):
  def __init__(self, dtype, t: T5EncoderModel, seqlen: int):
    super().__init__()
    self.dtype = dtype
    self.device = t.device
    self.t = t
  def forward(self, text_input_ids, attention_mask=None):
    return [self.t(text_input_ids, attention_mask)['last_hidden_state'].to(self.dtype)]

class TracingTransformerWrapper(nn.Module):
  def __init__(self, transformer):
    super().__init__()
    self.transformer = transformer
    self.config = transformer.config
    self.dtype = transformer.dtype
    self.device = transformer.device    
  def forward(self, hidden_states=None, encoder_hidden_states=None, timestep=None, encoder_attention_mask=None, **kwargs):
    return self.transformer(
      hidden_states=hidden_states, 
      encoder_hidden_states=encoder_hidden_states, 
      timestep=timestep, 
      encoder_attention_mask=encoder_attention_mask,
      added_cond_kwargs={"resolution": None, "aspect_ratio": None},
      return_dict=False)

class InferenceTransformerWrapper(nn.Module):
  def __init__(self, transformer: Transformer2DModel):
    super().__init__()
    self.transformer = transformer
    self.config = transformer.config
    self.dtype = transformer.dtype
    self.device = transformer.device
  def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, 
              encoder_attention_mask=None, added_cond_kwargs=None,
              return_dict=False):
    output = self.transformer(
      hidden_states, 
      encoder_hidden_states, 
      timestep, 
      encoder_attention_mask)
    return output

class SimpleWrapper(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model
  def forward(self, x):
    output = self.model(x)
    return output

class f32Wrapper(nn.Module):
  def __init__(self, original):
    super().__init__()
    self.original = original
  def forward(self, x):
    t = x.dtype
    y = x.to(torch.float32)
    output = self.original(y)
    return output.type(t)

sdpa_original = torch.nn.functional.scaled_dot_product_attention
def attention_wrapper(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
  if attn_mask is not None:
    return sdpa_original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
  else:
    return neuron_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

torch.nn.functional.scaled_dot_product_attention = attention_wrapper

def get_pipe(resolution, dtype):
  if resolution == 256:
    transformer: Transformer2DModel = Transformer2DModel.from_pretrained(
      "PixArt-alpha/PixArt-XL-2-256x256", 
      subfolder="transformer", 
      torch_dtype=dtype)
    return PixArtAlphaPipeline.from_pretrained(
      "PixArt-alpha/PixArt-XL-2-512x512", 
      transformer=transformer, 
      torch_dtype=dtype)
  elif resolution == 512:
    return PixArtAlphaPipeline.from_pretrained(
      "PixArt-alpha/PixArt-XL-2-512x512", 
      torch_dtype=dtype)
  else:
    raise Exception(f"Unsupport resolution {resolution} for pixart alpha")

# For saving compiler artifacts
COMPILER_WORKDIR_ROOT = 'pixart_alpha_compile_dir'

hidden_size = 4096
seqlen = 120

# Select the desired resolution.
resolution = 256
# resolution = 512

height = resolution
width = resolution

height_latent_size = height // 8
width_latent_size = width // 8

torch.manual_seed(1)
npy.random.seed(1)


# --- Compile T5 text encoder and save ---
pipe = get_pipe(resolution, DTYPE)
text_encoder = copy.deepcopy(pipe.text_encoder)
del pipe

for block in text_encoder.encoder.block:
  block.layer[1].DenseReluDense.act = torch.nn.GELU(approximate="tanh")

# Apply the wrapper to deal with custom return type
text_encoder = TracingT5Wrapper(text_encoder, seqlen)
sample_text_input_ids = torch.randint(low=0, high=18141, size=(1, seqlen))
sample_prompt_attention_mask = torch.randint(low=0, high=2, size=(1, seqlen))
sample_inputs = sample_text_input_ids, sample_prompt_attention_mask

text_encoder_neuron = torch_neuronx.trace(
  text_encoder,
  sample_inputs,
  compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
  compiler_args=["--enable-fast-loading-neuron-binaries"])

torch_neuronx.async_load(text_encoder_neuron)
text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
torch.jit.save(text_encoder_neuron, text_encoder_filename)

del text_encoder
del text_encoder_neuron

# --- Compile the transformer backbone and save ---
pipe = get_pipe(resolution, DTYPE)
transformer = copy.deepcopy(pipe.transformer)
del pipe

transformer = TracingTransformerWrapper(transformer)
sample_hidden_states = torch.rand([1, 4, height_latent_size, width_latent_size], dtype=DTYPE)
sample_encoder_hidden_states = torch.rand([1, seqlen, hidden_size], dtype=DTYPE)
sample_timestep = torch.ones((1,), dtype=torch.int64)
sample_encoder_attention_mask = torch.randint(low=0, high=2, size=(1, seqlen), dtype=torch.int64)
sample_inputs = sample_hidden_states, sample_encoder_hidden_states, sample_timestep, sample_encoder_attention_mask
transformer_neuron = torch_neuronx.trace(
  transformer,
  sample_inputs,
  compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'transformer'),
  compiler_args=["--model-type=transformer", "--enable-fast-loading-neuron-binaries"]
)

# Enable asynchronous and lazy loading to speed up model load
torch_neuronx.async_load(transformer_neuron)
torch_neuronx.lazy_load(transformer_neuron)

# save compiled transformer
transformer_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'transformer/model.pt')
torch.jit.save(transformer_neuron, transformer_filename)

del transformer
del transformer_neuron

# --- Compile the decoder and save ---
pipe = get_pipe(resolution, DTYPE)
decoder = copy.deepcopy(pipe.vae.decoder)
post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
del pipe

for upblock in decoder.up_blocks:
  for resnet in upblock.resnets:
    orig_resnet_norm1 = resnet.norm1
    orig_resnet_norm2 = resnet.norm2
    resnet.norm1 = f32Wrapper(orig_resnet_norm1)
    resnet.norm2 = f32Wrapper(orig_resnet_norm2)

for resnet in decoder.mid_block.resnets:
  orig_resnet_norm1 = resnet.norm1
  orig_resnet_norm2 = resnet.norm2
  resnet.norm1 = f32Wrapper(orig_resnet_norm1)
  resnet.norm2 = f32Wrapper(orig_resnet_norm2)

sample_inputs = torch.rand([1, 4, height_latent_size, width_latent_size], dtype=DTYPE)
decoder_neuron = torch_neuronx.trace(
  decoder, 
  sample_inputs, 
  compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
  compiler_args=["--model-type=unet-inference", "--enable-fast-loading-neuron-binaries"]
)

# Enable asynchronous loading to speed up model load
torch_neuronx.async_load(decoder_neuron)

# Save the compiled vae decoder
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
torch.jit.save(decoder_neuron, decoder_filename)

# delete unused objects
del decoder
del decoder_neuron

# --- Compile VAE post_quant_conv and save ---
post_quant_conv_neuron = torch_neuronx.trace(
  post_quant_conv, 
  sample_inputs,
  compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
  compiler_args=["--enable-fast-loading-neuron-binaries"]
)

# Enable asynchronous loading to speed up model load
torch_neuronx.async_load(post_quant_conv_neuron)

# Save the compiled vae post_quant_conv
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

# delete unused objects
del post_quant_conv
del post_quant_conv_neuron


# --- Load all compiled models ---
COMPILER_WORKDIR_ROOT = 'pixart_alpha_compile_dir'
text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
transformer_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'transformer/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

pipe = get_pipe(resolution, DTYPE)

_neuronTextEncoder = InferenceTextEncoderWrapper(DTYPE, pipe.text_encoder, seqlen)
_neuronTextEncoder.t = torch.jit.load(text_encoder_filename)
pipe.text_encoder = _neuronTextEncoder
assert pipe._execution_device is not None

device_ids = [0, 1]
_neuronTransformer = InferenceTransformerWrapper(pipe.transformer)
_neuronTransformer.transformer = torch_neuronx.DataParallel(torch.jit.load(transformer_filename), device_ids, set_dynamic_batching=False)
pipe.transformer = _neuronTransformer

pipe.vae.decoder = SimpleWrapper(torch.jit.load(decoder_filename))
pipe.vae.post_quant_conv = SimpleWrapper(torch.jit.load(post_quant_conv_filename))


# Run pipeline
prompt = [
  "a photo of an astronaut riding a horse on mars",
  "sonic on the moon",
  "elvis playing guitar while eating a hotdog",
  "saved by the bell",
  "engineers eating lunch at the opera",
  "panda eating bamboo on a plane",
  "A digital illustration of a steampunk flying machine in the sky with cogs and mechanisms, 4k, detailed, trending in artstation, fantasy vivid colors",
  "kids playing soccer at the FIFA World Cup"
]

# First do a warmup run so all the asynchronous loads can finish
image_warmup = pipe(prompt[0]).images[0]

total_time = 0
for x in prompt:
  start_time = time.time()
  image = pipe(x).images[0]
  total_time = total_time + (time.time()-start_time)
  image.save(f"image_{x}.png")
print("Average time: ", npy.round((total_time/len(prompt)), 2), "seconds")