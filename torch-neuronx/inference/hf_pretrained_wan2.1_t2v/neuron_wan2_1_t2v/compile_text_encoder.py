import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2" # Comment this line out if using trn1/inf2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2" # Comment this line out if using trn1/inf2
compiler_flags = """ --verbose=INFO --target=trn2 --lnc=2 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn2
# compiler_flags = """ --verbose=INFO --target=trn1 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn1/inf2
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import copy

from diffusers import AutoencoderKLWan, WanPipeline
import torch
import argparse
import torch_neuronx
import neuronx_distributed
from transformers.models.umt5 import UMT5EncoderModel
from torch import nn
from functools import partial

from transformers.models.umt5.modeling_umt5 import UMT5Block, UMT5LayerSelfAttention, UMT5LayerFF

from neuron_commons import attention_wrapper, f32Wrapper
from neuron_parallel_utils import get_sharded_data, shard_umt5_self_attention, shard_umt5_ff

torch.nn.functional.scaled_dot_product_attention = attention_wrapper


# class TracingUMT5WrapperTP(nn.Module):
#     def __init__(self, t: UMT5EncoderModel, seqlen: int):
#         super().__init__()
#         self.t = t
#         self.device = t.device
#         precomputed_bias = self.t.encoder.block[0].layer[0].SelfAttention.compute_bias(seqlen, seqlen)
#         precomputed_bias_tp = get_sharded_data(precomputed_bias, 1)
#         self.t.encoder.block[0].layer[0].SelfAttention.compute_bias = lambda *args, **kwargs: precomputed_bias_tp
    
#     def forward(self, text_input_ids, prompt_attention_mask=None):
#         return self.t(
#             text_input_ids, 
#             attention_mask=prompt_attention_mask
#         )
        
class TracingUMT5WrapperTP(nn.Module):
    def __init__(self, t: UMT5EncoderModel, seqlen: int, tp_degree: int):
        super().__init__()
        self.t = t
        self.device = t.device
        self.tp_degree = tp_degree
        
        # 为每个 block 预计算并分片 position bias
        for block_idx, block in enumerate(self.t.encoder.block):
            original_compute_bias = block.layer[0].SelfAttention.compute_bias
            precomputed_bias = original_compute_bias(seqlen, seqlen)
            
            # 根据注意力头的分片方式来分片 position_bias
            # position_bias shape: [1, num_heads, seq_len, seq_len]
            if tp_degree > 1:
                # 沿着 num_heads 维度分片
                num_heads = precomputed_bias.shape[1]
                heads_per_rank = num_heads // tp_degree
                rank = neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_rank()
                start_idx = rank * heads_per_rank
                end_idx = start_idx + heads_per_rank
                precomputed_bias_tp = precomputed_bias[:, start_idx:end_idx, :, :]
            else:
                precomputed_bias_tp = precomputed_bias
            
            # 替换 compute_bias 函数
            block.layer[0].SelfAttention.compute_bias = lambda *args, **kwargs: precomputed_bias_tp
    
    def forward(self, text_input_ids, prompt_attention_mask=None):
        return self.t(
            text_input_ids, 
            attention_mask=prompt_attention_mask
        )

def get_text_encoder(tp_degree: int, sequence_length: int):
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    DTYPE = torch.bfloat16
    text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=DTYPE, cache_dir="wan2.1_t2v_hf_cache_dir")
    text_encoder.eval()
    
    for idx, block in enumerate(text_encoder.encoder.block):
        block: UMT5Block = block
        block.layer[1].DenseReluDense.act = torch.nn.GELU(approximate="tanh")
        selfAttention: UMT5LayerSelfAttention = block.layer[0].SelfAttention
        ff: UMT5LayerFF = block.layer[1]
        layer_norm_0 = block.layer[0].layer_norm.to(torch.float32)
        layer_norm_1 = block.layer[1].layer_norm.to(torch.float32)       
        block.layer[1] = shard_umt5_ff(ff)
        block.layer[0].SelfAttention = shard_umt5_self_attention(tp_degree, selfAttention)
        block.layer[0].layer_norm = f32Wrapper(layer_norm_0)
        block.layer[1].layer_norm = f32Wrapper(layer_norm_1)
    
    final_layer_norm = text_encoder.encoder.final_layer_norm.to(torch.float32)
    text_encoder.encoder.final_layer_norm = f32Wrapper(final_layer_norm)
    
    # 传递 tp_degree 参数
    return TracingUMT5WrapperTP(text_encoder, sequence_length, tp_degree), {}

def compile_text_encoder(args):
    batch_size = 1 # batch_size = args.num_prompts
    sequence_length = args.max_sequence_length
    tp_degree = 4 # Use tensor parallel degree as 4 for trn2
    os.environ["LOCAL_WORLD_SIZE"] = "4"
    # tp_degree = 8 # Use tensor parallel degree as 8 for trn1/inf2, default: 8
    # os.environ["LOCAL_WORLD_SIZE"] = "8"
    get_text_encoder_f = partial(get_text_encoder, tp_degree, sequence_length)
    
    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    
    with torch.no_grad():
        sample_inputs = torch.ones((batch_size, sequence_length), dtype=torch.int64), \
            torch.ones((batch_size, sequence_length), dtype=torch.int64)
        # sample_inputs = torch.tensor([[49406, 18376, 525, 7496, 49407] + [0] * (sequence_length - 5)], dtype=torch.int64)
        
        compiled_text_encoder = neuronx_distributed.trace.parallel_model_trace(
            get_text_encoder_f,
            sample_inputs,
            compiler_workdir=f"{compiler_workdir}/text_encoder",
            compiler_args=compiler_flags,
            tp_degree=tp_degree,
        )
        compiled_model_dir = f"{compiled_models_dir}/text_encoder"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)           
        neuronx_distributed.trace.parallel_model_save(
            compiled_text_encoder, f"{compiled_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", help="number of prompts", type=int, default=1)
    parser.add_argument("--max_sequence_length", help="max sequence length.", type=int, default=300)
    parser.add_argument("--compiler_workdir", help="dir for compiler artifacts.", type=str, default="compiler_workdir")
    parser.add_argument("--compiled_models_dir", help="dir for compiled artifacts.", type=str,  default="compiled_models")
    args = parser.parse_args()
    compile_text_encoder(args)


# class NeuronTextEncoder(nn.Module):
#     def __init__(self, text_encoder):
#         super().__init__()
#         self.neuron_text_encoder = text_encoder
#         self.config = text_encoder.config
#         self.dtype = text_encoder.dtype
#         self.device = text_encoder.device

#     def forward(self, emb, attention_mask = None):
#         return [self.neuron_text_encoder(emb)['last_hidden_state']]

# DTYPE = torch.bfloat16
# model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
# COMPILER_WORKDIR_ROOT = "compile_workdir_latency_optimized"

# # --- Compile CLIP text encoder and save [PASS] ---

# text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=DTYPE, cache_dir="wan2.1_t2v_hf_cache_dir")

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