from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from transformers.models.umt5 import UMT5EncoderModel
from torch import nn
from types import SimpleNamespace

class InferenceTextEncoderWrapper(nn.Module):
    def __init__(self, dtype, t: UMT5EncoderModel, seqlen: int):
        super().__init__()
        self.dtype = dtype
        self.device = t.device
        self.t = t
    def forward(self, text_input_ids, attention_mask=None):
        # print('self.dtype:', self.dtype)
        # print('self.device:', self.device)
        # print('self.t:', self.t)
        # print('text_input_ids:', text_input_ids)
        # print('attention_mask:', attention_mask)
        result = self.t(text_input_ids, attention_mask)  # , attention_mask
        # print('result:', type(result), result)
        # return [result['last_hidden_state'].to(self.dtype)]
        return SimpleNamespace(last_hidden_state=result['last_hidden_state'].to(self.dtype))

class InferenceTransformerWrapper(nn.Module):
    def __init__(self, transformer: WanTransformer3DModel):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device
    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None, return_dict=False, **kwargs):  # encoder_attention_mask=None, added_cond_kwargs=None,
        # print('self.config:', self.config)
        # print('self.dtype:', self.dtype)
        # print('self.device:', self.device)
        # print('self.transformer:', self.transformer)
        # print('hidden_states:', hidden_states.shape, hidden_states)
        # print('timestep:', timestep)
        # print('encoder_hidden_states:', encoder_hidden_states.shape, encoder_hidden_states)
        # print('kwargs:', kwargs)
        output = self.transformer(
            hidden_states, 
            timestep,
            encoder_hidden_states, 
            # encoder_attention_mask
        )
        # print('output:', output.shape, output)
        return output

class SimpleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, **kwargs):
        print('self.model:', self.model)
        print('x:', x.shape, x)
        print('kwargs:', kwargs)
        output = self.model(x)
        print('output:', output.shape, output)
        return output

import torch
import math
from torch import nn

# from neuronxcc.starfish.penguin.targets.nki.private_api import vnc
from torch_neuronx.xla_impl.ops import nki_jit
from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
_flash_fwd_call = nki_jit()(attention_isa_kernel)


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


def attention_wrapper_sharded_without_swap(query, key, value):
    bs, n_head, q_len, d_head = query.shape
    q = query.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, q_len))
    v = value.clone().reshape((bs*n_head, q_len, d_head))
    attn_output = torch.zeros((bs*n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)
    use_sharded_attention_kernel = True # Use "need use_sharded_attention_kernel = True" in case of trn2
    # use_sharded_attention_kernel = False # We do not "need use_sharded_attention_kernel" in case of trn1/inf2, so we could make it false
    if use_sharded_attention_kernel:
        # grid = (vnc(2),)
        grid = (2,)
        _flash_fwd_call[grid](q, k, v, 0.117, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    else:
        _flash_fwd_call(q, k, v, 0.117, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    attn_output = attn_output.reshape((bs, n_head, q_len, d_head))
    return attn_output


sdpa_original = torch.nn.functional.scaled_dot_product_attention
def attention_wrapper(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
    if attn_mask is not None:
        return sdpa_original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    else:
        return neuron_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        
def attention_wrapper_for_transformer(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
    if attn_mask is not None:
        return sdpa_original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    else:
        return attention_wrapper_sharded_without_swap(query, key, value)
        
class f32Wrapper(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
    def forward(self, x):
        t = x.dtype
        y = x.to(torch.float32)
        output = self.original(y)
        return output.type(t)
    
    