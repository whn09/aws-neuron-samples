import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
# os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2" # Comment this line out if using trn1/inf2
# os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2" # Comment this line out if using trn1/inf2
# compiler_flags = """ --verbose=INFO --target=trn2 --lnc=2 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn2
compiler_flags = """ --verbose=INFO --target=trn1 --model-type=transformer --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn1/inf2
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import AutoencoderKLWan
import torch
import argparse
import torch_neuronx
from diffusers.models.autoencoders.vae import Decoder
from neuron_commons import attention_wrapper, f32Wrapper

torch.nn.functional.scaled_dot_product_attention =  attention_wrapper

def upcast_norms_to_f32(decoder: Decoder):
    for upblock in decoder.up_blocks:
        for resnet in upblock.resnets:
            orig_resnet_norm1 = resnet.norm1
            orig_resnet_norm2 = resnet.norm2
            resnet.norm1 = f32Wrapper(orig_resnet_norm1)
            resnet.norm2 = f32Wrapper(orig_resnet_norm2)
    # for attn in decoder.mid_block.attentions:
    #     orig_group_norm = attn.group_norm
    #     attn.group_norm = f32Wrapper(orig_group_norm)
    for resnet in decoder.mid_block.resnets:
        orig_resnet_norm1 = resnet.norm1
        orig_resnet_norm2 = resnet.norm2
        resnet.norm1 = f32Wrapper(orig_resnet_norm1)
        resnet.norm2 = f32Wrapper(orig_resnet_norm2)
    orig_norm_out = decoder.norm_out
    decoder.norm_out = f32Wrapper(orig_norm_out)

def compile_decoder(args):
    latent_height = args.height//8
    latent_width = args.width//8
    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    
    batch_size = 1
    frames = 4  # default: 16
    # height, width = 32,32  # default: 96, 96
    in_channels = 16
    
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="wan2.1_t2v_hf_cache_dir")
    
    decoder: Decoder = vae.decoder
    decoder.eval()
    upcast_norms_to_f32(decoder)
    
    with torch.no_grad():
        sample_inputs = torch.rand((batch_size, in_channels, frames, latent_height, latent_width), dtype=torch.float32)
        
        compiled_decoder = torch_neuronx.trace(
            decoder,
            sample_inputs,
            compiler_workdir=f"{compiler_workdir}/decoder",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False)
        
        compiled_model_dir = f"{compiled_models_dir}/decoder"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)
        torch.jit.save(compiled_decoder, f"{compiled_model_dir}/model.pt")

        compiled_post_quant_conv = torch_neuronx.trace(
            vae.post_quant_conv,
            sample_inputs,
            compiler_workdir=f"{compiler_workdir}/post_quant_conv",
            compiler_args=compiler_flags,
            inline_weights_to_neff=False)
        
        compiled_model_dir = f"{compiled_models_dir}/post_quant_conv"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)     
        torch.jit.save(compiled_post_quant_conv, f"{compiled_model_dir}/model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", help="height of generated video.", type=int, default=256)
    parser.add_argument("--width", help="height of generated video.", type=int, default=256)
    parser.add_argument("--compiler_workdir", help="dir for compiler artifacts.", type=str, default="compiler_workdir")
    parser.add_argument("--compiled_models_dir", help="dir for compiled artifacts.", type=str, default="compiled_models")
    args = parser.parse_args()
    compile_decoder(args)