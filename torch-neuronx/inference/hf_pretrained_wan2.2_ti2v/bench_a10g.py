"""A10G benchmark script for Wan2.2-TI2V-5B.

Strategy: text_encoder on CPU (runs once), transformer+VAE on GPU (no offload).
Transformer ~10GB (bf16) + VAE ~2.6GB (fp32) = ~12.6GB on GPU, leaves ~9GB for activations.
Text encoder embeddings are pre-computed on CPU and moved to CUDA before pipeline call.
"""
import torch
import time
import random
import gc
import numpy as np
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

CACHE = "/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
SEED = 42

PROMPT = "A cat walks on the grass, realistic"
NEG_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

CONFIGS = [
    (384, 512, 81, 16),
    (384, 512, 121, 24),
    (480, 640, 81, 16),
    (480, 640, 121, 24),
    (704, 1280, 81, 16),
    (704, 1280, 121, 24),
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    print("Loading model...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32, cache_dir=CACHE
    )
    pipe = WanPipeline.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=torch.bfloat16, cache_dir=CACHE
    )
    print("Model loaded to CPU.", flush=True)

    # Pre-compute text embeddings on CPU (text_encoder stays on CPU)
    print("Encoding prompts on CPU...", flush=True)
    t0 = time.time()
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        do_classifier_free_guidance=True,
        max_sequence_length=512,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    print("Prompt encoding done in {:.1f}s".format(time.time() - t0), flush=True)

    # Move embeddings to CUDA
    prompt_embeds = prompt_embeds.to("cuda")
    negative_prompt_embeds = negative_prompt_embeds.to("cuda")

    # Free text_encoder from memory entirely
    pipe.text_encoder = None
    pipe.tokenizer = None
    gc.collect()

    # Move transformer + VAE to GPU
    pipe.transformer = pipe.transformer.to("cuda")
    pipe.vae = pipe.vae.to("cuda")
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print("GPU memory after loading: {:.1f} GB".format(mem_gb), flush=True)

    results = []

    for h, w, nf, fps in CONFIGS:
        tag = "{}x{} {}fps {}f".format(w, h, fps, nf)
        print("\n" + "=" * 50, flush=True)
        print("Testing: " + tag, flush=True)
        print("=" * 50, flush=True)

        warmup_time = 0.0
        infer_time = 0.0

        common = dict(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=h,
            width=w,
            num_frames=nf,
            guidance_scale=5.0,
            num_inference_steps=50,
        )

        # Warmup
        try:
            torch.cuda.empty_cache()
            gen = torch.Generator(device="cuda").manual_seed(SEED + 1000)
            torch.cuda.synchronize()
            t0 = time.time()
            _ = pipe(**common, generator=gen).frames[0]
            torch.cuda.synchronize()
            warmup_time = time.time() - t0
            print("Warmup: {:.2f}s".format(warmup_time), flush=True)
        except Exception as e:
            err = str(e).lower()
            if "out of memory" in err or "outofmemory" in type(e).__name__.lower():
                print("OOM during warmup: " + tag, flush=True)
                results.append((tag, "OOM", 0, 0))
                torch.cuda.empty_cache()
                continue
            raise

        # Inference
        try:
            gen = torch.Generator(device="cuda").manual_seed(SEED)
            torch.cuda.synchronize()
            t0 = time.time()
            output = pipe(**common, generator=gen).frames[0]
            torch.cuda.synchronize()
            infer_time = time.time() - t0
            print("Inference: {:.2f}s".format(infer_time), flush=True)
            results.append((tag, "ok", warmup_time, infer_time))
            try:
                fname = "output_a10g_{}x{}_{}fps_{}f.mp4".format(w, h, fps, nf)
                export_to_video(output, fname, fps=fps)
                print("Saved: " + fname, flush=True)
            except Exception as save_err:
                print("Video save failed (non-fatal): " + str(save_err), flush=True)
        except Exception as e:
            err = str(e).lower()
            if "out of memory" in err or "outofmemory" in type(e).__name__.lower():
                print("OOM during inference: " + tag, flush=True)
                results.append((tag, "OOM", warmup_time, 0))
                torch.cuda.empty_cache()
            else:
                raise

    # Print results
    print("\n\n" + "=" * 70, flush=True)
    print("A10G Benchmark Results", flush=True)
    print("GPU: " + torch.cuda.get_device_name(0), flush=True)
    print("Steps: 50, Guidance: 5.0, Seed: 42", flush=True)
    print("=" * 70, flush=True)
    header = "{:<25} {:<8} {:<12} {:<12}".format("Config", "Status", "Warmup(s)", "Inference(s)")
    print(header)
    print("-" * 60)
    for tag, status, warmup, infer in results:
        if status == "OOM":
            print("{:<25} {:<8} {:<12} {:<12}".format(tag, status, "-", "-"))
        else:
            print("{:<25} {:<8} {:<12.2f} {:<12.2f}".format(tag, status, warmup, infer))

    # Write to file
    with open("test_results_a10g.txt", "w") as f:
        f.write("Wan2.2 TI2V A10G Benchmark Results\n")
        f.write("GPU: {}\n".format(torch.cuda.get_device_name(0)))
        f.write("Steps: 50, Guidance: 5.0, Seed: 42\n")
        f.write("=" * 50 + "\n")
        f.write("{:<12} {:<8} {:<8} {:<12} {:<12} {:<10}\n".format(
            "Resolution", "FPS", "Frames", "Warmup(s)", "Inference(s)", "Status"))
        f.write("-" * 60 + "\n")
        for tag, status, warmup, infer in results:
            parts = tag.split()
            res, fps_str, frames_str = parts[0], parts[1].replace("fps", ""), parts[2].replace("f", "")
            if status == "OOM":
                f.write("{:<12} {:<8} {:<8} {:<12} {:<12} {:<10}\n".format(
                    res, fps_str, frames_str, "-", "-", status))
            else:
                f.write("{:<12} {:<8} {:<8} {:<12.2f} {:<12.2f} {:<10}\n".format(
                    res, fps_str, frames_str, warmup, infer, status))
    print("\nResults saved to test_results_a10g.txt", flush=True)


if __name__ == "__main__":
    main()
