"""8-GPU A10G benchmark for Wan2.2-TI2V-5B on g5.48xlarge.

Uses Wan2.2 official codebase with FSDP + Ulysses Sequence Parallel.
Requires: pip install Wan2.2 (https://github.com/Wan-AI/Wan2.2)

Usage:
  # 8-GPU FSDP + Sequence Parallel (recommended)
  torchrun --nproc_per_node=8 bench_a10g_8gpu.py \
      --ckpt_dir /path/to/Wan2.2-TI2V-5B

  # With model offloading (frees transformer before VAE decode)
  torchrun --nproc_per_node=8 bench_a10g_8gpu.py \
      --ckpt_dir /path/to/Wan2.2-TI2V-5B --offload_model

Results on g5.48xlarge (8x A10G, PCIe):
  832x480 81f:  ~965s  (~19.3s/step, 50 steps)
  832x480 121f: ~1024s (~20.5s/step, 50 steps)
  1280x704:     OOM during VAE decode (VAE is not distributed)

Note: 8x A10G FSDP+SP is ~9.3x SLOWER per step than single A10G with
diffusers (~2.08s/step). PCIe interconnect (no NVLink) makes FSDP
all-gather communication dominate. Multi-GPU A10G is impractical for
video generation.
"""
import argparse
import gc
import os
import time

import torch
import torch.distributed as dist

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS

PROMPT = "A cat walks on the grass, realistic"
SEED = 42

CONFIGS = [
    # (size_key, frame_num, label)
    ("480*832", 81, "832x480 81f"),
    ("480*832", 121, "832x480 121f"),
    ("704*1280", 81, "1280x704 81f"),
    ("704*1280", 121, "1280x704 121f"),
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to Wan2.2-TI2V-5B checkpoint directory")
    parser.add_argument("--offload_model", action="store_true", default=False,
                        help="Offload transformer to CPU before VAE decode")
    parser.add_argument("--sample_steps", type=int, default=50)
    return parser.parse_args()


def main():
    args = get_args()
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)

    cfg = WAN_CONFIGS["ti2v-5B"]
    if rank == 0:
        print("=" * 70, flush=True)
        print("Wan2.2-TI2V-5B 8-GPU Benchmark (FSDP + Ulysses SP)", flush=True)
        print("GPUs: {} | Offload: {}".format(world_size, args.offload_model),
              flush=True)
        print("=" * 70, flush=True)
        print("\nLoading model...", flush=True)

    t_load_start = time.time()
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=True,
        dit_fsdp=True,
        use_sp=True,
        t5_cpu=False,
    )
    t_load = time.time() - t_load_start

    if rank == 0:
        mem_gb = torch.cuda.memory_allocated(local_rank) / 1024**3
        print("Model loaded in {:.1f}s, GPU {} memory: {:.1f} GB".format(
            t_load, local_rank, mem_gb), flush=True)

    results = []

    for size_key, frame_num, label in CONFIGS:
        size = SIZE_CONFIGS[size_key]
        if rank == 0:
            print("\n" + "=" * 50, flush=True)
            print("Testing: {}".format(label), flush=True)
            print("=" * 50, flush=True)

        # Warmup
        if rank == 0:
            print("Warmup...", flush=True)
        try:
            torch.cuda.empty_cache()
            gc.collect()
            dist.barrier()
            t0 = time.time()
            _ = wan_ti2v.generate(
                PROMPT, size=size, frame_num=frame_num,
                sampling_steps=args.sample_steps, seed=SEED + 1000,
                offload_model=args.offload_model,
            )
            torch.cuda.synchronize()
            warmup_t = time.time() - t0
            if rank == 0:
                print("Warmup: {:.2f}s".format(warmup_t), flush=True)
        except Exception as e:
            if rank == 0:
                print("Warmup failed: {}".format(e), flush=True)
            results.append((label, "FAIL", 0, 0))
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # Timed run
        if rank == 0:
            print("Inference...", flush=True)
        try:
            torch.cuda.empty_cache()
            gc.collect()
            dist.barrier()
            t0 = time.time()
            video = wan_ti2v.generate(
                PROMPT, size=size, frame_num=frame_num,
                sampling_steps=args.sample_steps, seed=SEED,
                offload_model=args.offload_model,
            )
            torch.cuda.synchronize()
            infer_t = time.time() - t0
            if rank == 0:
                print("Inference: {:.2f}s".format(infer_t), flush=True)
                try:
                    from wan.utils.utils import cache_video
                    fname = "output_8gpu_{}.mp4".format(
                        label.replace(" ", "_"))
                    cache_video(
                        tensor=video[None], save_file=fname,
                        fps=cfg.sample_fps, nrow=1, normalize=True,
                        value_range=(-1, 1))
                    print("Saved: " + fname, flush=True)
                except Exception as save_err:
                    print("Save failed: {}".format(save_err), flush=True)
            results.append((label, "ok", warmup_t, infer_t))
        except Exception as e:
            if rank == 0:
                print("Inference failed: {}".format(e), flush=True)
            results.append((label, "FAIL", warmup_t, 0))
            torch.cuda.empty_cache()
            gc.collect()

    # Print summary
    if rank == 0:
        print("\n\n" + "=" * 70, flush=True)
        print("RESULTS: {}-GPU FSDP+SP | g5.48xlarge ({}x A10G)".format(
            world_size, world_size), flush=True)
        print("Steps: {}, Seed: {}".format(args.sample_steps, SEED),
              flush=True)
        print("=" * 70, flush=True)
        print("{:<25} {:<8} {:<12} {:<12}".format(
            "Config", "Status", "Warmup(s)", "Inference(s)"), flush=True)
        print("-" * 60, flush=True)
        for label, status, warmup, infer in results:
            if status != "ok":
                print("{:<25} {:<8} {:<12} {:<12}".format(
                    label, status, "-", "-"), flush=True)
            else:
                print("{:<25} {:<8} {:<12.2f} {:<12.2f}".format(
                    label, status, warmup, infer), flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
