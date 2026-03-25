# Wan2.2 Text/Image-to-Video Inference on AWS Trainium2

This project implements [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) video generation on AWS Trainium2 (trn2.48xlarge) using the AWS Neuron SDK. Supports multiple resolutions from 512x384 up to 1280x704 (720P) with text-to-video and image-to-video generation.

## Multi-Resolution Performance (trn2.48xlarge vs H100 vs A10G)

| Resolution | FPS | Frames | Trn2 CP (s) | Trn2 CFG (s) | H100 (s) | A10G (s) | Decoder |
|-----------|-----|--------|-------------|--------------|-----------|----------|---------|
| 512x384 | 16 | 81 | 20.67 | **18.32** | 16.13 | 116.96 | stateful rolling |
| 512x384 | 24 | 121 | 30.07 | **26.44** | 24.48 | 180.45 | stateful rolling |
| 640x480 | 16 | 81 | **33.20** | 34.10 | 26.06 | 186.95 | stateful rolling |
| 640x480 | 24 | 121 | 49.29 | **45.15** | 39.67 | 290.18 | stateful rolling |
| 1280x704 | 16 | 81 | 163.99 | **155.06** | 87.66 | OOM | tiled |
| 1280x704 | 24 | 121 | 255.07 | **243.71** | 143.20 | OOM | tiled |

- **Trn2 CP**: Context Parallel (CP=2, sequence split across ranks, K/V all-gather in self-attention)
- **Trn2 CFG**: CFG Parallel (batch=2, cond+uncond in single forward pass, no K/V communication)
- **A10G**: g5.8xlarge (24GB VRAM). Text encoder runs on CPU (~432s per call, excluded from timing). Transformer (bf16) + VAE (fp32) on GPU, no model offloading. 720P exceeds 24GB VRAM.
- CFG Parallel is faster for most configs (up to -13%). At 640x480/81f the doubled attention compute outweighs the communication savings.
- Timing is pure inference (excludes model loading and warmup). See `test_results.txt`, `test_results_gpu.txt`, and `test_results_a10g.txt`.

## Quick Start

```bash
# Mount NVMe storage (trn2.48xlarge has ~7.6TB NVMe)
sudo ./setup_nvme.sh

# Activate Neuron virtual environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Install dependencies
pip install -r requirements.txt

# Compile all models (text encoder, transformer, decoder)
./compile.sh

# Text-to-Video (T2V)
python run_wan2.2_ti2v.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_wan2.2_ti2v_5b \
    --prompt "A cat walks on the grass, realistic"

# Image-to-Video (I2V)
python run_wan2.2_ti2v.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_wan2.2_ti2v_5b \
    --image input.png \
    --prompt "A cat walks on the grass, realistic"
```

## Architecture Overview

The Wan2.2 pipeline has 4 components compiled for Neuron, plus a CPU-based VAE encoder for I2V mode:

```
Text Prompt                Input Image (I2V only)
    |                           |
    v                           v
[Text Encoder]            [VAE Encoder]    CPU (Neuron bug NCC_IBIR158)
  UMT5, TP=4                   |
    |                     [quant_conv]     CPU, float32
    |                           |
    v                           v
[Transformer]   DiT-based diffusion, 50 denoising steps
    |            TP=4, CP=2, world_size=8 (CP or CFG Parallel)
    v            (I2V: frame 0 = image latent, frames 1-N = noise)
[post_quant_conv]  3D convolution, float32
    |
    v
[VAE Decoder]   Conv3D upsampling, bfloat16, rolling cache
    |
    v
Video Output (512x512, 81 frames)
```

### Performance Breakdown (512x384, 81 frames, CFG Parallel, trn2.48xlarge)

| Component | Time | Details |
|-----------|------|---------|
| Text Encoder | ~0.06s | UMT5, single call |
| Transformer | ~14s | 50 steps @ 0.27s/step (CFG Parallel, 1 forward/step) |
| VAE Decoder | ~3.7s | 11 calls (Stateful Rolling Cache, on-device) |
| Cache reset | ~0.5s | Parallel write zeros to 34 on-device buffers |
| post_quant_conv | ~0.003s | Single call |
| **Total** | **~18s** | CFG Parallel + Stateful rolling cache |

### Performance Breakdown (512x384, 81 frames, A10G g5.8xlarge)

| Component | Time | Details |
|-----------|------|---------|
| Text Encoder | ~432s | UMT5 on CPU (excluded from inference timing) |
| Transformer | ~104s | 50 steps @ 2.08s/step (bf16, single GPU) |
| VAE Decoder | ~13s | diffusers default chunked decode |
| **Total** | **~117s** | Text encoder on CPU, transformer+VAE on GPU |

- A10G is ~7.3x slower than H100 per denoising step, consistent with the compute gap (H100: 990 TFLOPS bf16 vs A10G: 125 TFLOPS bf16).
- Text encoder (UMT5, ~4.7B params) runs on CPU because A10G 24GB VRAM cannot fit all three models simultaneously. Each prompt encoding takes ~432s on CPU vs ~0.06s on GPU.

## Compilation

```bash
# Context Parallel (default)
./compile.sh [output_dir] [compiler_workdir]

# CFG Parallel (recommended for most resolutions)
CFG_PARALLEL=1 ./compile.sh [output_dir] [compiler_workdir]

# Default output: /opt/dlami/nvme/compiled_models_wan2.2_ti2v_5b
```

The compilation script compiles all components:
- **Text Encoder**: UMT5, TP=4, world_size=8
- **Transformer**: TP=4, CP=2, world_size=8 (Context Parallel or CFG Parallel)
- **Decoder (Rolling Cache)**: bfloat16, `--model-type=unet-inference`, flicker-free temporal caching
- **post_quant_conv**: float32

For multi-resolution compilation (including 720P with tiled decoder), use `test_resolutions.sh` (supports `CFG_PARALLEL=1`).

## Key Optimizations

### 1. Context Parallel & CFG Parallel (Transformer)

The transformer uses TP=4 for model parameter sharding and 2-way data parallelism (world_size=8). Two modes are supported:

**Context Parallel (CP)** — splits the sequence dimension across 2 ranks:
- Each CP rank processes 1/2 of the sequence tokens
- Self-attention requires K/V all-gather across CP ranks
- Cross-attention (text conditioning) doesn't need CP since text is shared
- Better for very long sequences where communication cost is small relative to compute

**CFG Parallel** — splits the batch dimension (cond/uncond) across 2 ranks:
- Each rank processes the full sequence for one batch item (cond or uncond)
- No K/V all-gather needed (each rank has full sequence)
- Reduces 2 forward passes per denoising step to 1 (batch=2)
- Eliminates one device↔CPU sync per step, improving device utilization (~75% vs ~50%)
- Best for short-to-medium sequences; at very long sequences the doubled attention compute O(n²) can outweigh communication savings

Compile with `CFG_PARALLEL=1 ./compile.sh` to use CFG Parallel. The inference script auto-detects the mode from the compiled model config.

Implementation: `neuron_wan2_2_ti2v/compile_transformer.py`

### 2. local_rms_norm (Compiler Bug Workaround)

The Neuron compiler generates incorrect all-reduce replica groups `[[0,1,2,3]]` for `DistributedRMSNorm`, which causes assertion errors at runtime with world_size=8 (expecting `[[0,1,2,3],[4,5,6,7]]`).

Solution: `local_rms_norm` computes RMSNorm locally on each rank's shard without any all-reduce:

```python
def local_rms_norm(x, weight, eps=1e-6):
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    return (weight * x_normed).to(x.dtype)
```

Applied to Q/K normalization in both self-attention and cross-attention. The difference from global norm is negligible for QK normalization since each rank's local hidden dimension (~1024) is large enough for stable statistics.

### 3. VAE Decoder: bfloat16 + Rolling Cache

The VAE decoder is dominated by Conv3D operations. Two key optimizations:

**bfloat16**: Halves memory bandwidth (the bottleneck for Conv3D).

**Stateful Rolling Cache**: The decoder's `feat_cache` (34 tensors, ~960MB) carries temporal context between chunked decoder calls. Unlike NoCache mode (which zeros the cache each call and causes flickering), Rolling Cache maintains temporal coherence for flicker-free video. The cache tensors are registered as `nn.Module` buffers, enabling automatic input-output aliasing in the Neuron compiler — the cache stays on-device (HBM) between calls, eliminating ~960MB host↔device roundtrip per call. Only the latent input (~300KB) is transferred each call.

### 4. Correct Compiler Model Type

ModelBuilder defaults to `--model-type=transformer` which optimizes for attention patterns. The VAE decoder is Conv3D-heavy, so we explicitly pass `--model-type=unet-inference`:

```python
traced_decoder = decoder_builder.compile(
    compiler_args="--model-type=unet-inference -O1 --auto-cast=none",
)
```

### 5. VAE Encoder (Image-to-Video)

For I2V mode, the input image is encoded into latent space using the VAE encoder + quant_conv. These run on CPU due to a Neuron compiler bug (NCC_IBIR158) in the Conv3D tensorizer. Since encoding runs only once per video, the overhead is negligible.

### 6. Temporal Chunked Decoding

The VAE decoder processes latent frames in chunks of 2 (CACHE_T=2) with causal temporal caching (`feat_cache`). For 81 frames (21 latent frames):
- Call 1: First frame (with `first_chunk=True`)
- Calls 2-11: Two frames per call

The compile scripts patch the diffusers `autoencoder_kl_wan.py` in-place via `sed` to replace `nearest-exact` with `nearest` for Trainium2 compatibility.

### 7. Tiled Spatial Decode (720P+)

At 720P (1280x704), the VAE decoder's Conv3D operators exceed the Neuron compiler's per-operator instruction limit (`NCC_EXTP003`: 1.2M instructions vs 300K limit). This is different from the total NEFF instruction limit (`NCC_EVRF007`) which can be bypassed with `--tiled-inst-limit`.

**Solution**: Compile the decoder at a small tile resolution (e.g., 384x512), then tile the full-resolution latent at inference time with overlap blending.

**Key design points**:
- The VAE's `feat_cache` is purely temporal (dim=2, CACHE_T=2) with no spatial context, so spatial tiles are fully independent
- All Conv3D kernels use 3x3x3 with padding=1 (same-padding), so spatial tiling introduces no boundary artifacts beyond the overlap region
- Each tile maintains its own independent rolling cache (34 tensors per tile)
- Memory-efficient: processes all tiles for one temporal chunk before moving to the next chunk

**Tiling parameters** (for 1280x704 with 384x512 tiles):
- Latent space: 44x80 → tiled with 24x32 tiles, overlap=4 latent pixels
- Produces 3x3 = 9 tiles per temporal chunk
- Overlap regions use linear ramp blending weights

**Blending**: Each tile gets a 2D weight mask with linear ramps in overlap regions:
- Interior pixels: weight = 1.0
- Overlap pixels: linear ramp from 0.0 to 1.0
- Image boundary pixels: weight = 1.0 (no ramp at edges)

Implementation: `DecoderWrapperV3Tiled` in `neuron_wan2_2_ti2v/neuron_commons.py`

## File Structure

### Compilation Scripts (`neuron_wan2_2_ti2v/`)

| File | Description |
|------|-------------|
| `compile_transformer.py` | Transformer (TP=4, CP=2 or CFG Parallel, local_rms_norm) |
| `compile_text_encoder.py` | Text encoder (ModelBuilder API) |
| `compile_decoder_nocache.py` | VAE decoder (bfloat16, NoCache, `--model-type=unet-inference`) |
| `compile_decoder_rolling.py` | VAE decoder with rolling cache (default, flicker-free) |
| `compile_decoder.py` | VAE decoder with external feat_cache (legacy) |
| `compile_encoder.py` | VAE encoder + quant_conv (unused due to NCC_IBIR158) |
| `cache_hf_model.py` | Download and cache HuggingFace model |

### Runtime

| File | Description |
|------|-------------|
| `run_wan2.2_ti2v.py` | Inference script (T2V and I2V) |
| `run_wan2.2_ti2v_gpu.py` | GPU inference script (H100 benchmark) |
| `bench_a10g.py` | A10G benchmark (text encoder on CPU, transformer+VAE on GPU) |

### Wrappers and Utilities (`neuron_wan2_2_ti2v/`)

| File | Description |
|------|-------------|
| `neuron_commons.py` | Decoder/encoder wrappers, attention utilities |
| `neuron_parallel_utils.py` | Tensor parallel utilities for UMT5 sharding |
| `distributed_rmsnorm.py` | Distributed RMSNorm (reference, not used due to compiler bug) |

### Shell Scripts

| File | Description |
|------|-------------|
| `setup_nvme.sh` | Mount NVMe RAID0 storage at `/opt/dlami/nvme` |
| `compile.sh` | Full compilation pipeline |
| `test_resolutions.sh` | Multi-resolution test suite (auto-tiling for 720P+) |

## Inference Options

### Text-to-Video (T2V)

```bash
python run_wan2.2_ti2v.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_wan2.2_ti2v_5b \
    --prompt "A cat walks on the grass, realistic" \
    --negative_prompt "blurred, low quality, static" \
    --output output.mp4
```

### Image-to-Video (I2V)

```bash
python run_wan2.2_ti2v.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_wan2.2_ti2v_5b \
    --image input.png \
    --prompt "A cat walks on the grass, realistic" \
    --output output_i2v.mp4
```

The I2V pipeline encodes the input image into the first latent frame, then generates the remaining frames via diffusion. The VAE encoder runs on CPU due to a Neuron compiler bug (NCC_IBIR158).

| Argument | Default | Description |
|----------|---------|-------------|
| `--compiled_models_dir` | `/opt/dlami/nvme/compiled_models_wan2.2_ti2v_5b` | Compiled model directory |
| `--image` | None | Input image for I2V (omit for T2V) |
| `--height` | 512 | Video height |
| `--width` | 512 | Video width |
| `--num_frames` | 81 | Number of frames (81 = 3.4s @ 24fps) |
| `--num_inference_steps` | 50 | Denoising steps (lower = faster but less quality) |
| `--max_sequence_length` | 512 | Max text token length |
| `--output` | `output.mp4` | Output video path |
| `--fps` | 16 | Output video FPS |
| `--num_runs` | 1 | Number of inference runs (for benchmarking, reports avg/min/max) |

## Environment

- **Instance**: trn2.48xlarge (8 Neuron cores)
- **Neuron SDK**: PyTorch 2.9 + NxD Inference
- **Virtual env**: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
- **Storage**: NVMe at `/opt/dlami/nvme` (recommended for compiled models)

### Required Environment Variables (set automatically by run scripts)

```bash
NEURON_RT_NUM_CORES=8
LOCAL_WORLD_SIZE=8
NEURON_RT_VIRTUAL_CORE_SIZE=2
NEURON_LOGICAL_NC_CONFIG=2
```

## Troubleshooting

### "nearest-exact" interpolation error
The compile scripts patch the diffusers `autoencoder_kl_wan.py` in-place via `sed` to replace `nearest-exact` with `nearest` for Trainium2 compatibility.

### Replica groups assertion error
If you see errors about replica groups `[[0,1,2,3]]` vs expected `[[0,1,2,3],[4,5,6,7]]`, this is the Neuron compiler bug with `DistributedRMSNorm`. The transformer uses `local_rms_norm` to avoid this.

### Out of memory
- Compiled models should be stored on NVMe (`/opt/dlami/nvme/`), not the root EBS volume
- The decoder uses bfloat16 to reduce memory

### Missing compiled models
All models (text encoder, transformer, decoder, post_quant_conv) must be compiled before inference. If any compiled model is missing, the run script will raise a `RuntimeError`. Run `compile.sh` first. The only exception is the VAE encoder (I2V mode), which runs on CPU due to a Neuron compiler bug (NCC_IBIR158).
