# FastFractal 
Package for fast image fractal conversion.  
**Warning**, the package is in its early stages and some functionalities may not work optimally.

## 5. Parameter reference

This section describes the encoding parameters you are expected to tune. Unless stated otherwise, all parameters apply to `encode_array`, and `encode`/`encode_to_file` pass them through.

### 5.1 Core parameters

- `pool_blocks: list[int]`  
  Block sizes (in pixels) used for range blocks. Typical: `[8]`, `[16]`, or multi-scale like `[16, 8]`.

- `dom_stride: int`  
  Stride (in pixels) for enumerating domain candidates. Smaller stride increases candidates (slower) but often improves quality.

- `k: int`  
  Number of best domain candidates retained per range (Top‑K). Larger improves search coverage but increases compute.

- `iters: int`  
  Iteration hint stored in the code; decoding still uses the explicit `iterations` parameter. Typical: 6–12.

- `quantized: bool`  
  If `True`, store parameters in 8‑bit quantized form (smaller codes, faster IO). If `False`, store float32 scaling/offset (larger codes).

- `s_clip: float`  
  Maximum absolute scaling factor during quantization. Scaling is clipped to `[-s_clip, +s_clip]` before mapping to 8‑bit.

- `o_min: float`, `o_max: float`  
  Output offset range used during quantization, mapped to 8‑bit. In normalized images, typical is `[0.0, 1.0]`.

- `transform_ids: list[int] | None`  
  Subset of transform IDs to consider. `None` defaults to canonical `0..7`.

### 5.2 Advanced parameters

- `backend: str`  
  Implementation backend for similarity/top‑k selection (pure Python variants). Use the library default unless benchmarking.

- `use_quadtree: bool`  
  Enables adaptive partitioning. When `True`, blocks may be split recursively if fit error is above threshold.

- `qt_min_size: int`  
  Minimum leaf size for quadtree splitting.

- `qt_eps: float`  
  Error threshold for quadtree decisions. Lower values split more aggressively (slower, potentially higher quality).

- `use_buckets: bool`  
  Enables bucket-based candidate reduction (speed optimization). Useful for large images.

- `bucket_bits: int`  
  Controls bucket granularity (more bits → more buckets, often better quality but less pruning benefit).

- `seed: int`  
  Random seed used where the search uses randomized sampling.

- `entropy_thresh: float`  
  Optional heuristic threshold for skipping low-entropy content (if supported by your build). Keep default unless you know you need it.

---

## 6. Benchmarks and parameter-space search (config tutorial)

The recommended workflow is:
1. Define a **benchmark dataset** (images, sizes, color modes).
2. Define a **parameter space** via a config file.
3. Run benchmarks with **warmup** and repeated measurements.
4. Select best settings using a primary objective (e.g., PSNR at fixed runtime budget).

### 6.1 Benchmark dataset layout

A minimal benchmark directory:

```
bench/
  images/
    lena.png
    peppers.png
    barnsley_fern.png
  results/
  configs/
```

Recommendations:
- Use a mix of textures (high-frequency) and smooth regions.
- Include both grayscale and RGB if you intend to support both.
- Keep a fixed input normalization (the library expects float32 in [0,1]).

### 6.2 Config file: defining a parameter grid

Use a simple YAML (or JSON) “grid search” format: each key maps to a list of candidate values. The benchmark runner enumerates the cartesian product.

Example (`bench/configs/grid.yaml`):

```yaml
images:
  - "bench/images/lena.png"
  - "bench/images/peppers.png"

decode_iterations: [6, 8, 10]

grid:
  pool_blocks:
    - [16]
    - [16, 8]

  dom_stride: [4, 8]
  k: [16, 32]
  quantized: [true]

  transform_ids:
    - [0, 1, 2, 3, 4, 5, 6, 7]
    - [0, 4]
    - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  use_quadtree: [false, true]
  qt_min_size: [8]
  qt_eps: [0.01, 0.02]
```

How to interpret `transform_ids` in the grid:
- Each entry is a **single candidate set** of transforms.
- The benchmark runner should pass that list directly into `encode_array(..., transform_ids=...)`.


### 6.3 Benchmark run settings

To produce stable results:
- **Warmup:** run 1–3 encode/decode cycles before timing (CPU caches, JIT effects in NumPy).
- **Iterations:** time each configuration for at least 3–10 runs and report mean/median.
- **Metrics:** at minimum:
  - Runtime (encode and decode separately)
  - Reconstruction quality (e.g., PSNR; optionally SSIM)
  - Code size (bytes)

A typical reporting row:
- `image`, `H×W×C`, `params`, `encode_ms`, `decode_ms@iters`, `psnr_db`, `size_bytes`

