# RightNow Tile API Reference

## Quick Start

```typescript
import { transpile, analyze, getAllPatternMatches } from './lib/transpiler';

// Basic transpilation
const result = await transpile(cudaCode);
console.log(result.tileCode);
```

## Core Functions

### `transpile(code: string): Promise<EnhancedTranspileResult>`

Main transpilation function. Converts CUDA to cuTile Python.

**Parameters:**
- `code` - CUDA source code containing `__global__` kernel

**Returns:**
```typescript
{
  tileCode: string;           // Generated cuTile Python code
  pattern: {
    archetype: string;        // See Archetypes below (18 patterns)
    variant?: string;         // Pattern variant (e.g., 'flash_attention_v2')
    confidence: number;       // 0.0 - 1.0
    evidence: Evidence[];     // Detection evidence
    warnings: string[];
  };
  ir: KernelIR;               // Intermediate representation
  enhancedIR?: EnhancedKernelIR;
  semanticAnalysis?: SemanticAnalysisResult;
  memoryAnalysis?: MemoryAnalysisResult;
  diagnostics?: Diagnostic[];
  validation: ValidationResult;
}
```

### `analyze(code: string): PatternAnalysis | null`

Analyze a kernel without generating code.

**Returns:**
```typescript
{
  kernel: CudaKernelInfo;
  allMatches: PatternMatch[];
  bestMatch: PatternMatch;
  confidence: number;
}
```

### `getAllPatternMatches(code: string): PatternMatch[]`

Get all pattern matches sorted by confidence.

---

## Pattern Types

### Archetypes (18 Patterns)

| Archetype | Description |
|-----------|-------------|
| `elementwise` | Per-element operations (a[i] = b[i] + c[i]) |
| `gemm` | Matrix multiplication (C = A @ B) |
| `reduction` | Aggregation (sum, max, min) |
| `scan` | Prefix sums (inclusive/exclusive) |
| `stencil` | Neighbor computations |
| `histogram` | Binned counting |
| `sparse` | Sparse matrix operations (SpMV, SpMM) |
| `attention` | Flash Attention, Multi-Head Attention |
| `fused` | Fused kernels (matmul + activation) |
| `fft` | Fast Fourier Transform |
| `convolution` | 1D/2D/3D convolutions, depthwise, grouped |
| `sorting` | Bitonic, radix, merge sort |
| `pooling` | Max/avg pooling, global pooling |
| `normalization` | LayerNorm, BatchNorm, RMSNorm, GroupNorm |
| `embedding` | Embedding lookup, embedding bag |
| `rope` | Rotary Position Embedding |
| `kvcache` | KV cache operations for LLM inference |
| `quantization` | INT8, INT4, FP8 quantization |

### Variants

**Attention:**
- `flash_attention` - Flash Attention algorithm
- `flash_attention_v2` - Flash Attention v2 with better memory efficiency
- `multi_head_attention` - Standard multi-head attention
- `causal_attention` - Causal/decoder attention with masking
- `cross_attention` - Cross attention (encoder-decoder)

**Normalization:**
- `layernorm` - Layer Normalization
- `rmsnorm` - Root Mean Square Normalization
- `batchnorm` - Batch Normalization
- `groupnorm` - Group Normalization
- `instancenorm` - Instance Normalization

**Convolution:**
- `conv_1d` - 1D convolution
- `conv_2d` - 2D convolution
- `conv_3d` - 3D convolution
- `conv_depthwise` - Depthwise separable convolution
- `conv_grouped` - Grouped convolution
- `conv_winograd` - Winograd convolution
- `conv_im2col` - im2col-based convolution
- `conv_implicit_gemm` - Implicit GEMM convolution

**Pooling:**
- `max_pool_2d` - 2D max pooling
- `avg_pool_2d` - 2D average pooling
- `global_avg_pool` - Global average pooling
- `global_max_pool` - Global max pooling
- `adaptive_avg_pool` - Adaptive average pooling
- `adaptive_max_pool` - Adaptive max pooling

**RoPE (Rotary Position Embedding):**
- `rope_standard` - Standard RoPE (LLaMA style)
- `rope_neox` - NeoX-style interleaved RoPE
- `rope_cached` - RoPE with precomputed sin/cos cache

**KV Cache:**
- `kvcache_append` - Standard KV cache append
- `kvcache_paged` - Paged attention KV cache
- `kvcache_prefix` - Prefix caching for shared prompts
- `kvcache_gqa` - Grouped Query Attention cache

**Quantization:**
- `quant_int8` - INT8 symmetric/asymmetric quantization
- `quant_int4` - INT4 group-wise quantization
- `quant_fp8` - FP8 (E4M3/E5M2) quantization
- `quantize` - Quantization kernel
- `dequantize` - Dequantization kernel

**Reduction:**
- `tree_reduction` - Shared memory tree reduction
- `warp_shuffle` - Uses `__shfl_down_sync`
- `multi_block` - Atomic inter-block accumulation
- `segmented` - Per-row/segment reduction

**Stencil:**
- `stencil_1d_3pt` - 1D 3-point (arr[i-1], arr[i], arr[i+1])
- `stencil_1d_5pt` - 1D 5-point
- `stencil_2d_5pt` - 2D cross pattern
- `stencil_2d_9pt` - 2D 3x3 box
- `stencil_3d` - 3D 7-point

**GEMM:**
- `naive_gemm` - Simple triple loop
- `tiled_gemm` - Shared memory tiling
- `register_blocked` - Register-level blocking

**Histogram:**
- `histogram_atomic` - Direct atomic increment
- `histogram_privatized` - Shared memory privatization
- `histogram_multipass` - Multi-pass histogram
- `histogram_weighted` - Weighted histogram
- `histogram_2d` - 2D histogram

**Sparse:**
- `spmv_csr` - CSR format SpMV
- `spmv_csr_warp` - Warp-level CSR SpMV
- `spmv_coo` - COO format SpMV
- `spmv_ell` - ELL format SpMV
- `spmm_csr` - CSR format SpMM
- `sddmm` - Sampled Dense-Dense Matrix Multiply

**Sorting:**
- `bitonic_sort` - Bitonic sorting network
- `bitonic_sort_shared` - Shared memory bitonic sort
- `radix_sort` - Radix sort
- `merge_sort` - Merge sort

**FFT:**
- `fft_radix2` - Radix-2 FFT
- `fft_radix4` - Radix-4 FFT
- `fft_radix8` - Radix-8 FFT
- `inverse_fft` - Inverse FFT
- `real_fft` - Real-to-complex FFT

**Fused:**
- `matmul_activation` - Matrix multiply + activation
- `matmul_bias_activation` - Matrix multiply + bias + activation
- `conv_batchnorm` - Convolution + batch normalization
- `layernorm_residual` - LayerNorm + residual add
- `multi_phase_fused` - Multi-phase fused kernel

**Embedding:**
- `embedding_lookup` - Standard embedding lookup
- `embedding_bag` - Sum/mean of multiple embeddings
- `positional_embedding` - Positional embedding addition

---

## Semantic Analysis

```typescript
interface SemanticAnalysisResult {
  reductionVariables: ReductionVariable[];
  inductionVariables: InductionVariable[];
  accessPatterns: AccessPatternClassification[];
  dataFlow: DataFlowInfo;
  hasBarrierDivergence: boolean;
  possibleRaces: RaceCondition[];
  parallelismType: ParallelismType;
  computeIntensity: ComputeIntensityMetrics;
}
```

### ReductionVariable
```typescript
{
  name: string;
  operation: 'sum' | 'product' | 'max' | 'min' | 'and' | 'or';
  scope: 'warp' | 'block' | 'global';
  usesAtomic: boolean;
  usesWarpShuffle: boolean;
  confidence: number;
}
```

---

## Memory Analysis

```typescript
interface MemoryAnalysisResult {
  globalMemory: {
    coalescingScore: number;      // 0.0 - 1.0
    hasCoalescedReads: boolean;
    stridedAccessCount: number;
    randomAccessCount: number;
  };
  sharedMemory: {
    isUsed: boolean;
    totalBytes: number;
    bankConflictRisk: number;     // 0.0 - 1.0
  };
  tileRecommendation: {
    recommended: TileConfig;
    alternatives: TileConfig[];
    justification: string[];
  };
  optimizationHints: MemoryOptimizationHint[];
}
```

---

## Diagnostics

### Diagnostic Codes

**Errors (Exx):**
- `E100` - Parse error
- `E300` - Race condition detected
- `E301` - Barrier divergence
- `E500` - Cannot generate code

**Warnings (Wxx):**
- `W200` - Low pattern confidence
- `W201` - Ambiguous pattern
- `W400` - Poor memory coalescing
- `W401` - Bank conflict risk

**Info (Ixx):**
- `I400` - Suboptimal tile size
- `I401` - Vectorization opportunity
- `I600` - Performance optimization available

### Accessing Diagnostics
```typescript
const result = await transpile(code);

for (const diag of result.diagnostics || []) {
  console.log(`[${diag.severity}] ${diag.code}: ${diag.message}`);
  if (diag.suggestions) {
    diag.suggestions.forEach(s => console.log(`  - ${s}`));
  }
}
```

---

## Type Mapping

| CUDA Type | cuTile Type |
|-----------|-------------|
| `float` | `ct.float32` |
| `double` | `ct.float64` |
| `half` / `__half` | `ct.float16` |
| `__nv_bfloat16` | `ct.bfloat16` |
| `int` | `ct.int32` |
| `int8_t` | `ct.int8` |
| `int64_t` | `ct.int64` |
| `unsigned int` | `ct.uint32` |
| `size_t` | `ct.int64` |
| `bool` | `ct.bool` |

---

## Examples

### Flash Attention
```cuda
__global__ void flash_attention(
    float* Q, float* K, float* V, float* O,
    int seq_len, int head_dim, float scale
) {
    // Multi-phase attention with online softmax
    extern __shared__ float smem[];
    // ... Q @ K^T, softmax, @ V
}
```

**Detected:** `attention` / `flash_attention` (confidence: 92%)

### LayerNorm
```cuda
__global__ void layer_norm(float* x, float* y, float* gamma, float* beta,
                           int N, int D, float eps) {
    int row = blockIdx.x;
    float mean = 0, var = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        mean += x[row * D + i];
    // ... reduction, normalize, scale + shift
}
```

**Detected:** `normalization` / `layernorm` (confidence: 88%)

### Reduction Kernel
```cuda
__global__ void sum_reduce(float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, sdata[0]);
}
```

**Detected:** `reduction` / `multi_block` (confidence: 85%)

### 2D Convolution
```cuda
__global__ void conv2d(float* input, float* kernel, float* output,
                       int H, int W, int C, int K, int kH, int kW) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // ... sliding window convolution
}
```

**Detected:** `convolution` / `conv_2d` (confidence: 82%)

### SpMV CSR
```cuda
__global__ void spmv_csr(int* row_ptr, int* col_idx, float* vals,
                         float* x, float* y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0;
        for (int j = row_ptr[row]; j < row_ptr[row+1]; j++) {
            sum += vals[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}
```

**Detected:** `sparse` / `spmv_csr` (confidence: 80%)

### INT8 Quantization
```cuda
__global__ void quantize_int8(float* input, int8_t* output,
                               float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx] / scale;
        val = fmaxf(-128.0f, fminf(127.0f, roundf(val)));
        output[idx] = (int8_t)val;
    }
}
```

**Detected:** `quantization` / `quant_int8` (confidence: 78%)
