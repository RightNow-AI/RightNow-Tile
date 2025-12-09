<p align="center">
  <img src="public/logo.webp" alt="RightNow Tile" width="80" />
</p>

<h1 align="center">RightNow Tile</h1>

<p align="center">
  <strong>CUDA SIMT to cuTile Python Transpiler</strong><br>
  Transform your CUDA kernels for NVIDIA Blackwell GPUs
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://nextjs.org/"><img src="https://img.shields.io/badge/Next.js-16-black" alt="Next.js 16"></a>
  <a href="https://www.typescriptlang.org/"><img src="https://img.shields.io/badge/TypeScript-5.9-blue" alt="TypeScript"></a>
  <a href="https://docs.nvidia.com/cuda/cutile-python/"><img src="https://img.shields.io/badge/cuTile-Blackwell-76B900" alt="cuTile"></a>
  <a href="https://discord.gg/sSJqgNnq6X"><img src="https://img.shields.io/badge/Discord-Join%20Us-5865F2" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://tile.rightnowai.co">Live Demo</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#supported-patterns">Patterns</a> •
  <a href="https://discord.gg/sSJqgNnq6X">Discord</a>
</p>

---

## What is RightNow Tile?

**RightNow Tile** is a production-grade transpiler that converts traditional CUDA SIMT (Single Instruction, Multiple Threads) kernels into [cuTile](https://docs.nvidia.com/cuda/cutile-python/) Python code — NVIDIA's new tile-based programming model optimized for **Blackwell GPUs** (compute capability 10.x+).

Part of the [RightNow AI](https://rightnowai.co) ecosystem — a code editor built for GPU kernel development.

<br>

## Why cuTile?

NVIDIA's cuTile represents a paradigm shift in GPU programming:

| Traditional CUDA | cuTile |
|------------------|--------|
| Thread-centric programming | Tile-centric programming |
| Manual memory coalescing | Automatic tile-based loads |
| Complex index calculations | Declarative tile operations |
| Low-level synchronization | High-level tile semantics |

**RightNow Tile** bridges the gap — take your existing CUDA kernels and transform them for next-gen hardware.

<br>

## Quick Start

```bash
# Clone the repository
git clone https://github.com/RightNow-AI/RightNow-Tile.git
cd RightNow-Tile

# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and start transpiling!

<br>

## Features

### Intelligent Pattern Detection

Automatically identifies **18 computational patterns** with **60+ variant-specific optimizations**:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Your CUDA     │ ──► │  Pattern Match   │ ──► │  Optimized      │
│   Kernel        │     │  + Analysis      │     │  cuTile Code    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### 9-Stage Transpilation Pipeline

```
CUDA Source
    │
    ▼
┌──────────────┐
│ 1. Extractor │  Parse kernel signatures, parameters, memory accesses
└──────┬───────┘
       ▼
┌──────────────┐
│ 2. Parser    │  Recognize 150+ CUDA intrinsics & index patterns
└──────┬───────┘
       ▼
┌──────────────┐
│ 3. Semantic  │  Detect reductions, dependencies, race conditions
└──────┬───────┘
       ▼
┌──────────────┐
│ 4. Memory    │  Analyze coalescing, bank conflicts, access patterns
└──────┬───────┘
       ▼
┌──────────────┐
│ 5. Pattern   │  Match against 18 patterns with confidence scoring
└──────┬───────┘
       ▼
┌──────────────┐
│ 6. IR Build  │  Generate intermediate representation with config
└──────┬───────┘
       ▼
┌──────────────┐
│ 7. Optimize  │  Select optimal tile sizes & configurations
└──────┬───────┘
       ▼
┌──────────────┐
│ 8. CodeGen   │  Apply variant-specific templates
└──────┬───────┘
       ▼
┌──────────────┐
│ 9. Validate  │  Verify correctness & generate diagnostics
└──────┴───────┘
       │
       ▼
  cuTile Python
```

### Modern Developer Experience

- **Monaco Editor** — VS Code-quality editing with syntax highlighting
- **Real-time Transpilation** — See results instantly
- **Dark/Light Themes** — Easy on the eyes
- **Expandable Output** — Full-screen code view
- **One-Click Copy** — Get your code ready to deploy

<br>

## Supported Patterns

### Core Compute Patterns

| Pattern | Variants | Use Cases | Confidence |
|---------|----------|-----------|------------|
| **GEMM** | `naive`, `tiled`, `register_blocked` | Matrix multiplication, deep learning | High |
| **Reduction** | `tree`, `warp_shuffle`, `multi_block`, `segmented` | Sum, max, min, dot product | High |
| **Scan** | `inclusive`, `exclusive`, `segmented` | Prefix sum, stream compaction | High |
| **Stencil** | `1d_3pt`, `1d_5pt`, `2d_5pt`, `2d_9pt`, `3d` | Image processing, PDE solvers | High |
| **Elementwise** | `simple`, `vectorized` | Point-wise operations | High |

### ML/Deep Learning Patterns

| Pattern | Variants | Use Cases | Confidence |
|---------|----------|-----------|------------|
| **Attention** | `flash_attention`, `flash_attention_v2`, `multi_head`, `causal`, `cross` | Transformer models | High |
| **Normalization** | `layernorm`, `rmsnorm`, `batchnorm`, `groupnorm`, `instancenorm` | Neural network layers | High |
| **Convolution** | `conv1d`, `conv2d`, `conv3d`, `depthwise`, `grouped`, `winograd`, `im2col` | CNNs, signal processing | High |
| **Pooling** | `max_pool_2d`, `avg_pool_2d`, `global_avg`, `global_max`, `adaptive` | Feature downsampling | High |
| **Embedding** | `lookup`, `embedding_bag`, `positional` | NLP, recommender systems | Medium |

### LLM/Transformer-Specific Patterns

| Pattern | Variants | Use Cases | Confidence |
|---------|----------|-----------|------------|
| **RoPE** | `standard`, `neox`, `cached` | Rotary position embeddings | High |
| **KV Cache** | `append`, `paged`, `prefix`, `gqa` | LLM inference optimization | High |
| **Quantization** | `int8`, `int4`, `fp8`, `dequantize` | Model compression | Medium |
| **Fused** | `matmul_activation`, `matmul_bias_activation`, `layernorm_residual` | Kernel fusion | Medium |

### Specialized Patterns

| Pattern | Variants | Use Cases | Confidence |
|---------|----------|-----------|------------|
| **FFT** | `radix2`, `radix4`, `radix8`, `inverse`, `real` | Signal processing | High |
| **Sparse** | `spmv_csr`, `spmv_csr_warp`, `spmv_coo`, `spmv_ell`, `spmm`, `sddmm` | Sparse matrix operations | Medium |
| **Histogram** | `atomic`, `privatized`, `multipass`, `weighted`, `2d` | Data distribution, statistics | Medium |
| **Sorting** | `bitonic`, `bitonic_shared`, `radix`, `merge` | Parallel sorting | Medium |

<br>

## Example

**Input: CUDA SIMT Kernel**
```cuda
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**Output: cuTile Python**
```python
import cuda_tile as ct
import cupy

TILE_SIZE = 256

@ct.kernel
def vector_add(a, b, c, n: ct.Constant[int], tile_size: ct.Constant[int]):
    """
    Elementwise kernel - auto-transpiled from CUDA
    Original: vectorAdd
    Confidence: 100%
    """
    pid = ct.bid(0)

    # Load input tiles
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # Compute
    result = a_tile + b_tile

    # Store result
    ct.store(c, index=(pid,), tile=result)


def launch_vector_add(a, b, c):
    """Launch the vector_add kernel"""
    n = a.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, vector_add, (a, b, c, TILE_SIZE))
```

### Flash Attention Example

**Input: Flash Attention CUDA Kernel**
```cuda
__global__ void flash_attention_kernel(
    float* Q, float* K, float* V, float* O,
    int seq_len, int head_dim, float scale
) {
    // Complex multi-phase attention implementation
    // with online softmax and tiled matrix multiply
    ...
}
```

**Output: cuTile Python (Flash Attention)**
```python
import cuda_tile as ct
import cupy

BLOCK_Q = 64
BLOCK_KV = 64

@ct.kernel
def flash_attention(
    Q, K, V, O,
    seq_len_q: ct.Constant[int],
    seq_len_kv: ct.Constant[int],
    head_dim: ct.Constant[int],
    scale: ct.Constant[float],
    block_q: ct.Constant[int],
    block_kv: ct.Constant[int]
):
    """
    Flash Attention kernel - auto-transpiled from CUDA
    Confidence: 95%
    Variant: flash_attention_v2
    """
    block_q_idx = ct.bid(0)
    head_idx = ct.bid(1)

    # Initialize output accumulator and softmax stats
    acc = ct.zeros((block_q, head_dim), dtype=ct.float32)
    m_i = ct.full((block_q,), float('-inf'), dtype=ct.float32)
    l_i = ct.zeros((block_q,), dtype=ct.float32)

    # Load Q tile (stays in registers)
    q_tile = ct.load(Q, index=(head_idx, block_q_idx), shape=(block_q, head_dim))

    # Iterate over K,V blocks with online softmax
    for block_kv_idx in range(0, ct.cdiv(seq_len_kv, block_kv)):
        k_tile = ct.load(K, index=(head_idx, block_kv_idx), shape=(block_kv, head_dim))
        v_tile = ct.load(V, index=(head_idx, block_kv_idx), shape=(block_kv, head_dim))

        # QK^T with scaling
        qk = ct.tile_matmul(q_tile, ct.transpose(k_tile)) * scale

        # Online softmax update
        m_ij = ct.reduce(qk, op=ct.max, axis=1)
        m_new = ct.maximum(m_i, m_ij)
        alpha = ct.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        p = ct.exp(qk - m_new[:, None])
        l_ij = ct.reduce(p, op=ct.sum, axis=1)
        l_i = l_i + l_ij

        # Accumulate output
        acc = acc + ct.tile_matmul(p, v_tile)
        m_i = m_new

    # Normalize and store
    out = acc / l_i[:, None]
    ct.store(O, index=(head_idx, block_q_idx), tile=out)
```

<br>

## API Usage

Use the transpiler programmatically:

```typescript
import { transpile } from './lib/transpiler';

const result = await transpile(cudaCode);

// Access results
result.tileCode              // Generated cuTile Python code
result.pattern.archetype     // Detected pattern (e.g., 'attention', 'gemm')
result.pattern.confidence    // Confidence score (0-1)
result.pattern.variant       // Specific variant (e.g., 'flash_attention_v2')
result.validation.isValid    // Validation status
result.diagnostics           // Warnings and suggestions
result.memoryAnalysis        // Memory access analysis
result.semanticAnalysis      // Semantic analysis results
```

### REST API

```bash
curl -X POST http://localhost:3000/api/transpile \
  -H "Content-Type: application/json" \
  -d '{"code": "__global__ void add(float* a, float* b, float* c, int n) { ... }"}'
```

<br>

## Project Structure

```
rightnow-tile/
├── app/
│   ├── api/transpile/        # REST API endpoint
│   ├── components/           # React components
│   │   ├── ScientificVisualization.tsx
│   │   ├── ThemeProvider.tsx
│   │   └── ThemeToggle.tsx
│   ├── page.tsx              # Main UI
│   └── globals.css           # Styling
├── lib/
│   ├── ast/                  # AST extraction & semantic analysis
│   │   ├── extractor.ts      # Kernel parsing
│   │   ├── semantic-analyzer.ts
│   │   ├── memory-analyzer.ts
│   │   ├── phase-analyzer.ts # Multi-phase kernel detection
│   │   └── types.ts          # 18 archetypes, 60+ variants
│   ├── parser/
│   │   └── intrinsics.ts     # 150+ CUDA intrinsics
│   ├── patterns/             # Pattern matchers (18 patterns)
│   │   └── matchers/
│   │       ├── attention.ts  # Flash Attention, MHA
│   │       ├── fused.ts      # Fused kernels
│   │       ├── fft.ts        # FFT variants
│   │       ├── gemm.ts       # Matrix multiply
│   │       ├── reduction.ts  # Reductions
│   │       ├── scan.ts       # Prefix sums
│   │       ├── stencil.ts    # Stencil patterns
│   │       ├── sparse.ts     # Sparse matrix ops
│   │       ├── histogram.ts  # Histogram
│   │       ├── convolution.ts # CNN convolutions
│   │       ├── sorting.ts    # Sorting algorithms
│   │       ├── pooling.ts    # Pooling layers
│   │       ├── normalization.ts # Norm layers
│   │       ├── embedding.ts  # Embeddings
│   │       ├── rope.ts       # Rotary embeddings
│   │       ├── kvcache.ts    # KV cache ops
│   │       ├── quantization.ts # Quantization
│   │       └── elementwise.ts
│   ├── ir/                   # Intermediate representation
│   │   ├── builder.ts        # 11 specialized IR types
│   │   ├── optimizer.ts
│   │   └── types.ts
│   ├── codegen/              # Code generation
│   │   ├── generator.ts      # Routes to all 18 archetypes
│   │   └── templates/        # 14 template files
│   │       ├── attention.ts
│   │       ├── fused.ts
│   │       ├── sparse.ts
│   │       ├── histogram.ts
│   │       ├── convolution.ts
│   │       ├── sorting.ts
│   │       ├── pooling.ts
│   │       ├── normalization.ts
│   │       ├── embedding.ts
│   │       ├── rope.ts
│   │       ├── kvcache.ts
│   │       ├── quantization.ts
│   │       ├── reduction.ts
│   │       └── stencil.ts
│   ├── validation/           # Validation & diagnostics
│   └── transpiler.ts         # Main entry point
├── docs/                     # Documentation
└── public/                   # Static assets
```

<br>

## Tech Stack

- **Framework**: [Next.js 16](https://nextjs.org/) with Turbopack
- **Language**: [TypeScript 5.9](https://www.typescriptlang.org/)
- **UI**: [React 19](https://react.dev/), [Tailwind CSS](https://tailwindcss.com/), [Framer Motion](https://www.framer.com/motion/)
- **Editor**: [Monaco Editor](https://microsoft.github.io/monaco-editor/)
- **Target**: [NVIDIA cuTile](https://docs.nvidia.com/cuda/cutile-python/)

<br>

## Requirements

- **Node.js** 18+
- **npm** or **yarn**
- For running generated code: **NVIDIA Blackwell GPU** (compute capability 10.x+)

<br>

## Production Deployment

```bash
# Build for production
npm run build

# Start production server
npm start
```

Deploy to Vercel, AWS, or any Node.js hosting platform.

<br>

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** your feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development

```bash
# Run development server
npm run dev

# Type checking
npx tsc --noEmit

# Build
npm run build
```

<br>

## Roadmap

- [x] Support for 18 CUDA patterns with 60+ variants
- [x] Flash Attention and Transformer-specific patterns
- [x] LLM inference patterns (RoPE, KV Cache, Quantization)
- [x] Comprehensive convolution support (Winograd, im2col)
- [ ] Batch transpilation for multiple kernels
- [ ] Performance benchmarking comparisons
- [ ] VS Code extension integration
- [ ] CLI tool for CI/CD pipelines
- [ ] CUDA to Triton transpilation

<br>

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

<br>

## Links

<p align="center">
  <a href="https://rightnowai.co"><strong>RightNow AI</strong></a> · GPU Kernel Code Editor<br><br>
  <a href="https://tile.rightnowai.co">Live Demo</a> •
  <a href="https://docs.nvidia.com/cuda/cutile-python/">cuTile Docs</a> •
  <a href="https://discord.gg/sSJqgNnq6X">Discord</a> •
  <a href="https://github.com/RightNow-AI/RightNow-Tile/issues">Issues</a>
</p>

---

<p align="center">
  Made with ♥ by <a href="https://rightnowai.co">RightNow AI</a>
</p>
