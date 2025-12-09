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
    archetype: string;        // 'gemm' | 'reduction' | 'scan' | 'stencil' | 'elementwise' | 'histogram' | 'sparse'
    variant?: string;         // Pattern variant (e.g., 'warp_shuffle')
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

### Archetypes

| Archetype | Description |
|-----------|-------------|
| `elementwise` | Per-element operations (a[i] = b[i] + c[i]) |
| `gemm` | Matrix multiplication (C = A @ B) |
| `reduction` | Aggregation (sum, max, min) |
| `scan` | Prefix sums (inclusive/exclusive) |
| `stencil` | Neighbor computations (convolution) |
| `histogram` | Binned counting |
| `sparse` | Sparse matrix operations (SpMV) |

### Variants

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

**Sparse:**
- `spmv_csr` - CSR format (row_ptr, col_idx, values)
- `spmv_coo` - COO format (row_ind, col_ind, values)
- `spmv_ell` - ELL format (padded fixed-width)

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
| `half` | `ct.float16` |
| `int` | `ct.int32` |
| `int8_t` | `ct.int8` |
| `int64_t` | `ct.int64` |
| `unsigned int` | `ct.uint32` |
| `size_t` | `ct.int64` |
| `bool` | `ct.bool` |

---

## Examples

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

### 2D Stencil
```cuda
__global__ void jacobi(float* in, float* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * w + x;

    if (x > 0 && x < w-1 && y > 0 && y < h-1) {
        out[i] = 0.25f * (in[i-1] + in[i+1] + in[i-w] + in[i+w]);
    }
}
```

**Detected:** `stencil` / `stencil_2d_5pt` (confidence: 75%)

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
