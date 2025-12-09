# RightNow Tile - Enhanced CUDA to cuTile Transpiler

## Architecture Overview

RightNow Tile is a production-grade transpiler that converts CUDA SIMT (Single Instruction, Multiple Threads) kernels to cuTile Python code optimized for NVIDIA Blackwell GPUs.

```
CUDA Source
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    ANALYSIS LAYER                           │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌─────────┐ │
│  │   AST    │──▶│ Enhanced │──▶│ Semantic  │──▶│ Pattern │ │
│  │Extractor │   │  Parser  │   │ Analyzer  │   │Detector │ │
│  └──────────┘   └──────────┘   └───────────┘   └─────────┘ │
│                       │        ┌───────────┐        │       │
│                       └───────▶│  Memory   │────────┘       │
│                                │ Analyzer  │                │
│                                └───────────┘                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 BLACKWELL GPU TARGET                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐  │
│  │    IR    │──▶│    IR    │──▶│ Template │──▶│ cuTile  │  │
│  │ Builder  │   │Optimizer │   │ CodeGen  │   │ Output  │  │
│  └──────────┘   └──────────┘   └──────────┘   └─────────┘  │
│                       ▲              │                      │
│                       │        ┌─────┴─────┐               │
│            optimization hints  │Diagnostics│               │
│                                └───────────┘               │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
lib/
├── transpiler.ts              # Main entry point
├── parser/                    # Enhanced parsing
│   ├── intrinsics.ts          # CUDA intrinsics database (~100)
│   ├── cuda-parser.ts         # Enhanced parser with semantic analysis
│   └── index.ts
├── ast/                       # AST and analysis
│   ├── types.ts               # Type definitions
│   ├── extractor.ts           # Kernel extraction
│   ├── semantic-analyzer.ts   # Data flow analysis
│   ├── memory-analyzer.ts     # Memory pattern analysis
│   └── index.ts
├── patterns/                  # Pattern detection
│   ├── types.ts               # Pattern types and variants
│   ├── matcher.ts             # Pattern orchestrator
│   └── matchers/
│       ├── elementwise.ts     # Elementwise operations
│       ├── gemm.ts            # Matrix multiplication
│       ├── reduction.ts       # Parallel reductions
│       ├── scan.ts            # Prefix sums
│       ├── stencil.ts         # Neighbor computations
│       ├── histogram.ts       # Histogram patterns
│       └── sparse.ts          # Sparse matrix operations
├── ir/                        # Intermediate representation
│   ├── types.ts               # Enhanced IR types
│   ├── builder.ts             # IR construction
│   ├── optimizer.ts           # Tile optimization
│   └── index.ts
├── codegen/                   # Code generation
│   ├── generator.ts           # Generic code generator
│   ├── templates/             # Variant-specific templates
│   │   ├── reduction.ts       # Reduction variants
│   │   ├── stencil.ts         # Stencil variants
│   │   └── index.ts
│   └── index.ts
└── validation/                # Validation and diagnostics
    ├── validator.ts           # Semantic validation
    ├── diagnostics.ts         # Error/warning system
    └── index.ts
```

## Components

### 1. Parser Enhancement (`lib/parser/`)

#### CUDA Intrinsics Database (`intrinsics.ts`)
Comprehensive database of ~100 CUDA intrinsics organized by category:

| Category | Examples | Pattern Hints |
|----------|----------|---------------|
| Warp Shuffle | `__shfl_down_sync`, `__shfl_xor_sync` | Reduction, Scan |
| Warp Vote | `__ballot_sync`, `__any_sync` | Reduction |
| Warp Reduce | `__reduce_add_sync`, `__reduce_max_sync` | Reduction |
| Atomic | `atomicAdd`, `atomicMax`, `atomicCAS` | Reduction, Histogram |
| Sync | `__syncthreads`, `__syncwarp` | All patterns |
| Math | `__fmaf`, `__expf`, `__rsqrtf` | Elementwise |
| Memory | `__ldg`, `__ldcs` | All patterns |

#### Enhanced Parser (`cuda-parser.ts`)
Advanced parsing capabilities:
- **Intrinsic Call Detection**: Identifies all CUDA intrinsic usage
- **Index Pattern Analysis**: Classifies memory access patterns (linear, 2D, strided, indirect)
- **Warp Operation Detection**: Finds warp-level primitives
- **Loop Analysis**: Detects stride halving/doubling, nested loops
- **Reduction Pattern Detection**: Identifies accumulation patterns
- **Shared Memory Analysis**: Tracks declarations and bank conflicts

### 2. Semantic Analysis (`lib/ast/semantic-analyzer.ts`)

Performs deep semantic analysis:

```typescript
interface SemanticAnalysisResult {
  reductionVariables: ReductionVariable[];   // sum, max, min, prod
  inductionVariables: InductionVariable[];   // Loop counters
  accessPatterns: AccessPatternClassification[];
  dataFlow: DataFlowInfo;                    // Dependencies
  hasBarrierDivergence: boolean;             // Deadlock risk
  possibleRaces: RaceCondition[];            // Race detection
  parallelismType: ParallelismType;
  computeIntensity: ComputeIntensityMetrics;
}
```

**Key Analyses:**
- Reduction variable detection (`sum += x`, `max = max(max, x)`)
- Data dependency analysis (RAW, WAR, WAW)
- Race condition detection in shared memory
- Barrier divergence checking
- Compute intensity estimation (FLOPS/byte)

### 3. Memory Analysis (`lib/ast/memory-analyzer.ts`)

Analyzes memory access patterns for optimization:

```typescript
interface MemoryAnalysisResult {
  globalMemory: GlobalMemoryAnalysis;        // Coalescing score
  sharedMemory: SharedMemoryAnalysis;        // Bank conflicts
  registers: RegisterAnalysis;               // Spilling risk
  tileRecommendation: TileRecommendation;    // Optimal sizes
  accessSummary: AccessSummary;              // Locality info
  optimizationHints: MemoryOptimizationHint[];
}
```

**Capabilities:**
- Coalescing efficiency scoring (0-100%)
- Bank conflict risk assessment
- Register pressure estimation
- Tile size recommendations per pattern
- Vectorization opportunities

### 4. Pattern Detection (`lib/patterns/`)

#### Supported Patterns (7 types)

| Pattern | Description | Variants |
|---------|-------------|----------|
| **Elementwise** | Per-element operations | `simple`, `vectorized` |
| **GEMM** | Matrix multiplication | `naive_gemm`, `tiled_gemm`, `register_blocked` |
| **Reduction** | Parallel aggregation | `tree_reduction`, `warp_shuffle`, `multi_block`, `segmented` |
| **Scan** | Prefix sums | `inclusive_scan`, `exclusive_scan`, `segmented_scan` |
| **Stencil** | Neighbor computations | `stencil_1d_3pt`, `stencil_1d_5pt`, `stencil_2d_5pt`, `stencil_2d_9pt`, `stencil_3d` |
| **Histogram** | Binned counting | `histogram_atomic`, `histogram_privatized` |
| **Sparse** | Sparse matrix ops | `spmv_csr`, `spmv_coo`, `spmv_ell` |

#### Evidence-Based Matching
Each pattern matcher uses weighted evidence:

```typescript
// Example: Reduction detection
if (hasStrideHalving) {
  addEvidence(evidence, 'stride_halving', 0.35, 'Stride halving detected');
}
if (hasWarpShuffle) {
  addEvidence(evidence, 'warp_shuffle', 0.15, 'Warp shuffle instructions');
}
// Negative evidence
if (hasMatrixMultiply) {
  addEvidence(evidence, 'matrix_multiply', -0.20, 'Likely GEMM');
}
```

### 5. IR Optimization (`lib/ir/optimizer.ts`)

Optimizes tile configurations based on:
- Pattern archetype and variant
- Memory analysis results
- Architecture constraints (shared memory, registers)

**Tile Configurations by Pattern:**

| Pattern | Default Config | Optimization |
|---------|----------------|--------------|
| GEMM | 128x128x32 | Register blocking, pipeline stages |
| Reduction | 256 threads | Warp shuffle, multi-block |
| Stencil | 16x16 tiles | Halo caching |
| Histogram | 256 threads | Privatization |

### 6. Code Generation (`lib/codegen/`)

#### Variant-Specific Templates
Templates for optimized code generation:

**Reduction Templates:**
- `generateTreeReduction()` - Shared memory tree reduction
- `generateWarpShuffleReduction()` - Warp shuffle intrinsics
- `generateMultiBlockReduction()` - Atomic accumulation
- `generateSegmentedReduction()` - Per-row reduction

**Stencil Templates:**
- `generateStencil1D3Point()` - 1D 3-point stencil
- `generateStencil2D5Point()` - 2D cross stencil
- `generateStencil2D9Point()` - 2D box stencil
- `generateStencil3D()` - 3D 7-point stencil

#### Type Mapping
CUDA to cuTile type conversion:

```typescript
const TYPE_MAP = {
  'float': 'ct.float32',
  'double': 'ct.float64',
  'half': 'ct.float16',
  'int': 'ct.int32',
  'int8_t': 'ct.int8',
  // ... more types
};
```

### 7. Diagnostics System (`lib/validation/diagnostics.ts`)

Comprehensive error/warning reporting:

**Diagnostic Codes:**

| Code | Severity | Category | Description |
|------|----------|----------|-------------|
| E100-E104 | Error | Parse | Parsing errors |
| E300-E301 | Error | Correctness | Race conditions, barrier divergence |
| W200-W204 | Warning | Pattern | Low confidence, ambiguous patterns |
| W300-W303 | Warning | Semantic | Dependencies, missing sync |
| W400-W403 | Warning | Memory | Poor coalescing, bank conflicts |
| I400-I402 | Info | Performance | Optimization suggestions |
| I600-I603 | Info | Performance | Performance hints |

## Usage

### Basic Transpilation

```typescript
import { transpile } from './lib/transpiler';

const cudaCode = `
__global__ void reduce(float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, sdata[0]);
}
`;

const result = await transpile(cudaCode);

console.log(result.pattern.archetype);  // 'reduction'
console.log(result.variant);            // 'tree_reduction' or 'multi_block'
console.log(result.pattern.confidence); // 0.85
console.log(result.tileCode);           // Generated cuTile Python
console.log(result.diagnostics);        // Any warnings/hints
```

### Enhanced Result

```typescript
interface EnhancedTranspileResult {
  tileCode: string;                      // Generated code
  pattern: PatternMatch;                 // Detected pattern
  variant?: PatternVariant;              // Specific variant
  ir: KernelIR;                          // Basic IR
  enhancedIR?: EnhancedKernelIR;         // Optimized IR
  semanticAnalysis?: SemanticAnalysisResult;
  memoryAnalysis?: MemoryAnalysisResult;
  diagnostics?: Diagnostic[];
  validation: ValidationResult;
}
```

## Pattern Detection Examples

### Reduction Detection
```cuda
// Detected as 'reduction' with variant 'warp_shuffle'
for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
```

### Stencil Detection
```cuda
// Detected as 'stencil' with variant 'stencil_2d_5pt'
out[i] = 0.25f * (in[i-1] + in[i+1] + in[i-width] + in[i+width]);
```

### Histogram Detection
```cuda
// Detected as 'histogram' with variant 'histogram_atomic'
int bin = data[tid] / binWidth;
atomicAdd(&hist[bin], 1);
```

### Sparse Matrix Detection
```cuda
// Detected as 'sparse' with variant 'spmv_csr'
for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
    sum += values[j] * x[col_idx[j]];
}
```

## Performance Considerations

### Memory Coalescing
The transpiler analyzes access patterns and warns about:
- Strided access (< 50% efficiency)
- Random/indirect access (< 10% efficiency)
- Non-aligned access

### Bank Conflicts
Shared memory bank conflict detection:
- Stride-based conflicts (multiples of 32)
- Column-major access in 2D arrays
- Suggests padding strategies

### Tile Size Optimization
Automatic tile size selection based on:
- Pattern type (GEMM needs larger tiles)
- Shared memory capacity (48KB default)
- Register pressure
- Occupancy targets

## Future Enhancements

- [ ] Full tree-sitter CUDA parser integration
- [ ] More pattern variants (tensor core, cooperative groups)
- [ ] Auto-tuning for tile sizes
- [ ] Multi-kernel fusion detection
- [ ] Performance modeling and prediction
