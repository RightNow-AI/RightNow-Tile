# Changelog

## [2.1.0] - Comprehensive Pattern Support (18 Archetypes, 60+ Variants)

### Added

#### New Pattern Matchers (11 new matchers)

- **Attention Pattern** (`lib/patterns/matchers/attention.ts`)
  - Flash Attention detection (Q @ K^T, online softmax, @ V)
  - Multi-head attention detection
  - Causal/cross attention variants
  - Variants: `flash_attention`, `flash_attention_v2`, `multi_head_attention`, `causal_attention`, `cross_attention`

- **Convolution Pattern** (`lib/patterns/matchers/convolution.ts`)
  - 1D/2D/3D convolution detection
  - Depthwise and grouped convolution
  - Winograd, im2col, implicit GEMM detection
  - Variants: `conv_1d`, `conv_2d`, `conv_3d`, `conv_depthwise`, `conv_grouped`, `conv_winograd`, `conv_im2col`, `conv_implicit_gemm`

- **Sorting Pattern** (`lib/patterns/matchers/sorting.ts`)
  - Bitonic sort network detection
  - Radix sort detection
  - Merge sort detection
  - Variants: `bitonic_sort`, `bitonic_sort_shared`, `radix_sort`, `merge_sort`

- **Pooling Pattern** (`lib/patterns/matchers/pooling.ts`)
  - Max/average pooling detection
  - Global and adaptive pooling
  - Variants: `max_pool_2d`, `avg_pool_2d`, `global_avg_pool`, `global_max_pool`, `adaptive_avg_pool`, `adaptive_max_pool`

- **Normalization Pattern** (`lib/patterns/matchers/normalization.ts`)
  - LayerNorm, BatchNorm, RMSNorm detection
  - GroupNorm, InstanceNorm detection
  - Variants: `layernorm`, `rmsnorm`, `batchnorm`, `groupnorm`, `instancenorm`

- **Embedding Pattern** (`lib/patterns/matchers/embedding.ts`)
  - Embedding table lookup detection
  - Embedding bag (sum/mean) detection
  - Positional embedding detection
  - Variants: `embedding_lookup`, `embedding_bag`, `positional_embedding`

- **RoPE Pattern** (`lib/patterns/matchers/rope.ts`)
  - Rotary Position Embedding detection
  - Standard (LLaMA) and NeoX-style detection
  - Precomputed sin/cos cache detection
  - Variants: `rope_standard`, `rope_neox`, `rope_cached`

- **KV Cache Pattern** (`lib/patterns/matchers/kvcache.ts`)
  - KV cache append detection
  - Paged attention cache detection
  - Prefix caching detection
  - GQA cache detection
  - Variants: `kvcache_append`, `kvcache_paged`, `kvcache_prefix`, `kvcache_gqa`

- **Quantization Pattern** (`lib/patterns/matchers/quantization.ts`)
  - INT8 symmetric/asymmetric quantization
  - INT4 group-wise quantization
  - FP8 (E4M3/E5M2) quantization
  - Dequantization kernels
  - Variants: `quant_int8`, `quant_int4`, `quant_fp8`, `quantize`, `dequantize`

- **FFT Pattern** (`lib/patterns/matchers/fft.ts`)
  - Butterfly computation detection
  - Bit-reversal permutation detection
  - Twiddle factor usage
  - Variants: `fft_radix2`, `fft_radix4`, `fft_radix8`, `inverse_fft`, `real_fft`

- **Fused Pattern** (`lib/patterns/matchers/fused.ts`)
  - Multi-phase kernel detection
  - Fused matmul + activation
  - Fused conv + batchnorm
  - Fused layernorm + residual
  - Variants: `matmul_activation`, `matmul_bias_activation`, `conv_batchnorm`, `layernorm_residual`

#### New Code Generation Templates (14 template files)

- **Attention Templates** (`lib/codegen/templates/attention.ts`)
  - Flash Attention with online softmax
  - Multi-head attention
  - Causal attention with masking

- **Convolution Templates** (`lib/codegen/templates/convolution.ts`)
  - Conv2D with im2col, implicit GEMM
  - Depthwise convolution
  - Winograd convolution

- **Sorting Templates** (`lib/codegen/templates/sorting.ts`)
  - Bitonic sort (shared memory variant)
  - Radix sort with histogram

- **Pooling Templates** (`lib/codegen/templates/pooling.ts`)
  - Max/average pooling 2D
  - Global pooling
  - Adaptive pooling

- **Normalization Templates** (`lib/codegen/templates/normalization.ts`)
  - LayerNorm, RMSNorm, BatchNorm
  - GroupNorm, InstanceNorm

- **Embedding Templates** (`lib/codegen/templates/embedding.ts`)
  - Embedding lookup
  - Embedding bag with reduction

- **RoPE Templates** (`lib/codegen/templates/rope.ts`)
  - Standard RoPE (LLaMA style)
  - NeoX-style interleaved RoPE
  - Cached RoPE with precomputed sin/cos

- **KV Cache Templates** (`lib/codegen/templates/kvcache.ts`)
  - KV cache append
  - Paged attention KV cache
  - GQA cache update

- **Quantization Templates** (`lib/codegen/templates/quantization.ts`)
  - INT8 quantize/dequantize
  - INT4 group-wise quantization
  - FP8 quantization

- **FFT Templates** (`lib/codegen/templates/fft.ts`)
  - Radix-2, Radix-4 FFT
  - Inverse FFT

- **Fused Templates** (`lib/codegen/templates/fused.ts`)
  - Matmul + activation fusion
  - Conv + batchnorm fusion

- **GEMM Templates** (`lib/codegen/templates/gemm.ts`)
  - Naive, tiled, register-blocked GEMM

- **Histogram Templates** (`lib/codegen/templates/histogram.ts`)
  - Atomic and privatized histogram

- **Sparse Templates** (`lib/codegen/templates/sparse.ts`)
  - CSR, COO, ELL SpMV
  - CSR SpMM

#### Enhanced Intrinsics Database

- Added 50+ new intrinsics for LLM patterns
- Type conversion intrinsics (`__float2half`, `__half2float`, etc.)
- Extended math intrinsics (`tanhf`, `erff`, `roundf`, etc.)
- Tensor core intrinsics (wmma operations)
- New pattern hints: convolution, sorting, pooling, normalization, embedding, rope, kvcache, quantization

### Changed

- **Pattern Matcher Orchestrator** (`lib/patterns/matcher.ts`)
  - Registered all 18 pattern matchers
  - Added phase analysis boosts for new patterns
  - Improved pattern disambiguation

- **Code Generator** (`lib/codegen/generator.ts`)
  - Added routing for all 18 archetypes
  - Template selection based on variant
  - Enhanced IR building for new patterns

- **IR Builder** (`lib/ir/builder.ts`)
  - Added specialized IR configs for each new archetype
  - LLM-specific IR configurations (attention, rope, kvcache)
  - ML/DL IR configurations (conv, pool, norm)

- **AST Types** (`lib/ast/types.ts`)
  - Added all 18 archetypes to KernelArchetype
  - Added 60+ variants to PatternVariant

### Fixed

- Syntax error in rope.ts pattern matcher (missing parenthesis)
- Type import error in convolution.ts (PatternEvidence → Evidence)
- Variant type casting in pattern matchers

---

## [2.0.0] - Enhanced Transpiler Architecture

### Added

#### Parser Enhancement
- **CUDA Intrinsics Database** (`lib/parser/intrinsics.ts`)
  - ~100 CUDA intrinsics categorized by type
  - Pattern hints for each intrinsic
  - Categories: warp_shuffle, warp_vote, warp_reduce, atomic, sync, math, memory, bit_manipulation

- **Enhanced CUDA Parser** (`lib/parser/cuda-parser.ts`)
  - Intrinsic call detection and classification
  - Index pattern analysis (linear, 2D, strided, indirect, complex)
  - Warp operation detection
  - Loop analysis with stride halving/doubling detection
  - Reduction pattern detection
  - Accumulation pattern detection
  - Shared memory usage analysis with bank conflict detection
  - Pattern signal generation for improved matching

#### Semantic Analysis Layer
- **Semantic Analyzer** (`lib/ast/semantic-analyzer.ts`)
  - Reduction variable detection (sum, product, max, min)
  - Induction variable detection
  - Access pattern classification
  - Data flow analysis (inputs, outputs, dependencies)
  - Race condition detection
  - Barrier divergence checking
  - Parallelism type classification
  - Compute intensity metrics

- **Memory Analyzer** (`lib/ast/memory-analyzer.ts`)
  - Global memory coalescing analysis
  - Shared memory bank conflict detection
  - Register pressure estimation
  - Tile size recommendations per pattern
  - Memory layout analysis
  - Optimization hint generation

#### New Pattern Matchers
- **Histogram Pattern** (`lib/patterns/matchers/histogram.ts`)
  - Detects atomic increment patterns
  - Bin calculation detection
  - Shared memory privatization detection
  - Variants: `histogram_atomic`, `histogram_privatized`

- **Sparse Matrix Pattern** (`lib/patterns/matchers/sparse.ts`)
  - CSR format detection (row_ptr, col_idx, values)
  - COO format detection
  - ELL format detection
  - Indirect access pattern detection
  - Variants: `spmv_csr`, `spmv_coo`, `spmv_ell`

#### Enhanced Existing Matchers
- **Reduction Matcher** - Added variants:
  - `tree_reduction` - Standard shared memory tree
  - `warp_shuffle` - Warp shuffle intrinsics
  - `multi_block` - Atomic inter-block
  - `segmented` - Per-row/segment reduction

- **Stencil Matcher** - Added variants:
  - `stencil_1d_3pt` - 1D 3-point stencil
  - `stencil_1d_5pt` - 1D 5-point stencil
  - `stencil_2d_5pt` - 2D cross pattern (5-point)
  - `stencil_2d_9pt` - 2D box pattern (9-point)
  - `stencil_3d` - 3D 7-point stencil

#### IR Optimization
- **Enhanced IR Types** (`lib/ir/types.ts`)
  - Extended IR with variant support
  - CUDA to cuTile type mapping
  - Tile strategy definitions
  - Optimization hint types
  - Memory layout info

- **IR Optimizer** (`lib/ir/optimizer.ts`)
  - Pattern-specific tile optimization
  - Variant-aware configuration
  - Memory analysis integration
  - Performance hint generation

#### Code Generation Templates
- **Reduction Templates** (`lib/codegen/templates/reduction.ts`)
  - Tree reduction template
  - Warp shuffle reduction template
  - Multi-block reduction template
  - Segmented reduction template

- **Stencil Templates** (`lib/codegen/templates/stencil.ts`)
  - 1D 3-point stencil template
  - 1D 5-point stencil template
  - 2D 5-point stencil template
  - 2D 9-point stencil template
  - 3D stencil template

#### Diagnostics System
- **Comprehensive Diagnostics** (`lib/validation/diagnostics.ts`)
  - Diagnostic codes (E1xx-E5xx, W2xx-W5xx, I4xx-I6xx)
  - Severity levels (error, warning, info, hint)
  - Categories (parse, pattern, semantic, memory, codegen, performance, correctness)
  - Suggestion system
  - Related diagnostic info

#### Visualization
- **Enhanced Pipeline Visualization** (`app/components/ScientificVisualization.tsx`)
  - 9-stage pipeline visualization
  - Analysis layer (blue) boundary
  - Blackwell GPU target (green) boundary
  - Pattern labels showing all 7 patterns
  - Variant information display
  - Memory analysis feedback loop
  - Tooltips with detailed descriptions

### Changed
- **Transpiler Integration** (`lib/transpiler.ts`)
  - Enhanced result type with semantic/memory analysis
  - Pattern detection including new histogram and sparse patterns
  - Variant-specific template selection
  - Diagnostic integration
  - Confidence boosting from enhanced analysis

- **AST Types** (`lib/ast/types.ts`)
  - Added `PatternVariant` type with all variants
  - Extended `PatternMatch` with optional variant field
  - Added histogram and sparse to `KernelArchetype`

- **Pattern Types** (`lib/patterns/types.ts`)
  - Updated `createPatternMatch` for variant support

### Fixed
- Pattern confidence calculation now includes enhanced analysis signals
- Visualization now accurately reflects the pipeline architecture

## [1.0.0] - Initial Release

### Added
- Basic CUDA to cuTile transpilation
- 5 pattern matchers (elementwise, gemm, reduction, scan, stencil)
- Evidence-weighted confidence scoring
- Template-based code generation
- Basic validation
- Scientific visualization component
- Theme support (light/dark mode)
