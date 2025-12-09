# Changelog

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
