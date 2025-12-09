// AST Types for CUDA Kernel Analysis

export interface CudaKernelInfo {
  name: string;
  parameters: ParameterInfo[];
  memoryAccesses: MemoryAccess[];
  sharedMemoryDecls: string[];
  loops: LoopInfo[];
  syncPoints: SyncPoint[];
  threadIndexUsage: ThreadIndexUsage;
  sourceText: string;
}

export interface ParameterInfo {
  name: string;
  type: string;
  isPointer: boolean;
  isConst: boolean;
}

export interface MemoryAccess {
  array: string;
  indexExpression: string;
  indexVars: string[];
  accessType: 'read' | 'write';
  hasNeighborOffset: boolean;
  offset?: number;
  line: number;
}

export interface LoopInfo {
  loopType: 'for' | 'while' | 'do-while';
  initVar: string;
  condition: string;
  update: string;
  body: string;
  nestLevel: number;
  containsSyncthreads: boolean;
  hasStrideHalving: boolean;
  hasStrideDoubling: boolean;
  startLine: number;
  endLine: number;
}

export interface SyncPoint {
  type: 'syncthreads' | 'syncwarp' | 'atomic';
  name: string;
  line: number;
}

export interface ThreadIndexUsage {
  usesThreadIdxX: boolean;
  usesThreadIdxY: boolean;
  usesThreadIdxZ: boolean;
  usesBlockIdxX: boolean;
  usesBlockIdxY: boolean;
  usesBlockIdxZ: boolean;
  usesBlockDim: boolean;
  globalIdExpression?: string;
}

// Pattern Detection Types
export type KernelArchetype =
  | 'attention'       // Flash Attention, Multi-Head Attention
  | 'fused'           // Fused kernels (e.g., matmul + activation)
  | 'fft'             // Fast Fourier Transform
  | 'gemm'            // Matrix multiplication
  | 'reduction'       // Sum, max, min reductions
  | 'scan'            // Prefix sum/scan
  | 'stencil'         // Stencil computations
  | 'elementwise'     // Element-wise operations
  | 'histogram'       // Histogram computation
  | 'sparse'          // Sparse matrix operations
  | 'convolution'     // Convolution operations
  | 'sorting'         // Sorting algorithms
  | 'pooling'         // Pooling operations
  | 'normalization'   // Normalization layers
  | 'embedding'       // Embedding lookups
  | 'rope'            // Rotary Position Embedding
  | 'kvcache'         // KV Cache operations
  | 'quantization';   // Quantization/dequantization

// Pattern variant for more specific classification
export type PatternVariant =
  // Attention variants
  | 'flash_attention' | 'flash_attention_v2' | 'multi_head_attention' | 'causal_attention' | 'cross_attention'
  // Fused kernel variants
  | 'matmul_activation' | 'matmul_bias_activation' | 'conv_batchnorm' | 'layernorm_residual' | 'multi_phase_fused'
  // FFT variants
  | 'fft_radix2' | 'fft_radix4' | 'fft_radix8' | 'inverse_fft' | 'real_fft'
  // Reduction variants
  | 'tree_reduction' | 'warp_shuffle' | 'multi_block' | 'segmented'
  // GEMM variants
  | 'naive_gemm' | 'tiled_gemm' | 'register_blocked'
  // Scan variants
  | 'inclusive_scan' | 'exclusive_scan' | 'segmented_scan'
  // Stencil variants
  | 'stencil_1d_3pt' | 'stencil_1d_5pt' | 'stencil_2d_5pt' | 'stencil_2d_9pt' | 'stencil_3d'
  // Histogram variants
  | 'histogram_atomic' | 'histogram_privatized' | 'histogram_multipass' | 'histogram_weighted' | 'histogram_2d'
  // Sparse variants
  | 'spmv_csr' | 'spmv_csr_warp' | 'spmv_coo' | 'spmv_ell' | 'spmm_csr' | 'sddmm'
  // Elementwise variants
  | 'vectorized' | 'simple'
  // Convolution variants
  | 'conv_1d' | 'conv_2d' | 'conv_3d' | 'conv_depthwise' | 'conv_grouped' | 'conv_winograd' | 'conv_im2col' | 'conv_implicit_gemm'
  // Sorting variants
  | 'bitonic_sort' | 'bitonic_sort_shared' | 'radix_sort' | 'merge_sort'
  // Pooling variants
  | 'max_pool_2d' | 'avg_pool_2d' | 'global_avg_pool' | 'global_max_pool' | 'adaptive_avg_pool' | 'adaptive_max_pool'
  // Normalization variants
  | 'layernorm' | 'rmsnorm' | 'batchnorm' | 'groupnorm' | 'instancenorm'
  // Embedding variants
  | 'embedding_lookup' | 'embedding_bag' | 'positional_embedding'
  // RoPE variants
  | 'rope_standard' | 'rope_neox' | 'rope_cached'
  // KV Cache variants
  | 'kvcache_append' | 'kvcache_paged' | 'kvcache_prefix' | 'kvcache_gqa'
  // Quantization variants
  | 'quant_int8' | 'quant_int4' | 'quant_fp8' | 'quantize' | 'dequantize';

export interface PatternMatch {
  archetype: KernelArchetype;
  variant?: PatternVariant;
  confidence: number;
  evidence: Evidence[];
  warnings: string[];
}

export interface Evidence {
  type: string;
  weight: number;
  description: string;
  line?: number;
}

// IR Types
export interface KernelIR {
  name: string;
  originalName: string;
  archetype: KernelArchetype;
  confidence: number;
  parameters: IRParameter[];
  loads: IRLoad[];
  operations: IROperation[];
  stores: IRStore[];
  tileConfig: TileConfig;
}

export interface IRParameter {
  cudaName: string;
  cuTileName: string;
  type: string;
  isConstant: boolean;
  constantAnnotation?: string;
}

export interface IRLoad {
  source: string;
  target: string;
  index: string;
  shape: string;
}

export interface IROperation {
  type: 'elementwise' | 'matmul' | 'reduce' | 'atomic' | 'accumulate';
  inputs: string[];
  output: string;
  op?: string;
  axis?: number;
}

export interface IRStore {
  source: string;
  target: string;
  index: string;
}

export interface TileConfig {
  tileSize: number;
  blockM?: number;
  blockN?: number;
  blockK?: number;
}

// Validation Types
export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  adjustedConfidence: number;
}

// Transpile Result
export interface TranspileResult {
  tileCode: string;
  pattern: PatternMatch;
  ir: KernelIR;
  validation: ValidationResult;
}
