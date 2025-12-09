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
export type KernelArchetype = 'gemm' | 'reduction' | 'scan' | 'stencil' | 'elementwise' | 'histogram' | 'sparse';

// Pattern variant for more specific classification
export type PatternVariant =
  | 'tree_reduction' | 'warp_shuffle' | 'multi_block' | 'segmented' // reduction variants
  | 'naive_gemm' | 'tiled_gemm' | 'register_blocked' // GEMM variants
  | 'inclusive_scan' | 'exclusive_scan' | 'segmented_scan' // scan variants
  | 'stencil_1d_3pt' | 'stencil_1d_5pt' | 'stencil_2d_5pt' | 'stencil_2d_9pt' | 'stencil_3d' // stencil variants
  | 'histogram_atomic' | 'histogram_privatized' // histogram variants
  | 'spmv_csr' | 'spmv_coo' | 'spmv_ell' // sparse variants
  | 'vectorized' | 'simple'; // elementwise variants

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
