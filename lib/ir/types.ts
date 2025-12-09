/**
 * Enhanced IR Types
 * Extended intermediate representation for advanced transpilation
 */

import { KernelArchetype, PatternVariant, TileConfig } from '../ast/types';

/**
 * Extended Kernel IR with variant and optimization hints
 */
export interface EnhancedKernelIR {
  // Basic info
  name: string;
  originalName: string;
  archetype: KernelArchetype;
  variant?: PatternVariant;
  confidence: number;

  // Structure
  parameters: EnhancedIRParameter[];
  loads: EnhancedIRLoad[];
  operations: EnhancedIROperation[];
  stores: EnhancedIRStore[];

  // Configuration
  tileConfig: EnhancedTileConfig;
  tileStrategy: TileStrategy;

  // Semantic info
  semanticInfo: SemanticIRInfo;

  // Optimization
  optimizationHints: OptimizationHint[];
  memoryLayout: MemoryLayoutInfo;
}

/**
 * Enhanced parameter with type mapping
 */
export interface EnhancedIRParameter {
  cudaName: string;
  cuTileName: string;
  cudaType: string;
  cuTileType: string;
  isPointer: boolean;
  isConstant: boolean;
  constantAnnotation?: string;
  tensorShape?: string;
}

/**
 * Enhanced load operation with access patterns
 */
export interface EnhancedIRLoad {
  source: string;
  target: string;
  index: string;
  shape: string;
  dtype: string;
  accessPattern: 'coalesced' | 'strided' | 'broadcast' | 'random';
  cacheHint?: 'streaming' | 'persistent' | 'evict_first';
  mask?: string;
  prefetch?: boolean;
}

/**
 * Enhanced operation with more detail
 */
export interface EnhancedIROperation {
  type: 'elementwise' | 'matmul' | 'reduce' | 'scan' | 'atomic' | 'stencil' | 'histogram' | 'accumulate';
  inputs: string[];
  output: string;
  op?: string;
  axis?: number;
  dtype?: string;
  // For stencil
  stencilKernel?: number[][];
  // For reduction
  reductionOp?: 'sum' | 'max' | 'min' | 'prod' | 'and' | 'or';
  // For histogram
  numBins?: number;
}

/**
 * Enhanced store operation
 */
export interface EnhancedIRStore {
  source: string;
  target: string;
  index: string;
  accessPattern: 'coalesced' | 'strided' | 'random';
  mask?: string;
  atomic?: boolean;
}

/**
 * Enhanced tile configuration
 */
export interface EnhancedTileConfig extends TileConfig {
  warpsPerBlock?: number;
  elementsPerThread?: number;
  vectorWidth?: number;
  stages?: number; // Pipeline stages
}

/**
 * Tile strategy information
 */
export interface TileStrategy {
  approach: 'blocked' | 'streaming' | 'hierarchical' | 'cooperative';
  dimensions: TileDimension[];
  justification: string;
  estimatedOccupancy?: number;
}

export interface TileDimension {
  name: string;
  size: number;
  axis: number;
}

/**
 * Semantic information from analysis
 */
export interface SemanticIRInfo {
  reductionOp?: 'sum' | 'max' | 'min' | 'prod';
  dataTypes: Map<string, DataTypeInfo>;
  inputArrays: string[];
  outputArrays: string[];
  intermediates: string[];
  hasDataDependency: boolean;
  isThreadSafe: boolean;
}

export interface DataTypeInfo {
  cudaType: string;
  cuTileType: string;
  bytes: number;
  isFloatingPoint: boolean;
}

/**
 * Optimization hint for code generation
 */
export interface OptimizationHint {
  category: 'vectorize' | 'unroll' | 'prefetch' | 'shared_memory' | 'register_blocking';
  target: string;
  suggestion: string;
  priority: 'high' | 'medium' | 'low';
  expectedBenefit?: string;
}

/**
 * Memory layout information
 */
export interface MemoryLayoutInfo {
  totalSharedMemory: number;
  registersPerThread: number;
  globalMemoryReads: number;
  globalMemoryWrites: number;
  sharedMemoryBankConflictFree: boolean;
}

/**
 * CUDA to cuTile type mapping
 */
export const TYPE_MAP: Record<string, string> = {
  // Floating point
  'float': 'ct.float32',
  'double': 'ct.float64',
  'half': 'ct.float16',
  '__half': 'ct.float16',
  '__half2': 'ct.float16',
  '__nv_bfloat16': 'ct.bfloat16',

  // Integer types
  'int': 'ct.int32',
  'int8_t': 'ct.int8',
  'int16_t': 'ct.int16',
  'int32_t': 'ct.int32',
  'int64_t': 'ct.int64',
  'unsigned int': 'ct.uint32',
  'uint8_t': 'ct.uint8',
  'uint16_t': 'ct.uint16',
  'uint32_t': 'ct.uint32',
  'uint64_t': 'ct.uint64',
  'size_t': 'ct.int64',
  'long long': 'ct.int64',
  'unsigned long long': 'ct.uint64',

  // Boolean
  'bool': 'ct.bool',

  // Pointers (extract base type)
  'float*': 'ct.float32',
  'double*': 'ct.float64',
  'int*': 'ct.int32',
  'half*': 'ct.float16',
};

/**
 * Get cuTile type for a CUDA type
 */
export function mapCudaType(cudaType: string): string {
  const normalized = cudaType.trim().replace(/\s+/g, ' ');
  return TYPE_MAP[normalized] || TYPE_MAP[normalized.replace('*', '').trim()] || 'ct.float32';
}

/**
 * Get type byte size
 */
export function getTypeBytes(cudaType: string): number {
  const sizes: Record<string, number> = {
    'float': 4,
    'double': 8,
    'half': 2,
    '__half': 2,
    'int': 4,
    'int8_t': 1,
    'int16_t': 2,
    'int32_t': 4,
    'int64_t': 8,
    'unsigned int': 4,
    'uint8_t': 1,
    'uint16_t': 2,
    'uint32_t': 4,
    'uint64_t': 8,
    'size_t': 8,
    'bool': 1,
  };
  const normalized = cudaType.trim().replace('*', '').replace(/\s+/g, ' ');
  return sizes[normalized] || 4;
}
