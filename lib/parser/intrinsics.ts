/**
 * CUDA Intrinsics Database
 * Comprehensive catalog of CUDA intrinsic functions for pattern detection
 */

export interface IntrinsicInfo {
  name: string;
  category: IntrinsicCategory;
  patternHints: PatternHint[];
  description: string;
  returnType?: string;
  hasSideEffects: boolean;
}

export type IntrinsicCategory =
  | 'warp_shuffle'
  | 'warp_vote'
  | 'warp_reduce'
  | 'atomic'
  | 'math'
  | 'sync'
  | 'memory'
  | 'type_conversion'
  | 'bit_manipulation';

export type PatternHint =
  | 'reduction'
  | 'scan'
  | 'histogram'
  | 'gemm'
  | 'stencil'
  | 'elementwise'
  | 'sparse';

// Warp Shuffle Intrinsics - Strong indicators of reduction/scan
export const WARP_SHUFFLE_INTRINSICS: Record<string, IntrinsicInfo> = {
  '__shfl_down_sync': {
    name: '__shfl_down_sync',
    category: 'warp_shuffle',
    patternHints: ['reduction'],
    description: 'Warp shuffle down - each thread gets value from thread with higher lane ID',
    returnType: 'T',
    hasSideEffects: false,
  },
  '__shfl_up_sync': {
    name: '__shfl_up_sync',
    category: 'warp_shuffle',
    patternHints: ['scan'],
    description: 'Warp shuffle up - each thread gets value from thread with lower lane ID',
    returnType: 'T',
    hasSideEffects: false,
  },
  '__shfl_xor_sync': {
    name: '__shfl_xor_sync',
    category: 'warp_shuffle',
    patternHints: ['reduction'],
    description: 'Warp shuffle XOR - butterfly reduction pattern',
    returnType: 'T',
    hasSideEffects: false,
  },
  '__shfl_sync': {
    name: '__shfl_sync',
    category: 'warp_shuffle',
    patternHints: ['reduction', 'scan'],
    description: 'Warp shuffle - direct lane access',
    returnType: 'T',
    hasSideEffects: false,
  },
  // Legacy intrinsics (pre-Volta)
  '__shfl_down': {
    name: '__shfl_down',
    category: 'warp_shuffle',
    patternHints: ['reduction'],
    description: 'Legacy warp shuffle down',
    returnType: 'T',
    hasSideEffects: false,
  },
  '__shfl_up': {
    name: '__shfl_up',
    category: 'warp_shuffle',
    patternHints: ['scan'],
    description: 'Legacy warp shuffle up',
    returnType: 'T',
    hasSideEffects: false,
  },
  '__shfl_xor': {
    name: '__shfl_xor',
    category: 'warp_shuffle',
    patternHints: ['reduction'],
    description: 'Legacy warp shuffle XOR',
    returnType: 'T',
    hasSideEffects: false,
  },
  '__shfl': {
    name: '__shfl',
    category: 'warp_shuffle',
    patternHints: ['reduction', 'scan'],
    description: 'Legacy warp shuffle',
    returnType: 'T',
    hasSideEffects: false,
  },
};

// Warp Vote Intrinsics
export const WARP_VOTE_INTRINSICS: Record<string, IntrinsicInfo> = {
  '__ballot_sync': {
    name: '__ballot_sync',
    category: 'warp_vote',
    patternHints: ['reduction'],
    description: 'Returns bitmask of threads where predicate is true',
    returnType: 'unsigned int',
    hasSideEffects: false,
  },
  '__all_sync': {
    name: '__all_sync',
    category: 'warp_vote',
    patternHints: ['reduction'],
    description: 'Returns true if predicate is true for all threads',
    returnType: 'int',
    hasSideEffects: false,
  },
  '__any_sync': {
    name: '__any_sync',
    category: 'warp_vote',
    patternHints: ['reduction'],
    description: 'Returns true if predicate is true for any thread',
    returnType: 'int',
    hasSideEffects: false,
  },
  '__activemask': {
    name: '__activemask',
    category: 'warp_vote',
    patternHints: [],
    description: 'Returns mask of active threads in warp',
    returnType: 'unsigned int',
    hasSideEffects: false,
  },
};

// Warp Reduce Intrinsics (SM 8.0+)
export const WARP_REDUCE_INTRINSICS: Record<string, IntrinsicInfo> = {
  '__reduce_add_sync': {
    name: '__reduce_add_sync',
    category: 'warp_reduce',
    patternHints: ['reduction'],
    description: 'Warp-level sum reduction',
    returnType: 'T',
    hasSideEffects: false,
  },
  '__reduce_max_sync': {
    name: '__reduce_max_sync',
    category: 'warp_reduce',
    patternHints: ['reduction'],
    description: 'Warp-level max reduction',
    returnType: 'T',
    hasSideEffects: false,
  },
  '__reduce_min_sync': {
    name: '__reduce_min_sync',
    category: 'warp_reduce',
    patternHints: ['reduction'],
    description: 'Warp-level min reduction',
    returnType: 'T',
    hasSideEffects: false,
  },
  '__reduce_and_sync': {
    name: '__reduce_and_sync',
    category: 'warp_reduce',
    patternHints: ['reduction'],
    description: 'Warp-level bitwise AND reduction',
    returnType: 'unsigned int',
    hasSideEffects: false,
  },
  '__reduce_or_sync': {
    name: '__reduce_or_sync',
    category: 'warp_reduce',
    patternHints: ['reduction'],
    description: 'Warp-level bitwise OR reduction',
    returnType: 'unsigned int',
    hasSideEffects: false,
  },
  '__reduce_xor_sync': {
    name: '__reduce_xor_sync',
    category: 'warp_reduce',
    patternHints: ['reduction'],
    description: 'Warp-level bitwise XOR reduction',
    returnType: 'unsigned int',
    hasSideEffects: false,
  },
};

// Atomic Intrinsics
export const ATOMIC_INTRINSICS: Record<string, IntrinsicInfo> = {
  'atomicAdd': {
    name: 'atomicAdd',
    category: 'atomic',
    patternHints: ['reduction', 'histogram'],
    description: 'Atomic addition',
    returnType: 'T',
    hasSideEffects: true,
  },
  'atomicSub': {
    name: 'atomicSub',
    category: 'atomic',
    patternHints: ['reduction'],
    description: 'Atomic subtraction',
    returnType: 'T',
    hasSideEffects: true,
  },
  'atomicMax': {
    name: 'atomicMax',
    category: 'atomic',
    patternHints: ['reduction'],
    description: 'Atomic maximum',
    returnType: 'T',
    hasSideEffects: true,
  },
  'atomicMin': {
    name: 'atomicMin',
    category: 'atomic',
    patternHints: ['reduction'],
    description: 'Atomic minimum',
    returnType: 'T',
    hasSideEffects: true,
  },
  'atomicInc': {
    name: 'atomicInc',
    category: 'atomic',
    patternHints: ['histogram'],
    description: 'Atomic increment (wrapping)',
    returnType: 'unsigned int',
    hasSideEffects: true,
  },
  'atomicDec': {
    name: 'atomicDec',
    category: 'atomic',
    patternHints: ['histogram'],
    description: 'Atomic decrement (wrapping)',
    returnType: 'unsigned int',
    hasSideEffects: true,
  },
  'atomicExch': {
    name: 'atomicExch',
    category: 'atomic',
    patternHints: [],
    description: 'Atomic exchange',
    returnType: 'T',
    hasSideEffects: true,
  },
  'atomicCAS': {
    name: 'atomicCAS',
    category: 'atomic',
    patternHints: [],
    description: 'Atomic compare-and-swap',
    returnType: 'T',
    hasSideEffects: true,
  },
  'atomicAnd': {
    name: 'atomicAnd',
    category: 'atomic',
    patternHints: ['reduction'],
    description: 'Atomic bitwise AND',
    returnType: 'T',
    hasSideEffects: true,
  },
  'atomicOr': {
    name: 'atomicOr',
    category: 'atomic',
    patternHints: ['reduction'],
    description: 'Atomic bitwise OR',
    returnType: 'T',
    hasSideEffects: true,
  },
  'atomicXor': {
    name: 'atomicXor',
    category: 'atomic',
    patternHints: ['reduction'],
    description: 'Atomic bitwise XOR',
    returnType: 'T',
    hasSideEffects: true,
  },
};

// Synchronization Intrinsics
export const SYNC_INTRINSICS: Record<string, IntrinsicInfo> = {
  '__syncthreads': {
    name: '__syncthreads',
    category: 'sync',
    patternHints: ['reduction', 'scan', 'gemm', 'stencil'],
    description: 'Block-level barrier synchronization',
    returnType: 'void',
    hasSideEffects: true,
  },
  '__syncthreads_count': {
    name: '__syncthreads_count',
    category: 'sync',
    patternHints: ['reduction'],
    description: 'Barrier with predicate count',
    returnType: 'int',
    hasSideEffects: true,
  },
  '__syncthreads_and': {
    name: '__syncthreads_and',
    category: 'sync',
    patternHints: ['reduction'],
    description: 'Barrier with AND reduction',
    returnType: 'int',
    hasSideEffects: true,
  },
  '__syncthreads_or': {
    name: '__syncthreads_or',
    category: 'sync',
    patternHints: ['reduction'],
    description: 'Barrier with OR reduction',
    returnType: 'int',
    hasSideEffects: true,
  },
  '__syncwarp': {
    name: '__syncwarp',
    category: 'sync',
    patternHints: ['reduction'],
    description: 'Warp-level barrier',
    returnType: 'void',
    hasSideEffects: true,
  },
  '__threadfence': {
    name: '__threadfence',
    category: 'sync',
    patternHints: ['reduction'],
    description: 'Memory fence (device-wide)',
    returnType: 'void',
    hasSideEffects: true,
  },
  '__threadfence_block': {
    name: '__threadfence_block',
    category: 'sync',
    patternHints: ['reduction', 'scan'],
    description: 'Memory fence (block-wide)',
    returnType: 'void',
    hasSideEffects: true,
  },
  '__threadfence_system': {
    name: '__threadfence_system',
    category: 'sync',
    patternHints: [],
    description: 'Memory fence (system-wide)',
    returnType: 'void',
    hasSideEffects: true,
  },
};

// Math Intrinsics
export const MATH_INTRINSICS: Record<string, IntrinsicInfo> = {
  '__fmaf': {
    name: '__fmaf',
    category: 'math',
    patternHints: ['gemm'],
    description: 'Fused multiply-add (single precision)',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__fma': {
    name: '__fma',
    category: 'math',
    patternHints: ['gemm'],
    description: 'Fused multiply-add (double precision)',
    returnType: 'double',
    hasSideEffects: false,
  },
  '__expf': {
    name: '__expf',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Fast exponential',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__logf': {
    name: '__logf',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Fast logarithm',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__powf': {
    name: '__powf',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Fast power',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__rsqrtf': {
    name: '__rsqrtf',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Fast reciprocal square root',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__sqrtf': {
    name: '__sqrtf',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Fast square root',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__sinf': {
    name: '__sinf',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Fast sine',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__cosf': {
    name: '__cosf',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Fast cosine',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__tanf': {
    name: '__tanf',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Fast tangent',
    returnType: 'float',
    hasSideEffects: false,
  },
  'fmaf': {
    name: 'fmaf',
    category: 'math',
    patternHints: ['gemm'],
    description: 'Fused multiply-add (standard)',
    returnType: 'float',
    hasSideEffects: false,
  },
  'fma': {
    name: 'fma',
    category: 'math',
    patternHints: ['gemm'],
    description: 'Fused multiply-add (double, standard)',
    returnType: 'double',
    hasSideEffects: false,
  },
};

// Memory Intrinsics
export const MEMORY_INTRINSICS: Record<string, IntrinsicInfo> = {
  '__ldg': {
    name: '__ldg',
    category: 'memory',
    patternHints: [],
    description: 'Load through texture cache',
    returnType: 'T',
    hasSideEffects: false,
  },
  '__ldcs': {
    name: '__ldcs',
    category: 'memory',
    patternHints: ['stencil'],
    description: 'Load with cache streaming hint',
    returnType: 'T',
    hasSideEffects: false,
  },
  '__ldca': {
    name: '__ldca',
    category: 'memory',
    patternHints: [],
    description: 'Load with cache all hint',
    returnType: 'T',
    hasSideEffects: false,
  },
  '__ldcg': {
    name: '__ldcg',
    category: 'memory',
    patternHints: [],
    description: 'Load with cache global hint',
    returnType: 'T',
    hasSideEffects: false,
  },
  '__stcs': {
    name: '__stcs',
    category: 'memory',
    patternHints: ['stencil'],
    description: 'Store with streaming hint',
    returnType: 'void',
    hasSideEffects: true,
  },
  '__stcg': {
    name: '__stcg',
    category: 'memory',
    patternHints: [],
    description: 'Store with global hint',
    returnType: 'void',
    hasSideEffects: true,
  },
};

// Bit Manipulation Intrinsics
export const BIT_INTRINSICS: Record<string, IntrinsicInfo> = {
  '__popc': {
    name: '__popc',
    category: 'bit_manipulation',
    patternHints: ['reduction'],
    description: 'Population count (count set bits)',
    returnType: 'int',
    hasSideEffects: false,
  },
  '__popcll': {
    name: '__popcll',
    category: 'bit_manipulation',
    patternHints: ['reduction'],
    description: 'Population count (64-bit)',
    returnType: 'int',
    hasSideEffects: false,
  },
  '__clz': {
    name: '__clz',
    category: 'bit_manipulation',
    patternHints: [],
    description: 'Count leading zeros',
    returnType: 'int',
    hasSideEffects: false,
  },
  '__clzll': {
    name: '__clzll',
    category: 'bit_manipulation',
    patternHints: [],
    description: 'Count leading zeros (64-bit)',
    returnType: 'int',
    hasSideEffects: false,
  },
  '__ffs': {
    name: '__ffs',
    category: 'bit_manipulation',
    patternHints: [],
    description: 'Find first set bit',
    returnType: 'int',
    hasSideEffects: false,
  },
  '__ffsll': {
    name: '__ffsll',
    category: 'bit_manipulation',
    patternHints: [],
    description: 'Find first set bit (64-bit)',
    returnType: 'int',
    hasSideEffects: false,
  },
  '__brev': {
    name: '__brev',
    category: 'bit_manipulation',
    patternHints: ['scan'],
    description: 'Bit reverse',
    returnType: 'unsigned int',
    hasSideEffects: false,
  },
  '__brevll': {
    name: '__brevll',
    category: 'bit_manipulation',
    patternHints: ['scan'],
    description: 'Bit reverse (64-bit)',
    returnType: 'unsigned long long',
    hasSideEffects: false,
  },
};

// Combined intrinsics map
export const ALL_INTRINSICS: Record<string, IntrinsicInfo> = {
  ...WARP_SHUFFLE_INTRINSICS,
  ...WARP_VOTE_INTRINSICS,
  ...WARP_REDUCE_INTRINSICS,
  ...ATOMIC_INTRINSICS,
  ...SYNC_INTRINSICS,
  ...MATH_INTRINSICS,
  ...MEMORY_INTRINSICS,
  ...BIT_INTRINSICS,
};

// Helper function to get intrinsic info
export function getIntrinsicInfo(name: string): IntrinsicInfo | undefined {
  return ALL_INTRINSICS[name];
}

// Helper function to check if a name is a CUDA intrinsic
export function isIntrinsic(name: string): boolean {
  return name in ALL_INTRINSICS;
}

// Helper function to get pattern hints for an intrinsic
export function getPatternHints(name: string): PatternHint[] {
  const info = ALL_INTRINSICS[name];
  return info?.patternHints || [];
}

// Helper function to get all intrinsics by category
export function getIntrinsicsByCategory(category: IntrinsicCategory): IntrinsicInfo[] {
  return Object.values(ALL_INTRINSICS).filter(i => i.category === category);
}

// Helper function to get all intrinsics that hint at a pattern
export function getIntrinsicsForPattern(pattern: PatternHint): IntrinsicInfo[] {
  return Object.values(ALL_INTRINSICS).filter(i => i.patternHints.includes(pattern));
}

// Build regex to match any intrinsic call
export function buildIntrinsicsRegex(): RegExp {
  const names = Object.keys(ALL_INTRINSICS).map(n => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  return new RegExp(`\\b(${names.join('|')})\\s*\\(`, 'g');
}
