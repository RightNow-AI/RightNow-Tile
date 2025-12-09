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
  | 'attention'
  | 'fft'
  | 'fused'
  | 'reduction'
  | 'scan'
  | 'histogram'
  | 'gemm'
  | 'stencil'
  | 'elementwise'
  | 'sparse'
  | 'convolution'
  | 'sorting'
  | 'pooling'
  | 'normalization'
  | 'embedding'
  | 'rope'
  | 'kvcache'
  | 'quantization';

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
  // Fused multiply-add (critical for GEMM and attention)
  '__fmaf': {
    name: '__fmaf',
    category: 'math',
    patternHints: ['gemm', 'attention'],
    description: 'Fused multiply-add (single precision)',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__fma': {
    name: '__fma',
    category: 'math',
    patternHints: ['gemm', 'attention'],
    description: 'Fused multiply-add (double precision)',
    returnType: 'double',
    hasSideEffects: false,
  },
  'fmaf': {
    name: 'fmaf',
    category: 'math',
    patternHints: ['gemm', 'attention'],
    description: 'Fused multiply-add (standard)',
    returnType: 'float',
    hasSideEffects: false,
  },
  'fma': {
    name: 'fma',
    category: 'math',
    patternHints: ['gemm', 'attention'],
    description: 'Fused multiply-add (double, standard)',
    returnType: 'double',
    hasSideEffects: false,
  },

  // Exponential functions (critical for softmax in attention)
  '__expf': {
    name: '__expf',
    category: 'math',
    patternHints: ['attention', 'elementwise'],
    description: 'Fast exponential (single precision)',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__exp': {
    name: '__exp',
    category: 'math',
    patternHints: ['attention', 'elementwise'],
    description: 'Fast exponential (double precision)',
    returnType: 'double',
    hasSideEffects: false,
  },
  'expf': {
    name: 'expf',
    category: 'math',
    patternHints: ['attention', 'elementwise'],
    description: 'Standard exponential (single precision)',
    returnType: 'float',
    hasSideEffects: false,
  },
  'exp': {
    name: 'exp',
    category: 'math',
    patternHints: ['attention', 'elementwise'],
    description: 'Standard exponential (double precision)',
    returnType: 'double',
    hasSideEffects: false,
  },
  '__exp2f': {
    name: '__exp2f',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Fast base-2 exponential',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__exp10f': {
    name: '__exp10f',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Fast base-10 exponential',
    returnType: 'float',
    hasSideEffects: false,
  },

  // Logarithm functions
  '__logf': {
    name: '__logf',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Fast natural logarithm',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__log2f': {
    name: '__log2f',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Fast base-2 logarithm',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__log10f': {
    name: '__log10f',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Fast base-10 logarithm',
    returnType: 'float',
    hasSideEffects: false,
  },

  // Power and root functions
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
    patternHints: ['attention', 'elementwise'],
    description: 'Fast reciprocal square root',
    returnType: 'float',
    hasSideEffects: false,
  },
  'rsqrtf': {
    name: 'rsqrtf',
    category: 'math',
    patternHints: ['attention', 'elementwise'],
    description: 'Reciprocal square root',
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

  // Trigonometric functions (critical for FFT)
  '__sinf': {
    name: '__sinf',
    category: 'math',
    patternHints: ['fft', 'elementwise'],
    description: 'Fast sine',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__cosf': {
    name: '__cosf',
    category: 'math',
    patternHints: ['fft', 'elementwise'],
    description: 'Fast cosine',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__sincosf': {
    name: '__sincosf',
    category: 'math',
    patternHints: ['fft'],
    description: 'Fast simultaneous sine and cosine',
    returnType: 'void',
    hasSideEffects: false,
  },
  'sincosf': {
    name: 'sincosf',
    category: 'math',
    patternHints: ['fft'],
    description: 'Simultaneous sine and cosine',
    returnType: 'void',
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

  // Min/Max functions (critical for attention softmax)
  'fmaxf': {
    name: 'fmaxf',
    category: 'math',
    patternHints: ['attention', 'reduction'],
    description: 'Maximum of two floats',
    returnType: 'float',
    hasSideEffects: false,
  },
  'fminf': {
    name: 'fminf',
    category: 'math',
    patternHints: ['attention', 'reduction'],
    description: 'Minimum of two floats',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__fmaxf': {
    name: '__fmaxf',
    category: 'math',
    patternHints: ['attention', 'reduction'],
    description: 'Fast maximum of two floats',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__fminf': {
    name: '__fminf',
    category: 'math',
    patternHints: ['attention', 'reduction'],
    description: 'Fast minimum of two floats',
    returnType: 'float',
    hasSideEffects: false,
  },

  // Rounding functions
  '__float2int_rn': {
    name: '__float2int_rn',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Convert float to int (round to nearest)',
    returnType: 'int',
    hasSideEffects: false,
  },
  '__int2float_rn': {
    name: '__int2float_rn',
    category: 'math',
    patternHints: ['elementwise'],
    description: 'Convert int to float (round to nearest)',
    returnType: 'float',
    hasSideEffects: false,
  },

  // Saturating arithmetic (for attention scaling)
  '__saturatef': {
    name: '__saturatef',
    category: 'math',
    patternHints: ['attention', 'elementwise'],
    description: 'Saturate to [0, 1]',
    returnType: 'float',
    hasSideEffects: false,
  },

  // Division and modulo
  '__fdividef': {
    name: '__fdividef',
    category: 'math',
    patternHints: ['attention', 'elementwise'],
    description: 'Fast division',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__frcp_rn': {
    name: '__frcp_rn',
    category: 'math',
    patternHints: ['attention', 'elementwise'],
    description: 'Fast reciprocal (round to nearest)',
    returnType: 'float',
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
    patternHints: ['fft'],
    description: 'Count leading zeros',
    returnType: 'int',
    hasSideEffects: false,
  },
  '__clzll': {
    name: '__clzll',
    category: 'bit_manipulation',
    patternHints: ['fft'],
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
  // Bit reversal - critical for FFT
  '__brev': {
    name: '__brev',
    category: 'bit_manipulation',
    patternHints: ['fft', 'scan'],
    description: 'Bit reverse (32-bit)',
    returnType: 'unsigned int',
    hasSideEffects: false,
  },
  '__brevll': {
    name: '__brevll',
    category: 'bit_manipulation',
    patternHints: ['fft', 'scan'],
    description: 'Bit reverse (64-bit)',
    returnType: 'unsigned long long',
    hasSideEffects: false,
  },
  // Byte permutation
  '__byte_perm': {
    name: '__byte_perm',
    category: 'bit_manipulation',
    patternHints: ['fft'],
    description: 'Byte permutation',
    returnType: 'unsigned int',
    hasSideEffects: false,
  },
  // Funnel shift (for bit rotation)
  '__funnelshift_l': {
    name: '__funnelshift_l',
    category: 'bit_manipulation',
    patternHints: [],
    description: 'Funnel shift left (32-bit)',
    returnType: 'unsigned int',
    hasSideEffects: false,
  },
  '__funnelshift_lc': {
    name: '__funnelshift_lc',
    category: 'bit_manipulation',
    patternHints: [],
    description: 'Funnel shift left (clamped)',
    returnType: 'unsigned int',
    hasSideEffects: false,
  },
  '__funnelshift_r': {
    name: '__funnelshift_r',
    category: 'bit_manipulation',
    patternHints: [],
    description: 'Funnel shift right (32-bit)',
    returnType: 'unsigned int',
    hasSideEffects: false,
  },
  '__funnelshift_rc': {
    name: '__funnelshift_rc',
    category: 'bit_manipulation',
    patternHints: [],
    description: 'Funnel shift right (clamped)',
    returnType: 'unsigned int',
    hasSideEffects: false,
  },
};

// Type Conversion Intrinsics (critical for quantization)
export const TYPE_CONVERSION_INTRINSICS: Record<string, IntrinsicInfo> = {
  // Float to int conversions (quantization)
  '__float2int_rz': {
    name: '__float2int_rz',
    category: 'type_conversion',
    patternHints: ['quantization'],
    description: 'Convert float to int (round toward zero)',
    returnType: 'int',
    hasSideEffects: false,
  },
  '__float2int_ru': {
    name: '__float2int_ru',
    category: 'type_conversion',
    patternHints: ['quantization'],
    description: 'Convert float to int (round up)',
    returnType: 'int',
    hasSideEffects: false,
  },
  '__float2int_rd': {
    name: '__float2int_rd',
    category: 'type_conversion',
    patternHints: ['quantization'],
    description: 'Convert float to int (round down)',
    returnType: 'int',
    hasSideEffects: false,
  },
  // Float to unsigned conversions
  '__float2uint_rn': {
    name: '__float2uint_rn',
    category: 'type_conversion',
    patternHints: ['quantization', 'histogram'],
    description: 'Convert float to unsigned int (round to nearest)',
    returnType: 'unsigned int',
    hasSideEffects: false,
  },
  // Half precision conversions (for mixed precision/quantization)
  '__float2half': {
    name: '__float2half',
    category: 'type_conversion',
    patternHints: ['quantization', 'attention'],
    description: 'Convert float to half precision',
    returnType: '__half',
    hasSideEffects: false,
  },
  '__half2float': {
    name: '__half2float',
    category: 'type_conversion',
    patternHints: ['quantization', 'attention'],
    description: 'Convert half precision to float',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__float2half_rn': {
    name: '__float2half_rn',
    category: 'type_conversion',
    patternHints: ['quantization'],
    description: 'Convert float to half (round to nearest)',
    returnType: '__half',
    hasSideEffects: false,
  },
  // BFloat16 conversions
  '__float2bfloat16': {
    name: '__float2bfloat16',
    category: 'type_conversion',
    patternHints: ['quantization', 'attention'],
    description: 'Convert float to bfloat16',
    returnType: '__nv_bfloat16',
    hasSideEffects: false,
  },
  '__bfloat162float': {
    name: '__bfloat162float',
    category: 'type_conversion',
    patternHints: ['quantization', 'attention'],
    description: 'Convert bfloat16 to float',
    returnType: 'float',
    hasSideEffects: false,
  },
  // Packed half operations
  '__floats2half2_rn': {
    name: '__floats2half2_rn',
    category: 'type_conversion',
    patternHints: ['quantization'],
    description: 'Convert two floats to packed half2',
    returnType: '__half2',
    hasSideEffects: false,
  },
  '__half22float2': {
    name: '__half22float2',
    category: 'type_conversion',
    patternHints: ['quantization'],
    description: 'Convert packed half2 to float2',
    returnType: 'float2',
    hasSideEffects: false,
  },
};

// Additional Math Intrinsics for new patterns
export const EXTENDED_MATH_INTRINSICS: Record<string, IntrinsicInfo> = {
  // Hyperbolic functions (for GELU, tanh activations in fused kernels)
  '__tanhf': {
    name: '__tanhf',
    category: 'math',
    patternHints: ['fused', 'normalization'],
    description: 'Fast hyperbolic tangent',
    returnType: 'float',
    hasSideEffects: false,
  },
  'tanhf': {
    name: 'tanhf',
    category: 'math',
    patternHints: ['fused', 'normalization'],
    description: 'Hyperbolic tangent',
    returnType: 'float',
    hasSideEffects: false,
  },
  // Error function (for GELU activation)
  '__erff': {
    name: '__erff',
    category: 'math',
    patternHints: ['fused'],
    description: 'Fast error function',
    returnType: 'float',
    hasSideEffects: false,
  },
  'erff': {
    name: 'erff',
    category: 'math',
    patternHints: ['fused'],
    description: 'Error function',
    returnType: 'float',
    hasSideEffects: false,
  },
  // Rounding for quantization
  'roundf': {
    name: 'roundf',
    category: 'math',
    patternHints: ['quantization'],
    description: 'Round to nearest integer',
    returnType: 'float',
    hasSideEffects: false,
  },
  'truncf': {
    name: 'truncf',
    category: 'math',
    patternHints: ['quantization'],
    description: 'Truncate to integer',
    returnType: 'float',
    hasSideEffects: false,
  },
  'floorf': {
    name: 'floorf',
    category: 'math',
    patternHints: ['pooling', 'convolution'],
    description: 'Floor function',
    returnType: 'float',
    hasSideEffects: false,
  },
  'ceilf': {
    name: 'ceilf',
    category: 'math',
    patternHints: ['pooling', 'convolution'],
    description: 'Ceiling function',
    returnType: 'float',
    hasSideEffects: false,
  },
  // Absolute value (for normalization, pooling)
  'fabsf': {
    name: 'fabsf',
    category: 'math',
    patternHints: ['normalization', 'elementwise'],
    description: 'Absolute value (float)',
    returnType: 'float',
    hasSideEffects: false,
  },
  '__fabsf': {
    name: '__fabsf',
    category: 'math',
    patternHints: ['normalization', 'elementwise'],
    description: 'Fast absolute value',
    returnType: 'float',
    hasSideEffects: false,
  },
  // Remainder/modulo (for RoPE position calculations)
  'fmodf': {
    name: 'fmodf',
    category: 'math',
    patternHints: ['rope', 'elementwise'],
    description: 'Floating-point remainder',
    returnType: 'float',
    hasSideEffects: false,
  },
  // Atan2 (for rotary embeddings angle calculations)
  'atan2f': {
    name: 'atan2f',
    category: 'math',
    patternHints: ['rope', 'elementwise'],
    description: 'Two-argument arctangent',
    returnType: 'float',
    hasSideEffects: false,
  },
  // Copysign (for sorting comparisons, quantization)
  'copysignf': {
    name: 'copysignf',
    category: 'math',
    patternHints: ['sorting', 'quantization'],
    description: 'Copy sign of a number',
    returnType: 'float',
    hasSideEffects: false,
  },
};

// Tensor Core / Matrix Intrinsics (critical for GEMM, attention)
export const TENSOR_INTRINSICS: Record<string, IntrinsicInfo> = {
  'wmma::load_matrix_sync': {
    name: 'wmma::load_matrix_sync',
    category: 'memory',
    patternHints: ['gemm', 'attention', 'convolution'],
    description: 'Load matrix fragment for tensor core',
    returnType: 'void',
    hasSideEffects: true,
  },
  'wmma::store_matrix_sync': {
    name: 'wmma::store_matrix_sync',
    category: 'memory',
    patternHints: ['gemm', 'attention', 'convolution'],
    description: 'Store matrix fragment from tensor core',
    returnType: 'void',
    hasSideEffects: true,
  },
  'wmma::mma_sync': {
    name: 'wmma::mma_sync',
    category: 'math',
    patternHints: ['gemm', 'attention', 'convolution'],
    description: 'Tensor core matrix multiply-accumulate',
    returnType: 'void',
    hasSideEffects: true,
  },
  'wmma::fill_fragment': {
    name: 'wmma::fill_fragment',
    category: 'memory',
    patternHints: ['gemm', 'attention'],
    description: 'Fill matrix fragment with value',
    returnType: 'void',
    hasSideEffects: true,
  },
  // MMA PTX intrinsics (for more fine-grained control)
  '__mma_m16n8k16_row_col_f16': {
    name: '__mma_m16n8k16_row_col_f16',
    category: 'math',
    patternHints: ['gemm', 'attention'],
    description: 'MMA 16x8x16 half precision',
    returnType: 'void',
    hasSideEffects: true,
  },
  '__mma_m16n8k8_row_col_tf32': {
    name: '__mma_m16n8k8_row_col_tf32',
    category: 'math',
    patternHints: ['gemm', 'attention'],
    description: 'MMA 16x8x8 TensorFloat-32',
    returnType: 'void',
    hasSideEffects: true,
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
  ...TYPE_CONVERSION_INTRINSICS,
  ...EXTENDED_MATH_INTRINSICS,
  ...TENSOR_INTRINSICS,
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
