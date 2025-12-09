/**
 * Parser Module Exports
 * Enhanced CUDA parsing with intrinsics detection and semantic analysis
 */

// Intrinsics database - types
export type { IntrinsicInfo, IntrinsicCategory, PatternHint } from './intrinsics';

// Intrinsics database - values
export {
  WARP_SHUFFLE_INTRINSICS,
  WARP_VOTE_INTRINSICS,
  WARP_REDUCE_INTRINSICS,
  ATOMIC_INTRINSICS,
  SYNC_INTRINSICS,
  MATH_INTRINSICS,
  MEMORY_INTRINSICS,
  BIT_INTRINSICS,
  ALL_INTRINSICS,
  getIntrinsicInfo,
  isIntrinsic,
  getPatternHints,
  getIntrinsicsByCategory,
  getIntrinsicsForPattern,
  buildIntrinsicsRegex,
} from './intrinsics';

// Enhanced parser types
export type {
  IntrinsicCall,
  IndexPattern,
  WarpOperation,
  LoopAnalysis,
  EnhancedParseResult,
  ReductionPattern,
  AccumulationPattern,
  ThreadIndexUsage,
  SharedMemoryUsage,
  SharedMemoryDecl,
  SharedMemoryAccess,
  PatternSignal,
} from './cuda-parser';

// Parser class and instance
export { EnhancedCudaParser, cudaParser } from './cuda-parser';
