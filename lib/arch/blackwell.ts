/**
 * Blackwell Architecture (SM100) Specific Optimizations
 * Supports NVIDIA Blackwell (GB100, GB102, GB104) architecture features
 */

export interface BlackwellConfig {
  // SM100 architecture parameters
  smVersion: 100;
  maxSharedMemory: number;         // Up to 228KB per SM
  maxRegistersPerThread: number;   // 255
  maxThreadsPerBlock: number;      // 1024
  maxThreadsPerSM: number;         // 2048
  maxBlocksPerSM: number;          // 32
  warpSize: number;                // 32

  // Tensor Memory Accelerator (TMA)
  tmaSupport: boolean;
  tmaMaxBytesPerCopy: number;      // 16MB
  tmaAsyncCopySupport: boolean;

  // WGMMA (Warpgroup Matrix Multiply-Accumulate)
  wgmmaSupport: boolean;
  wgmmaM: number;                  // 64 for Blackwell
  wgmmaN: number;                  // 64, 128, 256
  wgmmaK: number;                  // 16 for FP16, 8 for TF32

  // Cluster support
  clusterSupport: boolean;
  maxClustersPerGPC: number;       // 2
  maxBlocksPerCluster: number;     // 16

  // FP8 support
  fp8Support: boolean;
  fp8Types: ('e4m3' | 'e5m2')[];

  // Hardware barriers
  namedBarrierCount: number;       // 16
  arriveWaitBarriers: boolean;
}

/**
 * Default Blackwell GB100 configuration
 */
export const BLACKWELL_GB100_CONFIG: BlackwellConfig = {
  smVersion: 100,
  maxSharedMemory: 228 * 1024,     // 228KB
  maxRegistersPerThread: 255,
  maxThreadsPerBlock: 1024,
  maxThreadsPerSM: 2048,
  maxBlocksPerSM: 32,
  warpSize: 32,

  // TMA features
  tmaSupport: true,
  tmaMaxBytesPerCopy: 16 * 1024 * 1024,
  tmaAsyncCopySupport: true,

  // WGMMA features
  wgmmaSupport: true,
  wgmmaM: 64,
  wgmmaN: 256,
  wgmmaK: 16,

  // Cluster features
  clusterSupport: true,
  maxClustersPerGPC: 2,
  maxBlocksPerCluster: 16,

  // FP8 support
  fp8Support: true,
  fp8Types: ['e4m3', 'e5m2'],

  // Barriers
  namedBarrierCount: 16,
  arriveWaitBarriers: true,
};

/**
 * Blackwell tile configuration optimized for different archetypes
 */
export interface BlackwellTileConfig {
  blockM: number;
  blockN: number;
  blockK: number;
  stages: number;          // Pipeline stages for async copy
  warpsPerBlockM: number;
  warpsPerBlockN: number;
  useTMA: boolean;
  useWGMMA: boolean;
  useCluster: boolean;
  clusterDimX: number;
  clusterDimY: number;
  smemSwizzle: boolean;    // Bank conflict avoidance
  persistentKernel: boolean;
}

/**
 * Get optimized tile config for GEMM on Blackwell
 */
export function getBlackwellGEMMConfig(
  M: number,
  N: number,
  K: number,
  dtype: 'fp16' | 'bf16' | 'fp8' | 'tf32' | 'fp32'
): BlackwellTileConfig {
  // Optimal configurations for different sizes
  const isLargeProblem = M >= 4096 && N >= 4096;
  const isMediumProblem = M >= 1024 && N >= 1024;

  if (isLargeProblem && (dtype === 'fp16' || dtype === 'bf16' || dtype === 'fp8')) {
    // Large problem with tensor core types - use full cluster
    return {
      blockM: 128,
      blockN: 256,
      blockK: 64,
      stages: 4,
      warpsPerBlockM: 4,
      warpsPerBlockN: 1,
      useTMA: true,
      useWGMMA: true,
      useCluster: true,
      clusterDimX: 2,
      clusterDimY: 2,
      smemSwizzle: true,
      persistentKernel: true,
    };
  } else if (isMediumProblem) {
    // Medium problem
    return {
      blockM: 128,
      blockN: 128,
      blockK: 32,
      stages: 3,
      warpsPerBlockM: 4,
      warpsPerBlockN: 1,
      useTMA: true,
      useWGMMA: dtype !== 'fp32',
      useCluster: false,
      clusterDimX: 1,
      clusterDimY: 1,
      smemSwizzle: true,
      persistentKernel: false,
    };
  } else {
    // Small problem - simpler config
    return {
      blockM: 64,
      blockN: 64,
      blockK: 32,
      stages: 2,
      warpsPerBlockM: 2,
      warpsPerBlockN: 1,
      useTMA: false,
      useWGMMA: false,
      useCluster: false,
      clusterDimX: 1,
      clusterDimY: 1,
      smemSwizzle: false,
      persistentKernel: false,
    };
  }
}

/**
 * Get optimized tile config for Flash Attention on Blackwell
 */
export function getBlackwellAttentionConfig(
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  useCausal: boolean
): BlackwellTileConfig {
  const isLargeContext = seqLen >= 8192;
  const isMediumContext = seqLen >= 2048;

  if (isLargeContext) {
    // Long context attention - prioritize memory efficiency
    return {
      blockM: 128,
      blockN: 64,
      blockK: headDim,
      stages: 3,
      warpsPerBlockM: 4,
      warpsPerBlockN: 1,
      useTMA: true,
      useWGMMA: true,
      useCluster: true,
      clusterDimX: 2,
      clusterDimY: 1,
      smemSwizzle: true,
      persistentKernel: true,
    };
  } else if (isMediumContext) {
    return {
      blockM: 64,
      blockN: 64,
      blockK: headDim,
      stages: 2,
      warpsPerBlockM: 2,
      warpsPerBlockN: 1,
      useTMA: true,
      useWGMMA: true,
      useCluster: false,
      clusterDimX: 1,
      clusterDimY: 1,
      smemSwizzle: true,
      persistentKernel: false,
    };
  } else {
    return {
      blockM: 32,
      blockN: 32,
      blockK: headDim,
      stages: 2,
      warpsPerBlockM: 2,
      warpsPerBlockN: 1,
      useTMA: false,
      useWGMMA: false,
      useCluster: false,
      clusterDimX: 1,
      clusterDimY: 1,
      smemSwizzle: false,
      persistentKernel: false,
    };
  }
}

/**
 * Get optimized tile config for convolution on Blackwell
 */
export function getBlackwellConvConfig(
  batchSize: number,
  inChannels: number,
  outChannels: number,
  kernelSize: number,
  imageSize: number
): BlackwellTileConfig {
  const isLargeConv = outChannels >= 512 && inChannels >= 256;

  if (isLargeConv) {
    // Large convolution - use implicit GEMM
    return {
      blockM: 128,
      blockN: 128,
      blockK: 32,
      stages: 4,
      warpsPerBlockM: 4,
      warpsPerBlockN: 1,
      useTMA: true,
      useWGMMA: true,
      useCluster: true,
      clusterDimX: 2,
      clusterDimY: 1,
      smemSwizzle: true,
      persistentKernel: false,
    };
  } else {
    return {
      blockM: 64,
      blockN: 64,
      blockK: 32,
      stages: 2,
      warpsPerBlockM: 2,
      warpsPerBlockN: 1,
      useTMA: false,
      useWGMMA: false,
      useCluster: false,
      clusterDimX: 1,
      clusterDimY: 1,
      smemSwizzle: false,
      persistentKernel: false,
    };
  }
}

/**
 * Blackwell-specific intrinsics and their cuTile equivalents
 */
export const BLACKWELL_INTRINSICS_MAP: Record<string, string> = {
  // TMA intrinsics
  '__nv_tma_load_1d': 'ct.tma_load_1d',
  '__nv_tma_load_2d': 'ct.tma_load_2d',
  '__nv_tma_load_3d': 'ct.tma_load_3d',
  '__nv_tma_store_1d': 'ct.tma_store_1d',
  '__nv_tma_store_2d': 'ct.tma_store_2d',
  '__nv_tma_store_3d': 'ct.tma_store_3d',
  'cp_async_bulk_tensor_1d_global_to_shared': 'ct.tma_load_1d_async',
  'cp_async_bulk_tensor_2d_global_to_shared': 'ct.tma_load_2d_async',

  // WGMMA intrinsics
  '__nv_wgmma_m64n128k16_f16': 'ct.wgmma_f16_m64n128k16',
  '__nv_wgmma_m64n256k16_f16': 'ct.wgmma_f16_m64n256k16',
  '__nv_wgmma_m64n64k16_f16': 'ct.wgmma_f16_m64n64k16',
  '__nv_wgmma_m64n128k8_tf32': 'ct.wgmma_tf32_m64n128k8',
  '__nv_wgmma_m64n64k32_fp8': 'ct.wgmma_fp8_m64n64k32',

  // Cluster intrinsics
  '__cluster_dims': 'ct.cluster_dims',
  '__cluster_idx': 'ct.cluster_idx',
  '__cluster_block_idx': 'ct.cluster_block_idx',
  '__cluster_sync': 'ct.cluster_sync',
  '__cluster_arrive': 'ct.cluster_arrive',
  '__cluster_wait': 'ct.cluster_wait',

  // Barrier intrinsics
  '__namedbarrier_init': 'ct.named_barrier_init',
  '__namedbarrier_arrive': 'ct.named_barrier_arrive',
  '__namedbarrier_wait': 'ct.named_barrier_wait',
  '__namedbarrier_arrive_and_wait': 'ct.named_barrier_arrive_wait',

  // Async copy
  '__pipeline_memcpy_async': 'ct.memcpy_async',
  '__pipeline_commit': 'ct.pipeline_commit',
  '__pipeline_wait_prior': 'ct.pipeline_wait',
  'cp_async_wait_group': 'ct.async_wait_group',
  'cp_async_wait_all': 'ct.async_wait_all',

  // FP8 conversions
  '__nv_cvt_fp8_to_halfraw': 'ct.fp8_to_fp16',
  '__nv_cvt_halfraw_to_fp8': 'ct.fp16_to_fp8',
  '__nv_cvt_fp8x4_to_half2': 'ct.fp8x4_to_fp16x2',

  // Swizzle for bank conflict avoidance
  '__nv_swizzle_mode_t::SWIZZLE_128B': 'ct.swizzle_128b',
  '__nv_swizzle_mode_t::SWIZZLE_64B': 'ct.swizzle_64b',
  '__nv_swizzle_mode_t::SWIZZLE_32B': 'ct.swizzle_32b',
};

/**
 * Generate shared memory layout with swizzling for bank conflict avoidance
 */
export function generateSwizzledSmemLayout(
  tileM: number,
  tileN: number,
  dtype: 'fp16' | 'bf16' | 'fp8' | 'fp32'
): string {
  const bytesPerElement = dtype === 'fp32' ? 4 : dtype === 'fp8' ? 1 : 2;
  const swizzleBits = bytesPerElement === 1 ? 5 : bytesPerElement === 2 ? 4 : 3;

  return `
    # Swizzled shared memory layout for bank conflict avoidance
    SMEM_SWIZZLE_BITS = ${swizzleBits}
    SMEM_TILE_M = ${tileM}
    SMEM_TILE_N = ${tileN}
    SMEM_STRIDE = ${tileN} + (1 << SMEM_SWIZZLE_BITS)  # Padding for swizzle

    def swizzle_idx(row: int, col: int) -> int:
        \"\"\"Compute swizzled index for bank conflict-free access\"\"\"
        base = row * SMEM_STRIDE + col
        swizzle = (row >> SMEM_SWIZZLE_BITS) ^ (col >> SMEM_SWIZZLE_BITS)
        return base ^ (swizzle << SMEM_SWIZZLE_BITS)
`;
}

/**
 * Generate TMA descriptor initialization code
 */
export function generateTMADescriptor(
  arrayName: string,
  shape: number[],
  dtype: 'fp16' | 'bf16' | 'fp8' | 'fp32'
): string {
  const dims = shape.length;
  const cuTileType = dtype === 'fp32' ? 'ct.float32' :
                     dtype === 'fp16' ? 'ct.float16' :
                     dtype === 'bf16' ? 'ct.bfloat16' : 'ct.float8';

  return `
    # TMA descriptor for ${arrayName}
    ${arrayName}_desc = ct.tma_descriptor(
        ptr=${arrayName},
        shape=(${shape.join(', ')}),
        dtype=${cuTileType},
        swizzle=ct.swizzle_128b
    )
`;
}

/**
 * Generate WGMMA (Warpgroup Matrix Multiply-Accumulate) code
 */
export function generateWGMMACode(
  aName: string,
  bName: string,
  cName: string,
  M: number,
  N: number,
  K: number,
  dtype: 'fp16' | 'bf16' | 'fp8' | 'tf32'
): string {
  // WGMMA operates on warpgroup (4 warps = 128 threads)
  const wgmmaM = 64; // Fixed for Blackwell
  const wgmmaN = N <= 64 ? 64 : N <= 128 ? 128 : 256;
  const wgmmaK = dtype === 'tf32' ? 8 : dtype === 'fp8' ? 32 : 16;

  return `
    # WGMMA Matrix Multiply-Accumulate (Blackwell SM100)
    # Uses warpgroup (4 warps = 128 threads) for M${wgmmaM}xN${wgmmaN}xK${wgmmaK}

    # Initialize accumulator in registers
    ${cName}_frag = ct.zeros((${wgmmaM}, ${wgmmaN}), dtype=ct.float32)

    # Load A and B fragments
    ${aName}_frag = ct.wgmma_load_a(${aName}, shape=(${wgmmaM}, ${wgmmaK}))
    ${bName}_frag = ct.wgmma_load_b(${bName}, shape=(${wgmmaK}, ${wgmmaN}))

    # Execute WGMMA
    ${cName}_frag = ct.wgmma(
        ${aName}_frag,
        ${bName}_frag,
        ${cName}_frag,
        layout_a=ct.row_major,
        layout_b=ct.col_major
    )
`;
}

/**
 * Generate async copy with TMA
 */
export function generateTMACopy(
  srcName: string,
  dstName: string,
  shape: number[],
  isGlobalToShared: boolean
): string {
  const dims = shape.length;
  const loadFn = dims === 1 ? 'ct.tma_load_1d_async' :
                 dims === 2 ? 'ct.tma_load_2d_async' : 'ct.tma_load_3d_async';
  const storeFn = dims === 1 ? 'ct.tma_store_1d_async' :
                  dims === 2 ? 'ct.tma_store_2d_async' : 'ct.tma_store_3d_async';

  if (isGlobalToShared) {
    return `
    # Async TMA copy: global -> shared
    ${loadFn}(
        dst=${dstName},
        src_desc=${srcName}_desc,
        coord=(${shape.map((_, i) => `offset_${i}`).join(', ')}),
        barrier=mbar
    )
`;
  } else {
    return `
    # Async TMA copy: shared -> global
    ${storeFn}(
        src=${srcName},
        dst_desc=${dstName}_desc,
        coord=(${shape.map((_, i) => `offset_${i}`).join(', ')})
    )
`;
  }
}

/**
 * Generate cluster synchronization code
 */
export function generateClusterSync(): string {
  return `
    # Cluster-wide synchronization
    ct.cluster_sync()
`;
}

/**
 * Generate pipeline stages for software pipelining
 */
export function generatePipelineStages(
  stages: number,
  loadCode: string,
  computeCode: string
): string {
  return `
    # Software pipelining with ${stages} stages
    NUM_STAGES = ${stages}

    # Initialize pipeline barriers
    mbar = ct.create_mbarrier_array(NUM_STAGES)

    # Prologue: fill pipeline
    for stage in range(NUM_STAGES - 1):
        ct.mbarrier_arrive(mbar[stage])
${loadCode.split('\n').map(line => '        ' + line).join('\n')}
        ct.pipeline_commit()

    # Main loop: steady state
    for k_block in range(num_k_blocks):
        stage = k_block % NUM_STAGES

        # Wait for data to be ready
        ct.mbarrier_wait(mbar[stage])

        # Compute on current stage
${computeCode.split('\n').map(line => '        ' + line).join('\n')}

        # Issue next load
        next_stage = (stage + 1) % NUM_STAGES
        ct.mbarrier_arrive(mbar[next_stage])
${loadCode.split('\n').map(line => '        ' + line).join('\n')}
        ct.pipeline_commit()

    # Epilogue: drain pipeline
    for stage in range(NUM_STAGES - 1):
        drain_stage = (num_k_blocks + stage) % NUM_STAGES
        ct.mbarrier_wait(mbar[drain_stage])
${computeCode.split('\n').map(line => '        ' + line).join('\n')}
`;
}

/**
 * Calculate occupancy for Blackwell
 */
export function calculateBlackwellOccupancy(
  threadsPerBlock: number,
  registersPerThread: number,
  sharedMemoryPerBlock: number
): {
  blocksPerSM: number;
  occupancy: number;
  limitingFactor: 'threads' | 'registers' | 'shared_memory';
} {
  const config = BLACKWELL_GB100_CONFIG;

  // Blocks limited by threads
  const blocksByThreads = Math.floor(config.maxThreadsPerSM / threadsPerBlock);

  // Blocks limited by registers (registers allocated in groups of 256)
  const registersPerBlock = Math.ceil(registersPerThread * threadsPerBlock / 256) * 256;
  const totalRegistersPerSM = 65536; // 64K registers per SM
  const blocksByRegisters = Math.floor(totalRegistersPerSM / registersPerBlock);

  // Blocks limited by shared memory
  const blocksBySharedMem = Math.floor(config.maxSharedMemory / sharedMemoryPerBlock);

  // Final blocks per SM is minimum
  const blocksPerSM = Math.min(
    config.maxBlocksPerSM,
    blocksByThreads,
    blocksByRegisters,
    blocksBySharedMem
  );

  const activeThreads = blocksPerSM * threadsPerBlock;
  const occupancy = activeThreads / config.maxThreadsPerSM;

  // Determine limiting factor
  let limitingFactor: 'threads' | 'registers' | 'shared_memory';
  if (blocksPerSM === blocksByThreads) {
    limitingFactor = 'threads';
  } else if (blocksPerSM === blocksByRegisters) {
    limitingFactor = 'registers';
  } else {
    limitingFactor = 'shared_memory';
  }

  return { blocksPerSM, occupancy, limitingFactor };
}

/**
 * Blackwell Architecture Helper class
 */
export class BlackwellArchitecture {
  private config: BlackwellConfig;

  constructor(config: BlackwellConfig = BLACKWELL_GB100_CONFIG) {
    this.config = config;
  }

  /**
   * Check if feature is supported
   */
  supportsFeature(feature: 'tma' | 'wgmma' | 'cluster' | 'fp8' | 'barriers'): boolean {
    switch (feature) {
      case 'tma': return this.config.tmaSupport;
      case 'wgmma': return this.config.wgmmaSupport;
      case 'cluster': return this.config.clusterSupport;
      case 'fp8': return this.config.fp8Support;
      case 'barriers': return this.config.arriveWaitBarriers;
    }
  }

  /**
   * Get optimal tile configuration
   */
  getOptimalTileConfig(
    archetype: string,
    problemSize: { m?: number; n?: number; k?: number; seqLen?: number }
  ): BlackwellTileConfig {
    switch (archetype) {
      case 'gemm':
        return getBlackwellGEMMConfig(
          problemSize.m || 1024,
          problemSize.n || 1024,
          problemSize.k || 1024,
          'fp16'
        );
      case 'attention':
        return getBlackwellAttentionConfig(
          1,
          32,
          problemSize.seqLen || 4096,
          128,
          false
        );
      default:
        return {
          blockM: 128,
          blockN: 128,
          blockK: 32,
          stages: 3,
          warpsPerBlockM: 4,
          warpsPerBlockN: 1,
          useTMA: true,
          useWGMMA: false,
          useCluster: false,
          clusterDimX: 1,
          clusterDimY: 1,
          smemSwizzle: true,
          persistentKernel: false,
        };
    }
  }

  /**
   * Generate architecture-specific kernel header
   */
  generateKernelHeader(useTMA: boolean, useWGMMA: boolean, useCluster: boolean): string {
    const features: string[] = ['# Blackwell SM100 optimized kernel'];

    if (useTMA) features.push('# Uses Tensor Memory Accelerator (TMA)');
    if (useWGMMA) features.push('# Uses Warpgroup Matrix Multiply-Accumulate (WGMMA)');
    if (useCluster) features.push('# Uses Thread Block Clusters');

    return features.join('\n');
  }
}

// Export singleton
export const blackwellArch = new BlackwellArchitecture();
