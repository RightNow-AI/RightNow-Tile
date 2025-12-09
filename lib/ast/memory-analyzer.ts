/**
 * Memory Analyzer
 * Analyzes memory access patterns, coalescing efficiency, and suggests tile configurations
 */

import { CudaKernelInfo, MemoryAccess, KernelArchetype } from './types';
import { EnhancedParseResult, IndexPattern, SharedMemoryUsage } from '../parser';

// Memory analysis result types
export interface MemoryAnalysisResult {
  globalMemory: GlobalMemoryAnalysis;
  sharedMemory: SharedMemoryAnalysis;
  registers: RegisterAnalysis;
  tileRecommendation: TileRecommendation;
  accessSummary: AccessSummary;
  optimizationHints: MemoryOptimizationHint[];
}

export interface GlobalMemoryAnalysis {
  totalAccesses: number;
  readCount: number;
  writeCount: number;
  coalescingScore: number;
  hasCoalescedReads: boolean;
  hasCoalescedWrites: boolean;
  stridedAccessCount: number;
  randomAccessCount: number;
  estimatedBandwidthUtilization: number;
}

export interface SharedMemoryAnalysis {
  isUsed: boolean;
  totalBytes: number;
  bankConflictRisk: number;
  accessPatterns: SharedMemoryAccessPattern[];
  isTiledAccess: boolean;
  hasPadding: boolean;
}

export interface SharedMemoryAccessPattern {
  array: string;
  accessType: 'linear' | 'strided' | 'column' | 'diagonal';
  potentialConflicts: number;
}

export interface RegisterAnalysis {
  estimatedRegistersPerThread: number;
  hasRegisterSpilling: boolean;
  accumulatorCount: number;
  localArraysDetected: boolean;
}

export interface TileRecommendation {
  recommended: TileConfig;
  alternatives: TileConfig[];
  justification: string[];
  constraints: TileConstraint[];
}

export interface TileConfig {
  tileSize?: number;
  blockM?: number;
  blockN?: number;
  blockK?: number;
  warpsPerBlock?: number;
  elementsPerThread?: number;
}

export interface TileConstraint {
  type: 'memory' | 'registers' | 'occupancy' | 'alignment';
  description: string;
  limit: number;
}

export interface AccessSummary {
  inputArrays: ArrayAccessInfo[];
  outputArrays: ArrayAccessInfo[];
  inoutArrays: ArrayAccessInfo[];
  reuseFactor: number;
  hasTemporalLocality: boolean;
  hasSpatialLocality: boolean;
}

export interface ArrayAccessInfo {
  name: string;
  accessCount: number;
  uniquePatterns: number;
  isPredictable: boolean;
  bytesPerElement: number;
  totalBytesEstimate: number;
}

export interface MemoryOptimizationHint {
  category: 'coalescing' | 'bank_conflict' | 'tiling' | 'prefetch' | 'vectorize';
  severity: 'info' | 'warning' | 'critical';
  message: string;
  suggestion: string;
  expectedSpeedup?: string;
}

/**
 * Memory Analyzer class
 */
export class MemoryAnalyzer {
  private kernel: CudaKernelInfo | null = null;
  private enhanced: EnhancedParseResult | null = null;

  // Architecture constants
  private readonly WARP_SIZE = 32;
  private readonly SHARED_MEMORY_BANKS = 32;
  private readonly MAX_SHARED_MEMORY = 48 * 1024; // 48KB default
  private readonly MAX_REGISTERS_PER_THREAD = 255;
  private readonly CACHE_LINE_SIZE = 128; // bytes

  /**
   * Analyze memory patterns for a kernel
   */
  analyze(kernel: CudaKernelInfo, enhanced: EnhancedParseResult, archetype?: KernelArchetype): MemoryAnalysisResult {
    this.kernel = kernel;
    this.enhanced = enhanced;

    return {
      globalMemory: this.analyzeGlobalMemory(),
      sharedMemory: this.analyzeSharedMemory(),
      registers: this.analyzeRegisters(),
      tileRecommendation: this.recommendTileConfig(archetype),
      accessSummary: this.summarizeAccesses(),
      optimizationHints: this.generateOptimizationHints(archetype),
    };
  }

  /**
   * Analyze global memory access patterns
   */
  private analyzeGlobalMemory(): GlobalMemoryAnalysis {
    if (!this.kernel || !this.enhanced) {
      return this.defaultGlobalMemory();
    }

    const accesses = this.kernel.memoryAccesses;
    const indexPatterns = this.enhanced.indexPatterns;

    let coalescedCount = 0;
    let stridedCount = 0;
    let randomCount = 0;
    let readCoalesced = 0;
    let writeCoalesced = 0;

    for (const access of accesses) {
      const pattern = indexPatterns.find(p => p.array === access.array);
      if (!pattern) continue;

      switch (pattern.classification) {
        case 'linear':
          coalescedCount++;
          if (access.accessType === 'read') readCoalesced++;
          else writeCoalesced++;
          break;
        case 'strided':
        case '2d':
          stridedCount++;
          break;
        case 'indirect':
        case 'complex':
          randomCount++;
          break;
      }
    }

    const totalAccesses = accesses.length || 1;
    const coalescingScore = coalescedCount / totalAccesses;

    // Estimate bandwidth utilization
    // Perfect coalescing = 100%, random = ~3% (1/32 warp utilization)
    const estimatedBandwidthUtilization =
      (coalescedCount * 1.0 + stridedCount * 0.5 + randomCount * 0.03) / totalAccesses;

    return {
      totalAccesses: accesses.length,
      readCount: accesses.filter(a => a.accessType === 'read').length,
      writeCount: accesses.filter(a => a.accessType === 'write').length,
      coalescingScore,
      hasCoalescedReads: readCoalesced > 0,
      hasCoalescedWrites: writeCoalesced > 0,
      stridedAccessCount: stridedCount,
      randomAccessCount: randomCount,
      estimatedBandwidthUtilization,
    };
  }

  private defaultGlobalMemory(): GlobalMemoryAnalysis {
    return {
      totalAccesses: 0,
      readCount: 0,
      writeCount: 0,
      coalescingScore: 1.0,
      hasCoalescedReads: true,
      hasCoalescedWrites: true,
      stridedAccessCount: 0,
      randomAccessCount: 0,
      estimatedBandwidthUtilization: 1.0,
    };
  }

  /**
   * Analyze shared memory usage and bank conflicts
   */
  private analyzeSharedMemory(): SharedMemoryAnalysis {
    if (!this.enhanced) {
      return {
        isUsed: false,
        totalBytes: 0,
        bankConflictRisk: 0,
        accessPatterns: [],
        isTiledAccess: false,
        hasPadding: false,
      };
    }

    const smem = this.enhanced.sharedMemoryUsage;
    const isUsed = smem.declarations.length > 0;

    if (!isUsed) {
      return {
        isUsed: false,
        totalBytes: 0,
        bankConflictRisk: 0,
        accessPatterns: [],
        isTiledAccess: false,
        hasPadding: false,
      };
    }

    const accessPatterns: SharedMemoryAccessPattern[] = [];
    let totalConflicts = 0;

    for (const decl of smem.declarations) {
      const accesses = smem.accessPatterns.filter(a => a.array === decl.name);
      const conflicting = accesses.filter(a => a.hasBankConflictRisk).length;
      totalConflicts += conflicting;

      // Determine access pattern type
      let accessType: SharedMemoryAccessPattern['accessType'] = 'linear';
      for (const access of accesses) {
        if (/\*\s*\d+/.test(access.indexExpr)) {
          accessType = 'strided';
        } else if (/\[\s*\w+\s*\]\s*\[\s*threadIdx/.test(access.indexExpr)) {
          accessType = 'column';
        }
      }

      accessPatterns.push({
        array: decl.name,
        accessType,
        potentialConflicts: conflicting,
      });
    }

    // Check for padding (common bank conflict mitigation)
    const hasPadding = smem.declarations.some(d => {
      const size = typeof d.size === 'number' ? d.size : 0;
      return size > 0 && size % 33 === 0; // Padded to 33 instead of 32
    });

    // Check for tiled access pattern
    const isTiledAccess = accessPatterns.some(p => p.accessType === 'linear') &&
      smem.declarations.length >= 1;

    const bankConflictRisk = smem.accessPatterns.length > 0
      ? totalConflicts / smem.accessPatterns.length
      : 0;

    return {
      isUsed: true,
      totalBytes: smem.totalStaticBytes,
      bankConflictRisk,
      accessPatterns,
      isTiledAccess,
      hasPadding,
    };
  }

  /**
   * Estimate register usage
   */
  private analyzeRegisters(): RegisterAnalysis {
    if (!this.kernel || !this.enhanced) {
      return {
        estimatedRegistersPerThread: 32,
        hasRegisterSpilling: false,
        accumulatorCount: 0,
        localArraysDetected: false,
      };
    }

    // Count accumulators (likely held in registers)
    const accumulatorCount = this.enhanced.accumulationPatterns.length;

    // Estimate registers
    // Base: ~16 for kernel overhead
    // + 2 per parameter
    // + 4 per loop variable
    // + 8 per accumulator (may use FMA pairs)
    let estimatedRegisters = 16;
    estimatedRegisters += this.kernel.parameters.length * 2;
    estimatedRegisters += this.enhanced.loopAnalysis.length * 4;
    estimatedRegisters += accumulatorCount * 8;

    // Check for local arrays (may spill)
    const source = this.kernel.sourceText;
    const localArraysDetected = /\b(?:float|double|int)\s+\w+\s*\[\s*\d+\s*\]/.test(source) &&
      !/__shared__/.test(source);

    const hasRegisterSpilling = estimatedRegisters > 64 || localArraysDetected;

    return {
      estimatedRegistersPerThread: Math.min(estimatedRegisters, this.MAX_REGISTERS_PER_THREAD),
      hasRegisterSpilling,
      accumulatorCount,
      localArraysDetected,
    };
  }

  /**
   * Recommend tile configuration based on pattern and memory analysis
   */
  private recommendTileConfig(archetype?: KernelArchetype): TileRecommendation {
    const constraints: TileConstraint[] = [];
    const justification: string[] = [];
    let recommended: TileConfig;
    const alternatives: TileConfig[] = [];

    // Base recommendations by archetype
    switch (archetype) {
      case 'gemm':
        recommended = { blockM: 128, blockN: 128, blockK: 32, warpsPerBlock: 4 };
        justification.push('GEMM benefits from large tiles for register blocking');
        justification.push('128x128 tiles with K=32 balances shared memory and occupancy');
        alternatives.push({ blockM: 64, blockN: 64, blockK: 32 });
        alternatives.push({ blockM: 256, blockN: 128, blockK: 32 });
        constraints.push({
          type: 'memory',
          description: 'Shared memory for A and B tiles',
          limit: 2 * 128 * 32 * 4, // bytes
        });
        break;

      case 'reduction':
        recommended = { tileSize: 256, warpsPerBlock: 8 };
        justification.push('Reduction needs enough threads for efficient warp reduction');
        justification.push('256 threads = 8 warps provides good balance');
        alternatives.push({ tileSize: 512 });
        alternatives.push({ tileSize: 128 });
        constraints.push({
          type: 'occupancy',
          description: 'Multiple blocks per SM for hiding latency',
          limit: 4,
        });
        break;

      case 'scan':
        recommended = { tileSize: 256, elementsPerThread: 4 };
        justification.push('Scan requires power-of-2 tile sizes');
        justification.push('256 threads with 4 elements each = 1024 elements per block');
        alternatives.push({ tileSize: 512, elementsPerThread: 2 });
        alternatives.push({ tileSize: 128, elementsPerThread: 8 });
        break;

      case 'stencil':
        // Stencil tile size depends on halo
        const haloSize = this.estimateHaloSize();
        const tileSize = 16 + haloSize; // Include halo
        recommended = { tileSize, blockM: 16, blockN: 16 };
        justification.push(`Stencil with halo size ${haloSize} suggests ${tileSize}x${tileSize} tiles`);
        justification.push('2D tiling reduces redundant halo loads');
        alternatives.push({ blockM: 32, blockN: 8 });
        alternatives.push({ blockM: 8, blockN: 32 });
        constraints.push({
          type: 'memory',
          description: 'Shared memory for tile + halos',
          limit: tileSize * tileSize * 4,
        });
        break;

      case 'elementwise':
      default:
        recommended = { tileSize: 256, elementsPerThread: 4 };
        justification.push('Elementwise operations benefit from high throughput');
        justification.push('256 threads with coalesced access maximizes bandwidth');
        alternatives.push({ tileSize: 512, elementsPerThread: 2 });
        alternatives.push({ tileSize: 128, elementsPerThread: 8 });
        break;
    }

    // Adjust based on shared memory usage
    if (this.enhanced) {
      const smemBytes = this.enhanced.sharedMemoryUsage.totalStaticBytes;
      if (smemBytes > this.MAX_SHARED_MEMORY * 0.75) {
        justification.push('High shared memory usage may limit occupancy');
        constraints.push({
          type: 'memory',
          description: 'Shared memory exceeds 75% capacity',
          limit: this.MAX_SHARED_MEMORY,
        });
      }
    }

    // Register constraints
    const regAnalysis = this.analyzeRegisters();
    if (regAnalysis.hasRegisterSpilling) {
      justification.push('Register spilling detected - consider reducing tile size');
      constraints.push({
        type: 'registers',
        description: 'High register pressure',
        limit: this.MAX_REGISTERS_PER_THREAD,
      });
    }

    return { recommended, alternatives, justification, constraints };
  }

  /**
   * Estimate halo size for stencil patterns
   */
  private estimateHaloSize(): number {
    if (!this.enhanced) return 1;

    const offsets = this.enhanced.indexPatterns
      .filter(p => p.hasNeighborOffset)
      .flatMap(p => p.offsets.map(Math.abs));

    if (offsets.length === 0) return 1;
    return Math.max(...offsets);
  }

  /**
   * Summarize array accesses
   */
  private summarizeAccesses(): AccessSummary {
    if (!this.kernel || !this.enhanced) {
      return {
        inputArrays: [],
        outputArrays: [],
        inoutArrays: [],
        reuseFactor: 1,
        hasTemporalLocality: false,
        hasSpatialLocality: true,
      };
    }

    const arrays = new Map<string, { reads: number; writes: number }>();

    for (const access of this.kernel.memoryAccesses) {
      const info = arrays.get(access.array) || { reads: 0, writes: 0 };
      if (access.accessType === 'read') info.reads++;
      else info.writes++;
      arrays.set(access.array, info);
    }

    const inputArrays: ArrayAccessInfo[] = [];
    const outputArrays: ArrayAccessInfo[] = [];
    const inoutArrays: ArrayAccessInfo[] = [];

    for (const [name, info] of arrays) {
      const arrayInfo: ArrayAccessInfo = {
        name,
        accessCount: info.reads + info.writes,
        uniquePatterns: 1,
        isPredictable: true,
        bytesPerElement: 4,
        totalBytesEstimate: 0,
      };

      if (info.reads > 0 && info.writes > 0) {
        inoutArrays.push(arrayInfo);
      } else if (info.reads > 0) {
        inputArrays.push(arrayInfo);
      } else {
        outputArrays.push(arrayInfo);
      }
    }

    // Compute reuse factor
    const totalAccesses = this.kernel.memoryAccesses.length;
    const uniqueArrays = arrays.size;
    const reuseFactor = uniqueArrays > 0 ? totalAccesses / uniqueArrays : 1;

    // Check for temporal locality (same data accessed multiple times)
    const hasTemporalLocality = reuseFactor > 1.5;

    // Check for spatial locality (coalesced access)
    const hasCoalesced = this.enhanced.indexPatterns.some(p => p.classification === 'linear');
    const hasSpatialLocality = hasCoalesced;

    return {
      inputArrays,
      outputArrays,
      inoutArrays,
      reuseFactor,
      hasTemporalLocality,
      hasSpatialLocality,
    };
  }

  /**
   * Generate optimization hints based on analysis
   */
  private generateOptimizationHints(archetype?: KernelArchetype): MemoryOptimizationHint[] {
    const hints: MemoryOptimizationHint[] = [];

    if (!this.kernel || !this.enhanced) return hints;

    const globalMem = this.analyzeGlobalMemory();
    const sharedMem = this.analyzeSharedMemory();
    const registers = this.analyzeRegisters();

    // Coalescing hints
    if (globalMem.coalescingScore < 0.5) {
      hints.push({
        category: 'coalescing',
        severity: 'critical',
        message: `Low memory coalescing score (${(globalMem.coalescingScore * 100).toFixed(0)}%)`,
        suggestion: 'Reorganize data layout so consecutive threads access consecutive memory',
        expectedSpeedup: '2-10x bandwidth improvement',
      });
    } else if (globalMem.coalescingScore < 0.8) {
      hints.push({
        category: 'coalescing',
        severity: 'warning',
        message: `Moderate memory coalescing score (${(globalMem.coalescingScore * 100).toFixed(0)}%)`,
        suggestion: 'Consider transposing data or using shared memory staging',
        expectedSpeedup: '1.5-2x bandwidth improvement',
      });
    }

    // Bank conflict hints
    if (sharedMem.isUsed && sharedMem.bankConflictRisk > 0.3) {
      hints.push({
        category: 'bank_conflict',
        severity: sharedMem.bankConflictRisk > 0.5 ? 'critical' : 'warning',
        message: `Potential shared memory bank conflicts (${(sharedMem.bankConflictRisk * 100).toFixed(0)}% risk)`,
        suggestion: sharedMem.hasPadding
          ? 'Existing padding detected - verify it eliminates conflicts'
          : 'Add padding (+1 element per row) to eliminate bank conflicts',
        expectedSpeedup: '1.2-2x shared memory bandwidth',
      });
    }

    // Tiling hints
    if (!sharedMem.isUsed && archetype === 'gemm') {
      hints.push({
        category: 'tiling',
        severity: 'critical',
        message: 'GEMM pattern without shared memory tiling',
        suggestion: 'Use shared memory to tile matrix blocks and improve data reuse',
        expectedSpeedup: '5-20x performance improvement',
      });
    }

    if (!sharedMem.isUsed && archetype === 'stencil') {
      hints.push({
        category: 'tiling',
        severity: 'warning',
        message: 'Stencil pattern without shared memory tiling',
        suggestion: 'Use shared memory to cache halo regions and reduce global memory traffic',
        expectedSpeedup: '2-5x performance improvement',
      });
    }

    // Register spilling hints
    if (registers.hasRegisterSpilling) {
      hints.push({
        category: 'coalescing',
        severity: 'warning',
        message: 'High register pressure may cause spilling to local memory',
        suggestion: registers.localArraysDetected
          ? 'Move small local arrays to shared memory'
          : 'Reduce tile size or simplify inner loop computations',
        expectedSpeedup: '1.2-1.5x reduced memory traffic',
      });
    }

    // Vectorization hints
    if (globalMem.hasCoalescedReads && this.enhanced.indexPatterns.every(p => p.classification === 'linear')) {
      hints.push({
        category: 'vectorize',
        severity: 'info',
        message: 'Coalesced linear access pattern detected',
        suggestion: 'Consider using float4/int4 vectorized loads for improved throughput',
        expectedSpeedup: '1.2-1.5x memory bandwidth',
      });
    }

    return hints;
  }
}

// Export singleton instance
export const memoryAnalyzer = new MemoryAnalyzer();
