/**
 * IR Optimizer
 * Optimizes the intermediate representation for better code generation
 */

import { KernelIR, KernelArchetype, PatternVariant } from '../ast/types';
import { MemoryAnalysisResult } from '../ast/memory-analyzer';
import {
  EnhancedKernelIR,
  EnhancedTileConfig,
  TileStrategy,
  OptimizationHint,
  MemoryLayoutInfo,
  mapCudaType,
  getTypeBytes,
} from './types';

export class IROptimizer {
  // Architecture constants
  private readonly WARP_SIZE = 32;
  private readonly MAX_THREADS_PER_BLOCK = 1024;
  private readonly MAX_SHARED_MEMORY = 48 * 1024;
  private readonly MAX_REGISTERS_PER_THREAD = 255;
  private readonly CACHE_LINE_SIZE = 128;

  /**
   * Optimize the IR based on memory analysis
   */
  optimize(ir: KernelIR, memoryAnalysis?: MemoryAnalysisResult, variant?: PatternVariant): EnhancedKernelIR {
    // Create enhanced IR from basic IR
    const enhanced: EnhancedKernelIR = {
      name: ir.name,
      originalName: ir.originalName,
      archetype: ir.archetype,
      variant,
      confidence: ir.confidence,
      parameters: ir.parameters.map(p => ({
        ...p,
        cudaType: p.type,
        cuTileType: mapCudaType(p.type),
        isPointer: p.type.includes('*'),
      })),
      loads: ir.loads.map(l => ({
        ...l,
        dtype: 'ct.float32',
        accessPattern: 'coalesced' as const,
      })),
      operations: ir.operations.map(o => ({
        ...o,
        dtype: 'ct.float32',
      })),
      stores: ir.stores.map(s => ({
        ...s,
        accessPattern: 'coalesced' as const,
      })),
      tileConfig: ir.tileConfig,
      tileStrategy: this.determineTileStrategy(ir, memoryAnalysis),
      semanticInfo: this.extractSemanticInfo(ir),
      optimizationHints: [],
      memoryLayout: this.analyzeMemoryLayout(ir, memoryAnalysis),
    };

    // Apply optimizations
    enhanced.tileConfig = this.optimizeTileConfig(enhanced, memoryAnalysis);
    enhanced.optimizationHints = this.generateOptimizationHints(enhanced, memoryAnalysis);

    // Update load/store patterns based on memory analysis
    if (memoryAnalysis) {
      this.updateAccessPatterns(enhanced, memoryAnalysis);
    }

    return enhanced;
  }

  /**
   * Optimize tile configuration based on archetype and memory analysis
   */
  private optimizeTileConfig(ir: EnhancedKernelIR, memoryAnalysis?: MemoryAnalysisResult): EnhancedTileConfig {
    const base: EnhancedTileConfig = { ...ir.tileConfig };

    switch (ir.archetype) {
      case 'gemm':
        return this.optimizeGEMMTiles(base, ir.variant, memoryAnalysis);
      case 'reduction':
        return this.optimizeReductionTiles(base, ir.variant);
      case 'scan':
        return this.optimizeScanTiles(base, ir.variant);
      case 'stencil':
        return this.optimizeStencilTiles(base, ir.variant);
      case 'histogram':
        return this.optimizeHistogramTiles(base, ir.variant);
      case 'sparse':
        return this.optimizeSparseTiles(base, ir.variant);
      default:
        return this.optimizeElementwiseTiles(base, memoryAnalysis);
    }
  }

  private optimizeGEMMTiles(base: EnhancedTileConfig, variant?: PatternVariant, memory?: MemoryAnalysisResult): EnhancedTileConfig {
    // GEMM tile sizes depend on shared memory and register pressure
    switch (variant) {
      case 'tiled_gemm':
        return {
          ...base,
          blockM: 128,
          blockN: 128,
          blockK: 32,
          warpsPerBlock: 4,
          stages: 3, // Pipeline stages for Ampere+
        };
      case 'register_blocked':
        return {
          ...base,
          blockM: 64,
          blockN: 64,
          blockK: 16,
          warpsPerBlock: 4,
          elementsPerThread: 4,
        };
      case 'naive_gemm':
      default:
        return {
          ...base,
          blockM: 32,
          blockN: 32,
          blockK: 8,
          warpsPerBlock: 2,
        };
    }
  }

  private optimizeReductionTiles(base: EnhancedTileConfig, variant?: PatternVariant): EnhancedTileConfig {
    switch (variant) {
      case 'warp_shuffle':
        return {
          ...base,
          tileSize: 256,
          warpsPerBlock: 8,
          elementsPerThread: 4,
        };
      case 'multi_block':
        return {
          ...base,
          tileSize: 512,
          warpsPerBlock: 16,
          elementsPerThread: 2,
        };
      case 'segmented':
        return {
          ...base,
          tileSize: 128, // Smaller for per-row
          warpsPerBlock: 4,
          elementsPerThread: 8,
        };
      case 'tree_reduction':
      default:
        return {
          ...base,
          tileSize: 256,
          warpsPerBlock: 8,
        };
    }
  }

  private optimizeScanTiles(base: EnhancedTileConfig, variant?: PatternVariant): EnhancedTileConfig {
    switch (variant) {
      case 'exclusive_scan':
      case 'inclusive_scan':
        return {
          ...base,
          tileSize: 256,
          elementsPerThread: 4,
          warpsPerBlock: 8,
        };
      case 'segmented_scan':
        return {
          ...base,
          tileSize: 128,
          elementsPerThread: 8,
          warpsPerBlock: 4,
        };
      default:
        return {
          ...base,
          tileSize: 256,
          elementsPerThread: 4,
        };
    }
  }

  private optimizeStencilTiles(base: EnhancedTileConfig, variant?: PatternVariant): EnhancedTileConfig {
    switch (variant) {
      case 'stencil_2d_9pt':
        return {
          ...base,
          blockM: 16,
          blockN: 16,
          tileSize: 16,
        };
      case 'stencil_2d_5pt':
        return {
          ...base,
          blockM: 32,
          blockN: 8,
          tileSize: 32,
        };
      case 'stencil_3d':
        return {
          ...base,
          blockM: 8,
          blockN: 8,
          blockK: 8,
          tileSize: 8,
        };
      case 'stencil_1d_5pt':
        return {
          ...base,
          tileSize: 128,
          elementsPerThread: 4,
        };
      case 'stencil_1d_3pt':
      default:
        return {
          ...base,
          tileSize: 256,
          elementsPerThread: 4,
        };
    }
  }

  private optimizeHistogramTiles(base: EnhancedTileConfig, variant?: PatternVariant): EnhancedTileConfig {
    switch (variant) {
      case 'histogram_privatized':
        return {
          ...base,
          tileSize: 256,
          warpsPerBlock: 8,
          elementsPerThread: 16, // Each thread processes many elements
        };
      case 'histogram_atomic':
      default:
        return {
          ...base,
          tileSize: 128,
          warpsPerBlock: 4,
          elementsPerThread: 8,
        };
    }
  }

  private optimizeSparseTiles(base: EnhancedTileConfig, variant?: PatternVariant): EnhancedTileConfig {
    switch (variant) {
      case 'spmv_csr':
        return {
          ...base,
          tileSize: 256,
          warpsPerBlock: 8,
          elementsPerThread: 1, // One thread per row typically
        };
      case 'spmv_ell':
        return {
          ...base,
          tileSize: 256,
          warpsPerBlock: 8,
        };
      default:
        return {
          ...base,
          tileSize: 256,
        };
    }
  }

  private optimizeElementwiseTiles(base: EnhancedTileConfig, memory?: MemoryAnalysisResult): EnhancedTileConfig {
    const config: EnhancedTileConfig = {
      ...base,
      tileSize: 256,
      elementsPerThread: 4,
      warpsPerBlock: 8,
    };

    // If memory is bandwidth-bound, increase elements per thread
    if (memory && memory.accessSummary.reuseFactor < 1.5) {
      config.elementsPerThread = 8;
      config.vectorWidth = 4; // Use float4 loads
    }

    return config;
  }

  /**
   * Determine the tile strategy
   */
  private determineTileStrategy(ir: KernelIR, memory?: MemoryAnalysisResult): TileStrategy {
    switch (ir.archetype) {
      case 'gemm':
        return {
          approach: 'blocked',
          dimensions: [
            { name: 'M', size: ir.tileConfig.blockM || 128, axis: 0 },
            { name: 'N', size: ir.tileConfig.blockN || 128, axis: 1 },
            { name: 'K', size: ir.tileConfig.blockK || 32, axis: 2 },
          ],
          justification: 'GEMM uses blocked tiling for register and shared memory reuse',
          estimatedOccupancy: 0.5,
        };

      case 'reduction':
        return {
          approach: 'streaming',
          dimensions: [
            { name: 'tile', size: ir.tileConfig.tileSize || 256, axis: 0 },
          ],
          justification: 'Reduction streams through data with block-level accumulation',
          estimatedOccupancy: 0.75,
        };

      case 'stencil':
        return {
          approach: 'blocked',
          dimensions: [
            { name: 'Y', size: ir.tileConfig.blockM || 16, axis: 0 },
            { name: 'X', size: ir.tileConfig.blockN || 16, axis: 1 },
          ],
          justification: 'Stencil uses 2D tiles with halo regions in shared memory',
          estimatedOccupancy: 0.5,
        };

      case 'scan':
        return {
          approach: 'hierarchical',
          dimensions: [
            { name: 'tile', size: ir.tileConfig.tileSize || 256, axis: 0 },
          ],
          justification: 'Scan uses hierarchical approach with block-local then inter-block phases',
          estimatedOccupancy: 0.5,
        };

      case 'histogram':
        return {
          approach: 'cooperative',
          dimensions: [
            { name: 'tile', size: ir.tileConfig.tileSize || 256, axis: 0 },
          ],
          justification: 'Histogram uses cooperative groups for privatization',
          estimatedOccupancy: 0.5,
        };

      default:
        return {
          approach: 'streaming',
          dimensions: [
            { name: 'tile', size: ir.tileConfig.tileSize || 256, axis: 0 },
          ],
          justification: 'Elementwise operations stream through data',
          estimatedOccupancy: 0.75,
        };
    }
  }

  /**
   * Extract semantic information
   */
  private extractSemanticInfo(ir: KernelIR) {
    const inputArrays = new Set<string>();
    const outputArrays = new Set<string>();
    const intermediates = new Set<string>();

    for (const load of ir.loads) {
      inputArrays.add(load.source);
    }

    for (const store of ir.stores) {
      outputArrays.add(store.target);
    }

    // Find intermediates (used in both)
    for (const arr of inputArrays) {
      if (outputArrays.has(arr)) {
        intermediates.add(arr);
      }
    }

    // Detect reduction operation
    let reductionOp: 'sum' | 'max' | 'min' | 'prod' | undefined;
    for (const op of ir.operations) {
      if (op.type === 'reduce') {
        if (op.op === 'sum' || op.op === 'add') reductionOp = 'sum';
        else if (op.op === 'max') reductionOp = 'max';
        else if (op.op === 'min') reductionOp = 'min';
        else if (op.op === 'prod' || op.op === 'mul') reductionOp = 'prod';
      }
    }

    return {
      reductionOp,
      dataTypes: new Map<string, {
        cudaType: string;
        cuTileType: string;
        bytes: number;
        isFloatingPoint: boolean;
      }>(),
      inputArrays: Array.from(inputArrays),
      outputArrays: Array.from(outputArrays),
      intermediates: Array.from(intermediates),
      hasDataDependency: intermediates.size > 0,
      isThreadSafe: ir.archetype !== 'histogram', // Histogram may have atomics
    };
  }

  /**
   * Analyze memory layout
   */
  private analyzeMemoryLayout(ir: KernelIR, memory?: MemoryAnalysisResult): MemoryLayoutInfo {
    const tileConfig = ir.tileConfig;
    let totalSharedMemory = 0;

    // Estimate shared memory usage
    switch (ir.archetype) {
      case 'gemm':
        // A tile + B tile
        const blockM = tileConfig.blockM || 128;
        const blockN = tileConfig.blockN || 128;
        const blockK = tileConfig.blockK || 32;
        totalSharedMemory = (blockM * blockK + blockK * blockN) * 4; // float
        break;

      case 'reduction':
        totalSharedMemory = (tileConfig.tileSize || 256) * 4;
        break;

      case 'stencil':
        // Tile + halo
        const tileSize = tileConfig.tileSize || 16;
        totalSharedMemory = (tileSize + 2) * (tileSize + 2) * 4;
        break;

      case 'histogram':
        totalSharedMemory = 256 * 4; // Typical histogram bins
        break;

      default:
        totalSharedMemory = (tileConfig.tileSize || 256) * 4;
    }

    return {
      totalSharedMemory,
      registersPerThread: 32, // Estimate
      globalMemoryReads: ir.loads.length,
      globalMemoryWrites: ir.stores.length,
      sharedMemoryBankConflictFree: memory ? memory.sharedMemory.bankConflictRisk < 0.2 : true,
    };
  }

  /**
   * Generate optimization hints
   */
  private generateOptimizationHints(ir: EnhancedKernelIR, memory?: MemoryAnalysisResult): OptimizationHint[] {
    const hints: OptimizationHint[] = [];

    // Vectorization hint for elementwise
    if (ir.archetype === 'elementwise' && ir.loads.every(l => l.accessPattern === 'coalesced')) {
      hints.push({
        category: 'vectorize',
        target: 'loads',
        suggestion: 'Use float4 vectorized loads for improved memory bandwidth',
        priority: 'medium',
        expectedBenefit: '1.5-2x memory throughput',
      });
    }

    // Unroll hint for reductions
    if (ir.archetype === 'reduction' && ir.variant === 'warp_shuffle') {
      hints.push({
        category: 'unroll',
        target: 'warp_reduction',
        suggestion: 'Fully unroll warp shuffle reduction loop',
        priority: 'high',
        expectedBenefit: 'Eliminate loop overhead',
      });
    }

    // Shared memory hint for GEMM
    if (ir.archetype === 'gemm' && ir.memoryLayout.totalSharedMemory < this.MAX_SHARED_MEMORY) {
      hints.push({
        category: 'shared_memory',
        target: 'tile_staging',
        suggestion: 'Increase tile size to use more shared memory',
        priority: 'medium',
        expectedBenefit: 'Better data reuse',
      });
    }

    // Prefetch hint for stencil
    if (ir.archetype === 'stencil') {
      hints.push({
        category: 'prefetch',
        target: 'halo_region',
        suggestion: 'Prefetch next tile while computing current',
        priority: 'low',
        expectedBenefit: 'Hide memory latency',
      });
    }

    // Register blocking hint for GEMM
    if (ir.archetype === 'gemm' && ir.variant !== 'register_blocked') {
      hints.push({
        category: 'register_blocking',
        target: 'accumulator',
        suggestion: 'Use 4x4 register blocking for C accumulator',
        priority: 'high',
        expectedBenefit: '2-4x compute efficiency',
      });
    }

    return hints;
  }

  /**
   * Update access patterns based on memory analysis
   */
  private updateAccessPatterns(ir: EnhancedKernelIR, memory: MemoryAnalysisResult): void {
    // Update load patterns
    for (const load of ir.loads) {
      const accessInfo = memory.accessSummary.inputArrays.find(a => a.name === load.source);
      if (accessInfo) {
        load.accessPattern = accessInfo.isPredictable ? 'coalesced' : 'random';
      }
    }

    // Add cache hints based on reuse
    if (memory.accessSummary.hasTemporalLocality) {
      for (const load of ir.loads) {
        load.cacheHint = 'persistent';
      }
    } else {
      for (const load of ir.loads) {
        load.cacheHint = 'streaming';
      }
    }

    // Add prefetch for high bandwidth operations
    if (memory.globalMemory.estimatedBandwidthUtilization > 0.7) {
      for (const load of ir.loads) {
        load.prefetch = true;
      }
    }
  }
}

// Export singleton instance
export const irOptimizer = new IROptimizer();
