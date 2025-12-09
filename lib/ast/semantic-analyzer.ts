/**
 * Semantic Analyzer
 * Performs data flow analysis, dependency detection, and semantic classification
 */

import { CudaKernelInfo, MemoryAccess } from './types';
import { EnhancedParseResult, PatternSignal } from '../parser';

// Semantic analysis result types
export interface SemanticAnalysisResult {
  reductionVariables: ReductionVariable[];
  inductionVariables: InductionVariable[];
  accessPatterns: AccessPatternClassification[];
  dataFlow: DataFlowInfo;
  hasBarrierDivergence: boolean;
  possibleRaces: RaceCondition[];
  parallelismType: ParallelismType;
  computeIntensity: ComputeIntensityMetrics;
}

export interface ReductionVariable {
  name: string;
  operation: ReductionOp;
  scope: 'warp' | 'block' | 'global';
  initValue?: string;
  isPrivate: boolean;
  usesAtomic: boolean;
  usesWarpShuffle: boolean;
  confidence: number;
}

export type ReductionOp = 'sum' | 'product' | 'max' | 'min' | 'and' | 'or' | 'xor' | 'count';

export interface InductionVariable {
  name: string;
  initExpression: string;
  stepExpression: string;
  boundExpression: string;
  stepType: 'linear' | 'multiply' | 'divide' | 'shift';
  direction: 'increasing' | 'decreasing';
  dependsOnThreadIdx: boolean;
}

export interface AccessPatternClassification {
  array: string;
  pattern: 'coalesced' | 'strided' | 'random' | 'broadcast' | 'neighbor';
  stride?: number;
  neighborOffsets?: number[];
  coalescingEfficiency: number;
  isPredictable: boolean;
}

export interface DataFlowInfo {
  inputs: string[];
  outputs: string[];
  intermediates: string[];
  dependencies: DataDependency[];
  hasTrueDataDependency: boolean;
  hasAntiDependency: boolean;
  hasOutputDependency: boolean;
}

export interface DataDependency {
  from: string;
  to: string;
  type: 'true' | 'anti' | 'output';
  distance?: number;
  isLoopCarried: boolean;
}

export interface RaceCondition {
  array: string;
  type: 'read-write' | 'write-write';
  line1: number;
  line2: number;
  severity: 'error' | 'warning';
  suggestion: string;
}

export type ParallelismType =
  | 'data_parallel'
  | 'task_parallel'
  | 'reduction'
  | 'scan'
  | 'stencil'
  | 'irregular';

export interface ComputeIntensityMetrics {
  flopsPerByte: number;
  isComputeBound: boolean;
  isMemoryBound: boolean;
  arithmeticIntensity: 'low' | 'medium' | 'high';
}

/**
 * Semantic Analyzer class
 */
export class SemanticAnalyzer {
  private kernel: CudaKernelInfo | null = null;
  private enhanced: EnhancedParseResult | null = null;

  /**
   * Analyze kernel semantics combining basic and enhanced parse results
   */
  analyze(kernel: CudaKernelInfo, enhanced: EnhancedParseResult): SemanticAnalysisResult {
    this.kernel = kernel;
    this.enhanced = enhanced;

    return {
      reductionVariables: this.findReductionVariables(),
      inductionVariables: this.findInductionVariables(),
      accessPatterns: this.classifyAccessPatterns(),
      dataFlow: this.analyzeDataFlow(),
      hasBarrierDivergence: this.checkBarrierDivergence(),
      possibleRaces: this.detectPossibleRaces(),
      parallelismType: this.classifyParallelism(),
      computeIntensity: this.analyzeComputeIntensity(),
    };
  }

  /**
   * Find variables that are being reduced (accumulated)
   */
  private findReductionVariables(): ReductionVariable[] {
    if (!this.kernel || !this.enhanced) return [];

    const reductions: ReductionVariable[] = [];
    const source = this.kernel.sourceText;

    // Pattern 1: sum += expr
    const plusEqualsPattern = /(\w+)\s*\+=\s*([^;]+)/g;
    let match;
    while ((match = plusEqualsPattern.exec(source)) !== null) {
      const varName = match[1];
      if (!this.isLoopVariable(varName)) {
        reductions.push({
          name: varName,
          operation: 'sum',
          scope: this.determineReductionScope(varName),
          isPrivate: this.isPrivateVariable(varName),
          usesAtomic: false,
          usesWarpShuffle: false,
          confidence: 0.7,
        });
      }
    }

    // Pattern 2: prod *= expr
    const mulEqualsPattern = /(\w+)\s*\*=\s*([^;]+)/g;
    while ((match = mulEqualsPattern.exec(source)) !== null) {
      const varName = match[1];
      if (!this.isLoopVariable(varName)) {
        reductions.push({
          name: varName,
          operation: 'product',
          scope: this.determineReductionScope(varName),
          isPrivate: this.isPrivateVariable(varName),
          usesAtomic: false,
          usesWarpShuffle: false,
          confidence: 0.65,
        });
      }
    }

    // Pattern 3: var = max(var, expr) or var = min(var, expr)
    const maxMinPattern = /(\w+)\s*=\s*(max|min)\s*\(\s*\1\s*,\s*([^)]+)\)/g;
    while ((match = maxMinPattern.exec(source)) !== null) {
      reductions.push({
        name: match[1],
        operation: match[2] as ReductionOp,
        scope: this.determineReductionScope(match[1]),
        isPrivate: this.isPrivateVariable(match[1]),
        usesAtomic: false,
        usesWarpShuffle: false,
        confidence: 0.8,
      });
    }

    // Pattern 4: atomicAdd, atomicMax, etc.
    for (const rp of this.enhanced.reductionPatterns) {
      if (rp.hasAtomic) {
        const existing = reductions.find(r => r.name === rp.variable);
        if (existing) {
          existing.usesAtomic = true;
          existing.scope = 'global';
          existing.confidence = Math.min(existing.confidence + 0.2, 1.0);
        } else {
          reductions.push({
            name: rp.variable,
            operation: rp.operation as ReductionOp,
            scope: 'global',
            isPrivate: false,
            usesAtomic: true,
            usesWarpShuffle: false,
            confidence: 0.9,
          });
        }
      }

      if (rp.isWarpLevel) {
        const existing = reductions.find(r => r.name === rp.variable);
        if (existing) {
          existing.usesWarpShuffle = true;
          existing.scope = 'warp';
          existing.confidence = Math.min(existing.confidence + 0.15, 1.0);
        }
      }
    }

    // Deduplicate by name, keeping highest confidence
    const unique = new Map<string, ReductionVariable>();
    for (const r of reductions) {
      const existing = unique.get(r.name);
      if (!existing || r.confidence > existing.confidence) {
        unique.set(r.name, r);
      }
    }

    return Array.from(unique.values());
  }

  /**
   * Find induction variables (loop counters with predictable patterns)
   */
  private findInductionVariables(): InductionVariable[] {
    if (!this.kernel || !this.enhanced) return [];

    const inductionVars: InductionVariable[] = [];

    for (const loop of this.enhanced.loopAnalysis) {
      if (!loop.variable) continue;

      let stepType: InductionVariable['stepType'] = 'linear';
      let direction: InductionVariable['direction'] = 'increasing';

      if (loop.hasStrideHalving) {
        stepType = 'divide';
        direction = 'decreasing';
      } else if (loop.hasStrideDoubling) {
        stepType = 'multiply';
        direction = 'increasing';
      } else if (/--|\-=/.test(loop.update)) {
        direction = 'decreasing';
      } else if (/>>=/.test(loop.update)) {
        stepType = 'shift';
        direction = 'decreasing';
      } else if (/<<=/.test(loop.update)) {
        stepType = 'shift';
        direction = 'increasing';
      }

      inductionVars.push({
        name: loop.variable,
        initExpression: loop.init,
        stepExpression: loop.update,
        boundExpression: loop.condition,
        stepType,
        direction,
        dependsOnThreadIdx: /threadIdx|blockIdx|blockDim/.test(loop.init + loop.condition),
      });
    }

    return inductionVars;
  }

  /**
   * Classify memory access patterns for each array
   */
  private classifyAccessPatterns(): AccessPatternClassification[] {
    if (!this.kernel || !this.enhanced) return [];

    const patterns: AccessPatternClassification[] = [];
    const arrayPatterns = new Map<string, AccessPatternClassification>();

    for (const ip of this.enhanced.indexPatterns) {
      let pattern: AccessPatternClassification['pattern'] = 'coalesced';
      let stride: number | undefined;
      let coalescingEfficiency = 1.0;

      switch (ip.classification) {
        case 'linear':
          if (ip.dependsOnThreadIdx) {
            pattern = 'coalesced';
            coalescingEfficiency = 1.0;
          } else if (ip.dependsOnBlockIdx) {
            pattern = 'broadcast';
            coalescingEfficiency = 0.03; // Only 1/32 efficiency
          }
          break;
        case 'strided':
          pattern = 'strided';
          // Try to extract stride
          const strideMatch = ip.indexExpr.match(/\*\s*(\d+)/);
          if (strideMatch) {
            stride = parseInt(strideMatch[1]);
            coalescingEfficiency = 1 / stride;
          } else {
            coalescingEfficiency = 0.5;
          }
          break;
        case 'indirect':
          pattern = 'random';
          coalescingEfficiency = 0.1;
          break;
        case '2d':
          // 2D can be coalesced if inner dimension is thread-indexed
          if (/threadIdx\s*\.\s*x/.test(ip.indexExpr)) {
            pattern = 'coalesced';
            coalescingEfficiency = 0.9;
          } else {
            pattern = 'strided';
            coalescingEfficiency = 0.5;
          }
          break;
        case 'complex':
          pattern = 'random';
          coalescingEfficiency = 0.2;
          break;
      }

      // Check for stencil pattern
      if (ip.hasNeighborOffset && ip.offsets.length > 0) {
        pattern = 'neighbor';
        coalescingEfficiency = 0.8; // Neighbor accesses have some redundant loads
      }

      const existing = arrayPatterns.get(ip.array);
      if (!existing) {
        arrayPatterns.set(ip.array, {
          array: ip.array,
          pattern,
          stride,
          neighborOffsets: ip.hasNeighborOffset ? ip.offsets : undefined,
          coalescingEfficiency,
          isPredictable: ip.classification !== 'indirect' && ip.classification !== 'complex',
        });
      } else {
        // Merge patterns - worst case wins
        if (coalescingEfficiency < existing.coalescingEfficiency) {
          existing.pattern = pattern;
          existing.coalescingEfficiency = coalescingEfficiency;
          existing.stride = stride;
        }
        if (ip.hasNeighborOffset) {
          existing.neighborOffsets = [...(existing.neighborOffsets || []), ...ip.offsets];
        }
      }
    }

    return Array.from(arrayPatterns.values());
  }

  /**
   * Analyze data flow: inputs, outputs, and dependencies
   */
  private analyzeDataFlow(): DataFlowInfo {
    if (!this.kernel) {
      return {
        inputs: [],
        outputs: [],
        intermediates: [],
        dependencies: [],
        hasTrueDataDependency: false,
        hasAntiDependency: false,
        hasOutputDependency: false,
      };
    }

    const reads = new Set<string>();
    const writes = new Set<string>();
    const dependencies: DataDependency[] = [];

    // Categorize accesses
    for (const access of this.kernel.memoryAccesses) {
      if (access.accessType === 'read') {
        reads.add(access.array);
      } else {
        writes.add(access.array);
      }
    }

    // Find inputs (read but not written first)
    const inputs = Array.from(reads);
    // Find outputs (written)
    const outputs = Array.from(writes);
    // Intermediates are both read and written
    const intermediates = inputs.filter(i => writes.has(i));

    // Detect dependencies
    let hasTrueDataDependency = false;
    let hasAntiDependency = false;
    let hasOutputDependency = false;

    // Check for RAW (true) dependencies
    for (const arr of intermediates) {
      const arrAccesses = this.kernel.memoryAccesses.filter(a => a.array === arr);
      const readAccesses = arrAccesses.filter(a => a.accessType === 'read');
      const writeAccesses = arrAccesses.filter(a => a.accessType === 'write');

      for (const write of writeAccesses) {
        for (const read of readAccesses) {
          if (write.line < read.line) {
            dependencies.push({
              from: `${arr}[write:${write.line}]`,
              to: `${arr}[read:${read.line}]`,
              type: 'true',
              isLoopCarried: false,
            });
            hasTrueDataDependency = true;
          }
          if (read.line < write.line) {
            dependencies.push({
              from: `${arr}[read:${read.line}]`,
              to: `${arr}[write:${write.line}]`,
              type: 'anti',
              isLoopCarried: false,
            });
            hasAntiDependency = true;
          }
        }

        // Check for WAW (output) dependencies
        for (const otherWrite of writeAccesses) {
          if (write !== otherWrite && write.line !== otherWrite.line) {
            hasOutputDependency = true;
          }
        }
      }
    }

    return {
      inputs,
      outputs,
      intermediates,
      dependencies,
      hasTrueDataDependency,
      hasAntiDependency,
      hasOutputDependency,
    };
  }

  /**
   * Check for potential barrier divergence issues
   */
  private checkBarrierDivergence(): boolean {
    if (!this.kernel) return false;

    const source = this.kernel.sourceText;

    // Check for __syncthreads inside conditionals that depend on thread ID
    const conditionalSyncPattern = /if\s*\([^)]*(?:threadIdx|tid|thread_id)[^)]*\)\s*\{[^}]*__syncthreads/;
    if (conditionalSyncPattern.test(source)) {
      return true;
    }

    // Check for __syncthreads in early return paths
    const earlyReturnPattern = /if\s*\([^)]*\)\s*(?:return|break)\s*;[\s\S]*__syncthreads/;
    if (earlyReturnPattern.test(source)) {
      return true;
    }

    return false;
  }

  /**
   * Detect possible race conditions
   */
  private detectPossibleRaces(): RaceCondition[] {
    if (!this.kernel || !this.enhanced) return [];

    const races: RaceCondition[] = [];

    // Group accesses by array
    const accessesByArray = new Map<string, MemoryAccess[]>();
    for (const access of this.kernel.memoryAccesses) {
      const list = accessesByArray.get(access.array) || [];
      list.push(access);
      accessesByArray.set(access.array, list);
    }

    // Check shared memory for races (no automatic synchronization)
    for (const decl of this.enhanced.sharedMemoryUsage.declarations) {
      const accesses = accessesByArray.get(decl.name);
      if (!accesses || accesses.length < 2) continue;

      const writes = accesses.filter(a => a.accessType === 'write');
      const reads = accesses.filter(a => a.accessType === 'read');

      // Check for write-write races
      for (let i = 0; i < writes.length; i++) {
        for (let j = i + 1; j < writes.length; j++) {
          if (!this.hasSyncBetween(writes[i].line, writes[j].line)) {
            races.push({
              array: decl.name,
              type: 'write-write',
              line1: writes[i].line,
              line2: writes[j].line,
              severity: 'error',
              suggestion: 'Add __syncthreads() between writes to shared memory',
            });
          }
        }
      }

      // Check for read-write races
      for (const write of writes) {
        for (const read of reads) {
          if (Math.abs(write.line - read.line) > 1 && !this.hasSyncBetween(write.line, read.line)) {
            races.push({
              array: decl.name,
              type: 'read-write',
              line1: Math.min(write.line, read.line),
              line2: Math.max(write.line, read.line),
              severity: 'warning',
              suggestion: 'Consider adding __syncthreads() to prevent potential race',
            });
          }
        }
      }
    }

    return races;
  }

  /**
   * Classify the type of parallelism used
   */
  private classifyParallelism(): ParallelismType {
    if (!this.kernel || !this.enhanced) return 'data_parallel';

    const signals = this.enhanced.patternSignals;
    const hasReduction = signals.some(s => s.pattern === 'reduction');
    const hasScan = signals.some(s => s.pattern === 'scan');
    const hasStencil = signals.some(s => s.pattern === 'stencil');

    // Check for irregular access patterns
    const hasIndirect = this.enhanced.indexPatterns.some(p => p.classification === 'indirect');
    if (hasIndirect) return 'irregular';

    if (hasScan) return 'scan';
    if (hasReduction) return 'reduction';
    if (hasStencil) return 'stencil';

    // Default is data parallel (elementwise or GEMM)
    return 'data_parallel';
  }

  /**
   * Analyze compute intensity (ops per byte)
   */
  private analyzeComputeIntensity(): ComputeIntensityMetrics {
    if (!this.kernel || !this.enhanced) {
      return {
        flopsPerByte: 0,
        isComputeBound: false,
        isMemoryBound: true,
        arithmeticIntensity: 'low',
      };
    }

    // Count operations
    let flops = 0;
    const accPatterns = this.enhanced.accumulationPatterns;

    // FMA counts as 2 flops
    flops += accPatterns.filter(p => p.operation === 'fma').length * 2;
    flops += accPatterns.filter(p => p.operation === 'add' || p.operation === 'mul').length;

    // Count math intrinsics
    for (const call of this.enhanced.intrinsicCalls) {
      if (call.info.category === 'math') {
        flops += 1;
      }
    }

    // Count memory accesses
    const memoryAccesses = this.kernel.memoryAccesses.length;
    const bytesAccessed = memoryAccesses * 4; // Assume 4 bytes per access

    const flopsPerByte = bytesAccessed > 0 ? flops / bytesAccessed : 0;

    // Thresholds based on GPU architecture (simplified)
    // Modern GPUs need ~10-20 flops/byte to be compute bound
    const isComputeBound = flopsPerByte >= 10;
    const isMemoryBound = flopsPerByte < 5;

    let arithmeticIntensity: ComputeIntensityMetrics['arithmeticIntensity'];
    if (flopsPerByte < 2) {
      arithmeticIntensity = 'low';
    } else if (flopsPerByte < 10) {
      arithmeticIntensity = 'medium';
    } else {
      arithmeticIntensity = 'high';
    }

    return {
      flopsPerByte,
      isComputeBound,
      isMemoryBound,
      arithmeticIntensity,
    };
  }

  // Helper methods

  private isLoopVariable(name: string): boolean {
    if (!this.enhanced) return false;
    return this.enhanced.loopAnalysis.some(l => l.variable === name);
  }

  private determineReductionScope(varName: string): ReductionVariable['scope'] {
    if (!this.kernel) return 'block';

    const source = this.kernel.sourceText;

    // Check for atomic operations on this variable
    if (new RegExp(`atomic\\w+\\s*\\([^,]*${varName}`).test(source)) {
      return 'global';
    }

    // Check for warp shuffle
    if (new RegExp(`__shfl\\w*\\s*\\([^)]*${varName}`).test(source)) {
      return 'warp';
    }

    return 'block';
  }

  private isPrivateVariable(varName: string): boolean {
    if (!this.kernel) return true;

    const source = this.kernel.sourceText;

    // Check if it's in shared memory
    if (new RegExp(`__shared__[^;]*${varName}`).test(source)) {
      return false;
    }

    // Check if it's a pointer parameter
    const param = this.kernel.parameters.find(p => p.name === varName);
    if (param && param.isPointer) {
      return false;
    }

    return true;
  }

  private hasSyncBetween(line1: number, line2: number): boolean {
    if (!this.kernel) return false;

    const minLine = Math.min(line1, line2);
    const maxLine = Math.max(line1, line2);

    return this.kernel.syncPoints.some(
      s => s.line > minLine && s.line < maxLine
    );
  }
}

// Export singleton instance
export const semanticAnalyzer = new SemanticAnalyzer();
