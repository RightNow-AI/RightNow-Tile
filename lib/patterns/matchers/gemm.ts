// GEMM (General Matrix Multiply) Pattern Matcher
// Detects matrix multiplication kernels

import { CudaKernelInfo, PatternMatch, Evidence } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class GEMMMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText.toLowerCase();

    // === Primary Indicators (high weight) ===

    // 1. Accumulation loop pattern (sum += A[...] * B[...])
    const hasAccumulationLoop = this.detectAccumulationLoop(kernel);
    if (hasAccumulationLoop) {
      addEvidence(evidence, 'accumulation_loop', 0.25,
        'Found inner accumulation loop (k-dimension)');
    }

    // 2. Matrix access patterns (A[row][k], B[k][col], C[row][col])
    const matrixPatterns = this.analyzeMatrixAccessPatterns(kernel);
    if (matrixPatterns.hasRowMajorAccess && matrixPatterns.hasColMajorAccess) {
      addEvidence(evidence, 'matrix_access_pattern', 0.30,
        'Found A[row][k] * B[k][col] access pattern');
    } else if (matrixPatterns.hasRowMajorAccess || matrixPatterns.hasColMajorAccess) {
      addEvidence(evidence, 'partial_matrix_access', 0.15,
        'Found partial matrix access pattern');
    }

    // 3. Output matrix write pattern
    const hasMatrixOutput = this.detectMatrixOutput(kernel);
    if (hasMatrixOutput) {
      addEvidence(evidence, 'output_matrix_write', 0.20,
        'Found C[row][col] output pattern');
    }

    // === Secondary Indicators (medium weight) ===

    // 4. 2D thread indexing (row/col based)
    if (kernel.threadIndexUsage.usesBlockIdxX && kernel.threadIndexUsage.usesBlockIdxY) {
      addEvidence(evidence, '2d_block_indexing', 0.15,
        'Uses 2D block indexing (blockIdx.x and blockIdx.y)');
    } else if (kernel.threadIndexUsage.usesThreadIdxY) {
      addEvidence(evidence, '2d_thread_indexing', 0.10,
        'Uses 2D thread indexing');
    }

    // 5. Shared memory tiling (typical optimization)
    if (kernel.sharedMemoryDecls.length >= 2) {
      addEvidence(evidence, 'shared_memory_tiling', 0.10,
        'Found shared memory tiling (2+ shared arrays)');
    } else if (kernel.sharedMemoryDecls.length === 1) {
      addEvidence(evidence, 'shared_memory_single', 0.05,
        'Found shared memory usage');
    }

    // 6. Multiple __syncthreads (tile synchronization)
    const syncCount = kernel.syncPoints.filter(s => s.type === 'syncthreads').length;
    if (syncCount >= 2) {
      addEvidence(evidence, 'tiled_synchronization', 0.05,
        `Multiple __syncthreads (${syncCount}) suggests tiled algorithm`);
    }

    // 7. Name-based hints
    if (/matmul|gemm|mm_|matrix_mul|matrixmul/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests matrix multiplication');
    }

    // 8. Three pointer parameters (A, B, C matrices)
    const pointerParams = kernel.parameters.filter(p => p.isPointer);
    if (pointerParams.length >= 3) {
      addEvidence(evidence, 'three_matrix_params', 0.10,
        'Has 3+ pointer parameters (typical for A, B, C matrices)');
    }

    // 9. Nested loops (i, j, k pattern)
    const nestedLoopDepth = this.calculateNestedLoopDepth(kernel);
    if (nestedLoopDepth >= 2) {
      addEvidence(evidence, 'nested_loops', 0.10,
        `Found ${nestedLoopDepth}-deep nested loops`);
    }

    // === Negative Indicators ===

    // Stride halving suggests reduction
    const hasStrideHalving = kernel.loops.some(l => l.hasStrideHalving);
    if (hasStrideHalving) {
      addEvidence(evidence, 'stride_halving', -0.20,
        'Stride halving pattern (likely reduction)');
    }

    // Neighbor access suggests stencil
    const neighborAccesses = kernel.memoryAccesses.filter(a => a.hasNeighborOffset);
    if (neighborAccesses.length > 3) {
      addEvidence(evidence, 'neighbor_access', -0.15,
        'Multiple neighbor accesses (likely stencil)');
    }

    // Single loop without matrix patterns
    if (kernel.loops.length === 1 && !hasAccumulationLoop) {
      addEvidence(evidence, 'single_loop', -0.10,
        'Single loop without accumulation');
    }

    return createPatternMatch('gemm', evidence, warnings);
  }

  private detectAccumulationLoop(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // Look for patterns like: sum += a[...] * b[...]
    const accumulationPattern = /\w+\s*\+=\s*\w+\s*\[[^\]]+\]\s*\*\s*\w+\s*\[[^\]]+\]/;
    if (accumulationPattern.test(source)) {
      return true;
    }

    // Also check for: sum = sum + a[...] * b[...]
    const explicitAccum = /\w+\s*=\s*\w+\s*\+\s*\w+\s*\[[^\]]+\]\s*\*\s*\w+\s*\[[^\]]+\]/;
    if (explicitAccum.test(source)) {
      return true;
    }

    // Check for FMA pattern
    const fmaPattern = /fma\s*\(|__fmaf\s*\(/;
    if (fmaPattern.test(source)) {
      return true;
    }

    return false;
  }

  private analyzeMatrixAccessPatterns(kernel: CudaKernelInfo): {
    hasRowMajorAccess: boolean;
    hasColMajorAccess: boolean;
  } {
    let hasRowMajorAccess = false;
    let hasColMajorAccess = false;

    const source = kernel.sourceText;

    // Row-major pattern: arr[row * width + col] or arr[row][col]
    // A[i][k] pattern in GEMM
    const rowMajorPattern = /\w+\s*\[\s*\w+\s*\*\s*\w+\s*\+\s*\w+\s*\]/;
    const twoDimRowPattern = /\w+\s*\[\s*\w+\s*\]\s*\[\s*\w+\s*\]/;

    if (rowMajorPattern.test(source) || twoDimRowPattern.test(source)) {
      hasRowMajorAccess = true;
    }

    // Col-major pattern typically shares same structure
    // B[k][j] pattern - just check for different index variables
    const accesses = kernel.memoryAccesses.filter(a => a.indexExpression.includes('*'));
    if (accesses.length >= 2) {
      // Multiple linearized 2D accesses suggest matrix pattern
      hasColMajorAccess = true;
    }

    // Check for distinct index patterns
    const indexVarSets = kernel.memoryAccesses.map(a => new Set(a.indexVars));
    if (indexVarSets.length >= 2) {
      hasRowMajorAccess = true;
      hasColMajorAccess = true;
    }

    return { hasRowMajorAccess, hasColMajorAccess };
  }

  private detectMatrixOutput(kernel: CudaKernelInfo): boolean {
    const writes = kernel.memoryAccesses.filter(a => a.accessType === 'write');

    for (const write of writes) {
      // Check for 2D index pattern in write
      if (write.indexExpression.includes('*') ||
          write.indexExpression.includes('][')) {
        return true;
      }

      // Check for row/col variables
      const indexLower = write.indexExpression.toLowerCase();
      if ((indexLower.includes('row') || indexLower.includes('i')) &&
          (indexLower.includes('col') || indexLower.includes('j'))) {
        return true;
      }
    }

    return false;
  }

  private calculateNestedLoopDepth(kernel: CudaKernelInfo): number {
    if (kernel.loops.length === 0) return 0;
    return Math.max(...kernel.loops.map(l => l.nestLevel)) + 1;
  }
}

export const gemmMatcher = new GEMMMatcher();
