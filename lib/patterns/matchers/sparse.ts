/**
 * Sparse Matrix Pattern Matcher
 * Detects sparse matrix-vector multiplication (SpMV) and related patterns
 * Supports CSR, COO, ELL, and other sparse formats
 */

import { CudaKernelInfo, PatternMatch, Evidence, PatternVariant } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class SparseMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;

    // === Primary Indicators (high weight) ===

    // 1. CSR format detection (row_ptr, col_idx, values)
    const csrInfo = this.detectCSRFormat(kernel, source);
    if (csrInfo.found) {
      addEvidence(evidence, 'csr_format', 0.35,
        `CSR format arrays detected: ${csrInfo.arrays.join(', ')}`);
    }

    // 2. Row pointer iteration (row_ptr[row] to row_ptr[row+1])
    const rowPtrInfo = this.detectRowPtrIteration(source);
    if (rowPtrInfo.found) {
      addEvidence(evidence, 'row_ptr_iteration', 0.30,
        `Row pointer iteration: ${rowPtrInfo.pattern}`);
    }

    // 3. Indirect array access (x[col_idx[j]])
    const indirectInfo = this.detectIndirectAccess(kernel, source);
    if (indirectInfo.found) {
      addEvidence(evidence, 'indirect_access', 0.25,
        `Indirect access: ${indirectInfo.count} patterns found`);
    }

    // === Format-Specific Indicators ===

    // 4. COO format detection (row_ind, col_ind, values)
    const cooInfo = this.detectCOOFormat(kernel, source);
    if (cooInfo.found) {
      addEvidence(evidence, 'coo_format', 0.30,
        'COO format arrays detected');
    }

    // 5. ELL format detection (fixed elements per row, padded)
    const ellInfo = this.detectELLFormat(source);
    if (ellInfo.found) {
      addEvidence(evidence, 'ell_format', 0.30,
        'ELL format pattern detected');
    }

    // === Secondary Indicators (medium weight) ===

    // 6. Inner product accumulation in row loop
    const hasInnerProduct = this.detectInnerProduct(source);
    if (hasInnerProduct) {
      addEvidence(evidence, 'inner_product', 0.15,
        'Inner product accumulation pattern');
    }

    // 7. Thread-per-row or warp-per-row mapping
    const threadMapping = this.detectThreadMapping(source);
    if (threadMapping !== 'none') {
      addEvidence(evidence, 'sparse_thread_mapping', 0.10,
        `Thread mapping: ${threadMapping}`);
    }

    // 8. Variable-length row processing
    if (rowPtrInfo.found && this.hasVariableLengthLoop(source)) {
      addEvidence(evidence, 'variable_length_rows', 0.10,
        'Variable-length row iteration detected');
    }

    // 9. Warp shuffle for segmented reduction
    const hasSegmentedReduction = this.detectSegmentedReduction(source);
    if (hasSegmentedReduction) {
      addEvidence(evidence, 'segmented_reduction', 0.15,
        'Warp-level segmented reduction');
    }

    // 10. Name hints
    if (/spmv|sparse|csr|coo|ell|matrix.?vec|mv_/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests sparse operation');
    }

    // 11. Parameter names suggest sparse format
    const sparseParams = kernel.parameters.filter(p =>
      /row_?ptr|col_?idx|col_?ind|values?|nnz|row_?start|row_?end/i.test(p.name)
    );
    if (sparseParams.length >= 2) {
      addEvidence(evidence, 'sparse_params', 0.15,
        `Sparse format parameters: ${sparseParams.map(p => p.name).join(', ')}`);
    }

    // === Negative Indicators ===

    // Dense nested loops suggest GEMM
    const hasTripleNest = kernel.loops.filter(l => l.nestLevel >= 2).length > 0;
    if (hasTripleNest && !indirectInfo.found) {
      addEvidence(evidence, 'dense_loops', -0.25,
        'Triple-nested loops without indirect access (likely GEMM)');
    }

    // Stride halving suggests reduction
    if (kernel.loops.some(l => l.hasStrideHalving)) {
      addEvidence(evidence, 'stride_halving', -0.15,
        'Stride halving pattern (likely reduction)');
    }

    // Neighbor offsets suggest stencil
    const neighborAccesses = kernel.memoryAccesses.filter(a => a.hasNeighborOffset);
    if (neighborAccesses.length > 2 && !indirectInfo.found) {
      addEvidence(evidence, 'neighbor_access', -0.20,
        'Neighbor access without indirection (likely stencil)');
    }

    // Atomic increment suggests histogram
    if (/atomicAdd\s*\([^,]+,\s*1\s*\)/.test(source) && !indirectInfo.found) {
      addEvidence(evidence, 'atomic_count', -0.15,
        'Atomic counting without indirect access (likely histogram)');
    }

    const match = createPatternMatch('sparse', evidence, warnings);

    // Determine variant
    if (match.confidence > 0.3) {
      match.variant = this.determineVariant(csrInfo, cooInfo, ellInfo);
    }

    return match;
  }

  /**
   * Detect CSR format arrays (row_ptr, col_idx, values)
   */
  private detectCSRFormat(kernel: CudaKernelInfo, source: string): { found: boolean; arrays: string[] } {
    const arrays: string[] = [];

    // Look for row pointer parameter
    const hasRowPtr = kernel.parameters.some(p =>
      /row_?ptr|row_?off|row_?start|csrRowPtr/i.test(p.name)
    );
    if (hasRowPtr) arrays.push('row_ptr');

    // Look for column index parameter
    const hasColIdx = kernel.parameters.some(p =>
      /col_?idx|col_?ind|csrColInd/i.test(p.name)
    );
    if (hasColIdx) arrays.push('col_idx');

    // Look for values parameter
    const hasValues = kernel.parameters.some(p =>
      /values?|data|csrVal|nonzero/i.test(p.name)
    );
    if (hasValues) arrays.push('values');

    // Also detect via access patterns
    // row_ptr[row] and row_ptr[row + 1] pattern
    if (/\w+\s*\[\s*row\s*\][\s\S]{0,50}\w+\s*\[\s*row\s*\+\s*1\s*\]/.test(source)) {
      if (!arrays.includes('row_ptr')) arrays.push('row_ptr');
    }

    return { found: arrays.length >= 2, arrays };
  }

  /**
   * Detect row pointer iteration pattern
   */
  private detectRowPtrIteration(source: string): { found: boolean; pattern: string } {
    // Pattern: for(j = row_ptr[row]; j < row_ptr[row+1]; j++)
    if (/for\s*\([^)]*=\s*\w+\s*\[\s*\w+\s*\][^)]*<\s*\w+\s*\[\s*\w+\s*\+\s*1\s*\]/.test(source)) {
      return { found: true, pattern: 'standard row_ptr loop' };
    }

    // Pattern: row_start = row_ptr[row]; row_end = row_ptr[row+1]
    if (/\w+\s*=\s*\w+\s*\[\s*\w+\s*\][\s\S]{0,100}\w+\s*=\s*\w+\s*\[\s*\w+\s*\+\s*1\s*\]/.test(source)) {
      return { found: true, pattern: 'cached row bounds' };
    }

    // Pattern: while(j < row_ptr[row + 1])
    if (/while\s*\([^)]*<\s*\w+\s*\[\s*\w+\s*\+\s*1\s*\]/.test(source)) {
      return { found: true, pattern: 'while loop iteration' };
    }

    return { found: false, pattern: '' };
  }

  /**
   * Detect indirect array access (x[col_idx[j]])
   */
  private detectIndirectAccess(kernel: CudaKernelInfo, source: string): { found: boolean; count: number } {
    // Pattern: array[index_array[i]]
    const indirectPattern = /\w+\s*\[\s*\w+\s*\[\s*\w+\s*\]\s*\]/g;
    const matches = source.match(indirectPattern) || [];

    // Also check AST for indirect classification
    const indirectFromAST = kernel.memoryAccesses.filter(a => {
      // Index contains array access
      return /\w+\s*\[/.test(a.indexExpression);
    });

    const totalCount = new Set([
      ...matches,
      ...indirectFromAST.map(a => a.indexExpression)
    ]).size;

    return { found: totalCount > 0, count: totalCount };
  }

  /**
   * Detect COO format (coordinate format)
   */
  private detectCOOFormat(kernel: CudaKernelInfo, source: string): { found: boolean } {
    // COO has separate row and column index arrays
    const hasRowInd = kernel.parameters.some(p =>
      /row_?ind|row_?idx|cooRowInd/i.test(p.name)
    );
    const hasColInd = kernel.parameters.some(p =>
      /col_?ind|col_?idx|cooColInd/i.test(p.name)
    );

    // Access pattern: row_ind[i], col_ind[i] with same index
    const hasSameIndexAccess = /\w+\s*\[\s*(\w+)\s*\][\s\S]{0,100}\w+\s*\[\s*\1\s*\]/.test(source);

    return { found: hasRowInd && hasColInd && hasSameIndexAccess };
  }

  /**
   * Detect ELL format (ELLPACK)
   */
  private detectELLFormat(source: string): { found: boolean } {
    // ELL format characteristics:
    // 1. Fixed stride access: data[row + col * num_rows]
    const hasEllAccess = /\w+\s*\[\s*\w+\s*\+\s*\w+\s*\*\s*(?:num_?rows|n_?rows|rows)/i.test(source);

    // 2. Column-major layout with fixed elements per row
    const hasColumnMajor = /\w+\s*\[\s*\w+\s*\*\s*(?:max_?nnz|ell_?width|num_?cols)/i.test(source);

    // 3. Zero-padding check (skip if padded value)
    const hasPaddingCheck = /if\s*\([^)]*(?:!=|==)\s*(?:-1|0xFFFFFFFF|INVALID)/i.test(source);

    return { found: hasEllAccess || (hasColumnMajor && hasPaddingCheck) };
  }

  /**
   * Detect inner product accumulation
   */
  private detectInnerProduct(source: string): boolean {
    // Pattern: sum += val * x[col]
    const innerProductPattern =
      /\w+\s*\+=\s*\w+(?:\s*\[\s*\w+\s*\])?\s*\*\s*\w+\s*\[/;
    return innerProductPattern.test(source);
  }

  /**
   * Detect thread mapping strategy
   */
  private detectThreadMapping(source: string): 'thread_per_row' | 'warp_per_row' | 'none' {
    // Thread per row: row = threadIdx.x + blockIdx.x * blockDim.x
    const threadPerRow = /row\s*=\s*(?:threadIdx|blockIdx)/.test(source) ||
      /(?:threadIdx|blockIdx)[^;]+row/i.test(source);

    // Warp per row: row = (tid / 32) or row = (tid / warpSize)
    const warpPerRow = /row\s*=.*(?:\/\s*32|\/\s*warpSize|>>\s*5)/.test(source) ||
      /lane\s*=.*(?:%\s*32|&\s*31|%\s*warpSize)/.test(source);

    if (warpPerRow) return 'warp_per_row';
    if (threadPerRow) return 'thread_per_row';
    return 'none';
  }

  /**
   * Detect variable-length loop (data-dependent bounds)
   */
  private hasVariableLengthLoop(source: string): boolean {
    // Loop bound from array access
    return /for\s*\([^)]*<\s*\w+\s*\[\s*\w+\s*(?:\+\s*1)?\s*\]/.test(source);
  }

  /**
   * Detect segmented reduction using warp shuffle
   */
  private detectSegmentedReduction(source: string): boolean {
    const hasWarpShuffle = /__shfl_down|__shfl_xor|__reduce_\w+_sync/.test(source);
    const hasSegmentMask = /lane_?mask|segment_?end|row_?end|__ballot/.test(source);

    return hasWarpShuffle && hasSegmentMask;
  }

  /**
   * Determine the sparse matrix variant
   */
  private determineVariant(
    csrInfo: { found: boolean; arrays: string[] },
    cooInfo: { found: boolean },
    ellInfo: { found: boolean }
  ): PatternVariant {
    if (csrInfo.found && csrInfo.arrays.length >= 2) {
      return 'spmv_csr';
    }
    if (cooInfo.found) {
      return 'spmv_coo';
    }
    if (ellInfo.found) {
      return 'spmv_ell';
    }
    return 'spmv_csr'; // default to CSR
  }
}

export const sparseMatcher = new SparseMatcher();
