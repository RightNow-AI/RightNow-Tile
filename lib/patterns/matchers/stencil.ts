// Stencil Pattern Matcher
// Detects neighbor-based computations (convolution, Jacobi, etc.)
// Supports variants: 1D (3pt, 5pt), 2D (5pt, 9pt), 3D

import { CudaKernelInfo, PatternMatch, Evidence, PatternVariant } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

interface StencilInfo {
  dimensions: 1 | 2 | 3;
  pointCount: number;
  offsets: { x: number[]; y: number[]; z: number[] };
  isCross: boolean; // Cross/plus pattern (5-point 2D)
  isBox: boolean;   // Box pattern (9-point 2D)
}

export class StencilMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;

    // Detailed stencil analysis
    const stencilInfo = this.analyzeStencilStructure(kernel, source);

    // === Primary Indicators (high weight) ===

    // 1. Neighbor access patterns (arr[i-1], arr[i+1], etc.) - Critical!
    const neighborAccesses = this.findNeighborAccesses(kernel);
    if (neighborAccesses.count >= 3) {
      addEvidence(evidence, 'neighbor_access_multiple', 0.40,
        `${neighborAccesses.count} neighbor offset accesses detected (${neighborAccesses.dimensions}D stencil)`);
    } else if (neighborAccesses.count >= 2) {
      addEvidence(evidence, 'neighbor_access_pair', 0.25,
        `${neighborAccesses.count} neighbor accesses detected`);
    } else if (neighborAccesses.count === 1) {
      addEvidence(evidence, 'single_neighbor', 0.10,
        'Single neighbor access (partial stencil)');
    }

    // 2. Symmetric access pairs ([i-1] AND [i+1])
    const symmetricPairs = this.countSymmetricPairs(kernel);
    if (symmetricPairs > 0) {
      addEvidence(evidence, 'symmetric_access', 0.20,
        `${symmetricPairs} symmetric neighbor pair(s) detected`);
    }

    // 3. Boundary checking (halo/ghost cell handling)
    const hasBoundaryChecks = this.detectBoundaryChecks(source);
    if (hasBoundaryChecks) {
      addEvidence(evidence, 'boundary_check', 0.15,
        'Boundary/halo region checking detected');
    }

    // === Secondary Indicators (medium weight) ===

    // 4. 2D/3D thread indexing (common for stencils)
    if (kernel.threadIndexUsage.usesThreadIdxY || kernel.threadIndexUsage.usesBlockIdxY) {
      addEvidence(evidence, 'multidim_indexing', 0.10,
        'Multi-dimensional thread indexing');
    }

    // 5. Shared memory for halo caching
    if (kernel.sharedMemoryDecls.length > 0) {
      const hasHaloLoading = this.detectHaloLoading(source);
      if (hasHaloLoading) {
        addEvidence(evidence, 'halo_caching', 0.10,
          'Shared memory halo caching detected');
      } else {
        addEvidence(evidence, 'shared_memory', 0.05,
          'Shared memory present');
      }
    }

    // 6. Averaging/weighted sum operation
    const hasAveraging = this.detectAveragingOperation(source);
    if (hasAveraging) {
      addEvidence(evidence, 'averaging_operation', 0.10,
        'Averaging/weighted sum operation detected');
    }

    // 7. Separate input and output arrays (out-of-place)
    const hasDistinctIO = this.detectDistinctInputOutput(kernel);
    if (hasDistinctIO) {
      addEvidence(evidence, 'distinct_io', 0.05,
        'Separate input and output arrays');
    }

    // 8. Name hints
    if (/stencil|jacobi|laplace|convolution|blur|filter|diffusion|heat/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests stencil operation');
    }

    // 9. Regular grid access (fixed offsets)
    const hasFixedOffsets = this.detectFixedOffsets(kernel);
    if (hasFixedOffsets) {
      addEvidence(evidence, 'fixed_offsets', 0.10,
        'Fixed/regular offset pattern (structured stencil)');
    }

    // 10. Width/height variables in index (2D grid)
    const has2DGridVars = /width|height|pitch|stride/i.test(source);
    if (has2DGridVars && neighborAccesses.count > 0) {
      addEvidence(evidence, '2d_grid_vars', 0.05,
        '2D grid dimension variables present');
    }

    // === Negative Indicators ===

    // Stride halving suggests reduction
    const hasStrideHalving = kernel.loops.some(l => l.hasStrideHalving);
    if (hasStrideHalving) {
      addEvidence(evidence, 'stride_halving', -0.20,
        'Stride halving pattern (likely reduction)');
    }

    // Matrix multiply pattern suggests GEMM
    const hasMatrixMultiply = /\w+\s*\+=\s*\w+\[[^\]]+\]\s*\*\s*\w+\[[^\]]+\]/.test(source);
    if (hasMatrixMultiply && kernel.loops.length >= 2) {
      addEvidence(evidence, 'matrix_multiply', -0.15,
        'Matrix multiply pattern (likely GEMM)');
    }

    // No neighbor access at all - major negative
    if (neighborAccesses.count === 0) {
      addEvidence(evidence, 'no_neighbor_access', -0.30,
        'No neighbor access patterns detected');
    }

    // Atomic operations unusual in stencils
    const atomicCount = kernel.syncPoints.filter(s => s.type === 'atomic').length;
    if (atomicCount > 0) {
      addEvidence(evidence, 'has_atomics', -0.10,
        'Atomic operations (unusual for stencil)');
      warnings.push('Atomics detected - verify this is a stencil');
    }

    const match = createPatternMatch('stencil', evidence, warnings);

    // Determine variant based on stencil structure
    if (match.confidence > 0.3) {
      match.variant = this.determineVariant(stencilInfo);
    }

    return match;
  }

  /**
   * Analyze the stencil structure to determine dimensions and point count
   */
  private analyzeStencilStructure(kernel: CudaKernelInfo, source: string): StencilInfo {
    const xOffsets: number[] = [];
    const yOffsets: number[] = [];
    const zOffsets: number[] = [];

    // Extract X offsets (1D patterns): [i-1], [i+1], [i-2], etc.
    const x1DPattern = /\[\s*\w+\s*([-+])\s*(\d+)\s*\]/g;
    let match;
    while ((match = x1DPattern.exec(source)) !== null) {
      const sign = match[1] === '-' ? -1 : 1;
      const offset = sign * parseInt(match[2]);
      if (!xOffsets.includes(offset)) xOffsets.push(offset);
    }

    // Extract Y offsets (2D patterns): [i - width], [i + width], etc.
    const y2DPattern = /\[\s*\w+\s*([-+])\s*(?:width|w|pitch|stride|cols?|nx)\s*\]/gi;
    while ((match = y2DPattern.exec(source)) !== null) {
      const sign = match[1] === '-' ? -1 : 1;
      if (!yOffsets.includes(sign)) yOffsets.push(sign);
    }

    // Also check for 2D array access: arr[y-1][x] or arr[row-1][col]
    const y2DArrayPattern = /\[\s*\w+\s*([-+])\s*1\s*\]\s*\[/g;
    while ((match = y2DArrayPattern.exec(source)) !== null) {
      const sign = match[1] === '-' ? -1 : 1;
      if (!yOffsets.includes(sign)) yOffsets.push(sign);
    }

    // Extract Z offsets (3D): [i - width*height], arr[z-1][y][x]
    const z3DPattern = /\[\s*\w+\s*([-+])\s*(?:slice|depth|nz|width\s*\*\s*height)\s*\]/gi;
    while ((match = z3DPattern.exec(source)) !== null) {
      const sign = match[1] === '-' ? -1 : 1;
      if (!zOffsets.includes(sign)) zOffsets.push(sign);
    }

    // Check for 3D array access
    const z3DArrayPattern = /\[\s*\w+\s*([-+])\s*1\s*\]\s*\[\s*\w+\s*\]\s*\[/g;
    while ((match = z3DArrayPattern.exec(source)) !== null) {
      const sign = match[1] === '-' ? -1 : 1;
      if (!zOffsets.includes(sign)) zOffsets.push(sign);
    }

    // Determine dimensions
    let dimensions: 1 | 2 | 3 = 1;
    if (zOffsets.length > 0) {
      dimensions = 3;
    } else if (yOffsets.length > 0) {
      dimensions = 2;
    }

    // Count points (unique access positions)
    let pointCount = xOffsets.length;
    if (dimensions >= 2) {
      pointCount = xOffsets.length + yOffsets.length;
    }
    if (dimensions === 3) {
      pointCount += zOffsets.length;
    }

    // Include center point if not explicitly counted
    if (!xOffsets.includes(0) && pointCount > 0) {
      pointCount += 1;
    }

    // Determine pattern type for 2D
    const isCross = dimensions === 2 &&
      xOffsets.length <= 2 && yOffsets.length <= 2 &&
      !this.hasCornerAccess(source);

    const isBox = dimensions === 2 && this.hasCornerAccess(source);

    return {
      dimensions,
      pointCount: Math.max(pointCount, kernel.memoryAccesses.filter(a => a.hasNeighborOffset).length + 1),
      offsets: { x: xOffsets, y: yOffsets, z: zOffsets },
      isCross,
      isBox,
    };
  }

  /**
   * Check for corner access (indicates box/9-point stencil)
   */
  private hasCornerAccess(source: string): boolean {
    // Corner patterns: [i-1-width], [i+1+width], etc.
    const cornerPattern = /\[\s*\w+\s*[-+]\s*(?:1|width|w)\s*[-+]\s*(?:1|width|w)\s*\]/i;
    return cornerPattern.test(source);
  }

  /**
   * Determine stencil variant based on analysis
   */
  private determineVariant(info: StencilInfo): PatternVariant {
    if (info.dimensions === 3) {
      return 'stencil_3d';
    }

    if (info.dimensions === 2) {
      if (info.isBox || info.pointCount >= 9) {
        return 'stencil_2d_9pt';
      }
      return 'stencil_2d_5pt';
    }

    // 1D variants
    if (info.pointCount >= 5) {
      return 'stencil_1d_5pt';
    }
    return 'stencil_1d_3pt';
  }

  private findNeighborAccesses(kernel: CudaKernelInfo): { count: number; dimensions: number } {
    const neighborAccesses = kernel.memoryAccesses.filter(a => a.hasNeighborOffset);

    // Determine dimensionality by checking for 2D offsets
    const source = kernel.sourceText;
    let dimensions = 1;

    // Check for 2D patterns: [idx - width], [idx + width]
    if (/\[\s*\w+\s*[-+]\s*width\s*\]/i.test(source) ||
        /\[\s*\w+\s*[-+]\s*\w+\s*\]\s*\[\s*\w+\s*\]/i.test(source)) {
      dimensions = 2;
    }

    // Check for 3D patterns
    if (/\[\s*\w+\s*\]\s*\[\s*\w+\s*\]\s*\[\s*\w+\s*[-+]\s*1\s*\]/.test(source)) {
      dimensions = 3;
    }

    return { count: neighborAccesses.length, dimensions };
  }

  private countSymmetricPairs(kernel: CudaKernelInfo): number {
    let pairs = 0;
    const accesses = kernel.memoryAccesses.filter(a => a.hasNeighborOffset && a.offset !== undefined);

    const seen = new Set<string>();

    for (const access of accesses) {
      if (access.offset === undefined) continue;

      const key = `${access.array}:${Math.abs(access.offset)}`;
      const oppositeOffset = -access.offset;

      // Check if we have the opposite offset for same array
      const hasOpposite = accesses.some(a =>
        a.array === access.array &&
        a.offset === oppositeOffset
      );

      if (hasOpposite && !seen.has(key)) {
        pairs++;
        seen.add(key);
      }
    }

    return pairs;
  }

  private detectBoundaryChecks(source: string): boolean {
    // Common boundary check patterns
    const patterns = [
      /if\s*\(\s*\w+\s*>\s*0\s*&&\s*\w+\s*</,            // if (x > 0 && x < ...)
      /if\s*\(\s*\w+\s*>=\s*1\s*&&\s*\w+\s*</,           // if (x >= 1 && x < ...)
      /if\s*\(\s*\w+\s*<\s*\w+\s*-\s*1\s*\)/,            // if (x < width - 1)
      /\w+\s*>\s*0\s*\?\s*\w+\s*\[[^\]]+\-/,             // ternary boundary check
      /clamped|clamp\s*\(/i,                              // clamp function
      /min\s*\(\s*max\s*\(/,                              // min(max(...)) clamping
      /boundary|border|edge|halo/i,                       // Variable names
    ];

    return patterns.some(p => p.test(source));
  }

  private detectHaloLoading(source: string): boolean {
    // Halo loading patterns
    const patterns = [
      /halo/i,
      /ghost/i,
      /apron/i,
      // Loading more than block size into shared memory
      /\+\s*\d+\s*\]\s*=.*\[.*blockIdx/,
      // Explicit halo region loading
      /threadIdx\.x\s*[<>=]+\s*\d+\s*\|\|/,
    ];

    return patterns.some(p => p.test(source));
  }

  private detectAveragingOperation(source: string): boolean {
    // Averaging patterns
    const patterns = [
      /\/\s*[4589]\.0?f?(?!\d)/,                         // Division by common stencil counts
      /\*\s*0\.25/,                                       // Multiply by 1/4
      /\*\s*0\.2(?:0|5)/,                                // Multiply by 1/5 or 1/4
      /\+[^;]+\+[^;]+\+[^;]+\)\s*[*/]/,                  // Sum then divide/multiply
      /average|mean|avg/i,                                // Variable names
    ];

    return patterns.some(p => p.test(source));
  }

  private detectDistinctInputOutput(kernel: CudaKernelInfo): boolean {
    const readArrays = new Set(
      kernel.memoryAccesses
        .filter(a => a.accessType === 'read')
        .map(a => a.array)
    );

    const writeArrays = new Set(
      kernel.memoryAccesses
        .filter(a => a.accessType === 'write')
        .map(a => a.array)
    );

    // Check if any write array is different from all read arrays
    for (const writeArr of writeArrays) {
      if (!readArrays.has(writeArr)) {
        return true;
      }
    }

    // Also check for common in/out naming patterns
    const source = kernel.sourceText.toLowerCase();
    return /\bin\b.*\bout\b|\binput\b.*\boutput\b|\bsrc\b.*\bdst\b/i.test(source);
  }

  private detectFixedOffsets(kernel: CudaKernelInfo): boolean {
    const offsets = kernel.memoryAccesses
      .filter(a => a.offset !== undefined)
      .map(a => Math.abs(a.offset!));

    // Fixed offsets are typically small integers (1, 2, width, etc.)
    const hasSmallOffsets = offsets.some(o => o >= 1 && o <= 10);

    // Check for consistent offset patterns
    const uniqueOffsets = new Set(offsets);
    const isRegular = uniqueOffsets.size <= 5; // Few unique offsets = structured stencil

    return hasSmallOffsets && isRegular;
  }
}

export const stencilMatcher = new StencilMatcher();
