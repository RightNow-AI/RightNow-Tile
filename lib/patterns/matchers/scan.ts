// Scan (Prefix Sum) Pattern Matcher
// Detects parallel scan algorithms (inclusive/exclusive)

import { CudaKernelInfo, PatternMatch, Evidence } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class ScanMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;

    // === Primary Indicators (high weight) ===

    // 1. Both stride doubling AND halving (up-sweep + down-sweep) - Critical!
    const hasStrideDoubling = kernel.loops.some(l => l.hasStrideDoubling);
    const hasStrideHalving = kernel.loops.some(l => l.hasStrideHalving);

    if (hasStrideDoubling && hasStrideHalving) {
      addEvidence(evidence, 'dual_phase_stride', 0.40,
        'Both up-sweep (stride*=2) and down-sweep (stride/=2) phases detected');
    } else if (hasStrideDoubling) {
      addEvidence(evidence, 'stride_doubling', 0.15,
        'Stride doubling pattern (possible up-sweep phase)');
    } else if (hasStrideHalving) {
      addEvidence(evidence, 'stride_halving_only', 0.10,
        'Stride halving only (may be reduction or partial scan)');
    }

    // 2. Complex index arithmetic characteristic of scan
    const hasScanIndexPattern = this.detectScanIndexPattern(source);
    if (hasScanIndexPattern) {
      addEvidence(evidence, 'scan_index_arithmetic', 0.20,
        'Scan index pattern: (tid+1)*stride*2-1 or similar');
    }

    // 3. Multiple __syncthreads (typically 3+ for scan)
    const syncCount = kernel.syncPoints.filter(s => s.type === 'syncthreads').length;
    if (syncCount >= 3) {
      addEvidence(evidence, 'multi_sync', 0.15,
        `${syncCount} __syncthreads calls (expected for scan)`);
    } else if (syncCount === 2) {
      addEvidence(evidence, 'dual_sync', 0.10,
        '2 __syncthreads calls');
    }

    // === Secondary Indicators (medium weight) ===

    // 4. Shared memory (required for efficient scan)
    if (kernel.sharedMemoryDecls.length > 0) {
      addEvidence(evidence, 'shared_memory', 0.10,
        'Shared memory for scan buffer');
    }

    // 5. Power-of-2 operations in index calculations
    const powerOf2Ops = this.countPowerOf2Operations(source);
    if (powerOf2Ops >= 2) {
      addEvidence(evidence, 'power_of_2_arithmetic', 0.10,
        `${powerOf2Ops} power-of-2 operations in indices`);
    }

    // 6. Bank conflict avoidance pattern
    const hasBankConflictAvoid = this.detectBankConflictPattern(source);
    if (hasBankConflictAvoid) {
      addEvidence(evidence, 'bank_conflict_avoidance', 0.10,
        'Bank conflict avoidance pattern detected');
    }

    // 7. Exclusive scan initialization (output[0] = 0 at end)
    const hasExclusiveInit = this.detectExclusiveScanInit(source);
    if (hasExclusiveInit) {
      addEvidence(evidence, 'exclusive_scan_init', 0.10,
        'Exclusive scan initialization detected');
    }

    // 8. Name hints
    if (/scan|prefix|cumsum|cumulative|prescan/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests scan operation');
    }

    // 9. Two distinct loop phases
    if (kernel.loops.length >= 2) {
      const doublingLoops = kernel.loops.filter(l => l.hasStrideDoubling).length;
      const halvingLoops = kernel.loops.filter(l => l.hasStrideHalving).length;
      if (doublingLoops >= 1 && halvingLoops >= 1) {
        addEvidence(evidence, 'two_phase_loops', 0.10,
          'Two distinct phase loops detected');
      }
    }

    // === Negative Indicators ===

    // Atomic operations suggest reduction, not scan
    const atomicCount = kernel.syncPoints.filter(s => s.type === 'atomic').length;
    if (atomicCount > 0) {
      addEvidence(evidence, 'has_atomics', -0.15,
        'Atomic operations present (typical for reduction, not scan)');
      warnings.push('Atomics unusual in scan - may be block-level reduction');
    }

    // Simple linear access suggests elementwise
    const linearAccesses = kernel.memoryAccesses.filter(a =>
      !a.hasNeighborOffset && !a.indexExpression.includes('*')
    );
    if (linearAccesses.length === kernel.memoryAccesses.length && kernel.memoryAccesses.length > 0) {
      addEvidence(evidence, 'only_linear_access', -0.20,
        'Only linear access patterns (likely elementwise)');
    }

    // No stride patterns at all
    if (!hasStrideDoubling && !hasStrideHalving) {
      addEvidence(evidence, 'no_stride_patterns', -0.25,
        'No stride patterns detected');
    }

    return createPatternMatch('scan', evidence, warnings);
  }

  private detectScanIndexPattern(source: string): boolean {
    // Scan patterns typically have complex index expressions like:
    // (tid + 1) * stride * 2 - 1
    // tid * 2 * stride + stride - 1
    // index - stride

    const patterns = [
      /\(\s*\w+\s*\+\s*1\s*\)\s*\*\s*\w+\s*\*\s*2/,  // (tid + 1) * stride * 2
      /\w+\s*\*\s*2\s*\*\s*\w+\s*[-+]/,              // tid * 2 * stride +/-
      /index\s*[-+]\s*stride/i,                       // index +/- stride
      /\w+\s*[-+]\s*stride\s*\]/,                    // [...- stride]
      /2\s*\*\s*\w+\s*\*\s*\(/,                      // 2 * stride * (
    ];

    return patterns.some(p => p.test(source));
  }

  private countPowerOf2Operations(source: string): number {
    let count = 0;

    // Count left/right shifts
    const shifts = source.match(/<<|>>/g);
    if (shifts) count += shifts.length;

    // Count *2 and /2 operations
    const mult2 = source.match(/\*\s*2(?!\d)/g);
    if (mult2) count += mult2.length;

    const div2 = source.match(/\/\s*2(?!\d)/g);
    if (div2) count += div2.length;

    return count;
  }

  private detectBankConflictPattern(source: string): boolean {
    // Bank conflict avoidance: index + (index >> LOG_NUM_BANKS)
    // or: index + index / NUM_BANKS

    const patterns = [
      /\w+\s*\+\s*\(\s*\w+\s*>>\s*\d+\s*\)/,        // idx + (idx >> 5)
      /\w+\s*\+\s*\w+\s*\/\s*\d+/,                   // idx + idx / 32
      /CONFLICT_FREE_OFFSET/i,                        // Macro name
      /NUM_BANKS|LOG_NUM_BANKS/i,                    // Bank constants
    ];

    return patterns.some(p => p.test(source));
  }

  private detectExclusiveScanInit(source: string): boolean {
    // Exclusive scan typically sets last element to 0 before down-sweep
    // or first element to 0 in output

    const patterns = [
      /\w+\s*\[\s*\w+\s*-\s*1\s*\]\s*=\s*0/,        // arr[n-1] = 0
      /\w+\s*\[\s*0\s*\]\s*=\s*0.*output/i,          // arr[0] = 0 (output)
      /temp\s*=.*;\s*\w+\s*\[.*\]\s*=\s*\w+\s*\[.*\];\s*\w+\s*\[.*\]\s*=\s*temp/,  // swap pattern
    ];

    return patterns.some(p => p.test(source));
  }
}

export const scanMatcher = new ScanMatcher();
