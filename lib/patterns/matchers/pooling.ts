/**
 * Pooling Pattern Matcher
 * Detects max pooling, average pooling, and global pooling patterns
 */

import { CudaKernelInfo, PatternMatch, Evidence, PatternVariant } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class PoolingMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;

    // === Primary Indicators (high weight) ===

    // 1. Max pooling: nested loops with max operation
    const maxPoolInfo = this.detectMaxPooling(source);
    if (maxPoolInfo.found) {
      addEvidence(evidence, 'max_pool_pattern', 0.35,
        `Max pooling: ${maxPoolInfo.type}`);
    }

    // 2. Average pooling: sum and divide
    const avgPoolInfo = this.detectAvgPooling(source);
    if (avgPoolInfo.found) {
      addEvidence(evidence, 'avg_pool_pattern', 0.35,
        `Average pooling: ${avgPoolInfo.type}`);
    }

    // 3. Global pooling: reduce entire spatial dimension
    const globalPoolInfo = this.detectGlobalPooling(source, kernel);
    if (globalPoolInfo.found) {
      addEvidence(evidence, 'global_pool_pattern', 0.30,
        'Global pooling detected');
    }

    // === Secondary Indicators (medium weight) ===

    // 4. Window-based iteration (pool_size loops)
    const hasWindowLoop = this.detectWindowLoop(source);
    if (hasWindowLoop) {
      addEvidence(evidence, 'window_loop', 0.20,
        'Pooling window iteration');
    }

    // 5. Stride-based output indexing
    const hasStrideOutput = /\w+\s*\*\s*stride|\/\s*stride|out_\w+\s*\*\s*\d+/.test(source);
    if (hasStrideOutput) {
      addEvidence(evidence, 'stride_output', 0.10,
        'Stride-based output mapping');
    }

    // 6. Bounds checking for padding
    const hasBoundsCheck = /if\s*\([^)]*[<>]=?\s*0|if\s*\([^)]*[<>]=?\s*\w*_?[hw]/.test(source);
    if (hasBoundsCheck && hasWindowLoop) {
      addEvidence(evidence, 'padding_check', 0.10,
        'Padding bounds checking');
    }

    // 7. Comparison with -inf or very negative value (max pool init)
    const hasNegInfInit = /-inf|FLT_MIN|-1e38|-3\.4e38|FLOAT_MIN/i.test(source);
    if (hasNegInfInit) {
      addEvidence(evidence, 'neg_inf_init', 0.15,
        'Negative infinity initialization (max pool)');
    }

    // 8. Name hints
    if (/pool|maxpool|avgpool|global_?pool/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests pooling');
    }

    // 9. Parameter names
    const poolParams = kernel.parameters.filter(p =>
      /pool_?size|kernel_?size|stride|pad/i.test(p.name)
    );
    if (poolParams.length >= 2) {
      addEvidence(evidence, 'pool_params', 0.10,
        'Pooling parameters detected');
    }

    // 10. Adaptive pooling (output size specified)
    const isAdaptive = /adaptive|out_h\s*\/|out_w\s*\//i.test(source);
    if (isAdaptive) {
      addEvidence(evidence, 'adaptive_pool', 0.15,
        'Adaptive pooling detected');
    }

    // === Negative Indicators ===

    // Convolution patterns (filter weights)
    if (/weight|filter|kernel\s*\[/.test(source) && !/pool/i.test(source)) {
      addEvidence(evidence, 'conv_pattern', -0.20,
        'Weight access suggests convolution');
    }

    // Matmul patterns
    if (/\[\s*\w+\s*\]\s*\[\s*\w+\s*\]\s*\*\s*\[\s*\w+\s*\]\s*\[\s*\w+\s*\]/.test(source)) {
      addEvidence(evidence, 'matmul_pattern', -0.25,
        'Matrix multiplication pattern');
    }

    const match = createPatternMatch('pooling' as any, evidence, warnings);

    if (match.confidence > 0.3) {
      match.variant = this.determineVariant(maxPoolInfo, avgPoolInfo, globalPoolInfo, isAdaptive);
    }

    return match;
  }

  /**
   * Detect max pooling pattern
   */
  private detectMaxPooling(source: string): { found: boolean; type: string } {
    // Explicit max function in loop
    if (/max_val\s*=\s*(?:fmax|max)\s*\(|max_val\s*=\s*\w+\s*>\s*max_val\s*\?/.test(source)) {
      return { found: true, type: 'explicit max' };
    }

    // Conditional max update
    if (/if\s*\([^)]*>\s*max_val|if\s*\(\s*\w+\s*>\s*\w+\s*\)/.test(source) &&
        /pool|max_val|maximum/i.test(source)) {
      return { found: true, type: 'conditional max' };
    }

    // fmaxf intrinsic
    if (/fmaxf\s*\(/.test(source)) {
      return { found: true, type: 'fmaxf intrinsic' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect average pooling pattern
   */
  private detectAvgPooling(source: string): { found: boolean; type: string } {
    // Sum followed by division
    if (/sum\s*\+=|sum_val\s*\+=/.test(source) && /\/\s*(?:pool_?size|count|area|num)/i.test(source)) {
      return { found: true, type: 'sum and divide' };
    }

    // Accumulate then normalize
    if (/\+=\s*\w+\s*\[/.test(source) && /\*\s*(?:1\.0|inv).*(?:pool|count)/i.test(source)) {
      return { found: true, type: 'accumulate and normalize' };
    }

    // Pre-computed weight
    if (/weight\s*=\s*1\.0\s*\/|scale\s*=\s*1\.0\s*\//.test(source)) {
      return { found: true, type: 'pre-computed scale' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect global pooling pattern
   */
  private detectGlobalPooling(source: string, kernel: CudaKernelInfo): { found: boolean } {
    // Reduce entire spatial dimension
    const hasFullReduce = /for\s*\([^)]*0[^)]*h\s*\)|for\s*\([^)]*0[^)]*w\s*\)/.test(source);
    const hasGlobalHint = /global|gap|gmp/i.test(kernel.name);

    if (hasFullReduce && hasGlobalHint) {
      return { found: true };
    }

    // Single output per channel
    if (/out_h\s*==\s*1|out_w\s*==\s*1|output_size\s*==\s*1/i.test(source)) {
      return { found: true };
    }

    return { found: false };
  }

  /**
   * Detect pooling window loop
   */
  private detectWindowLoop(source: string): boolean {
    // Nested loops for pool window
    const patterns = [
      /for\s*\([^)]*ph|for\s*\([^)]*pw/i,
      /for\s*\([^)]*kh|for\s*\([^)]*kw/i,
      /for\s*\([^)]*pool_h|for\s*\([^)]*pool_w/i,
      /for\s*\([^)]*i\s*=\s*0[^)]*pool_size/i,
    ];

    return patterns.some(p => p.test(source));
  }

  /**
   * Determine pooling variant
   */
  private determineVariant(
    maxPoolInfo: { found: boolean; type: string },
    avgPoolInfo: { found: boolean; type: string },
    globalPoolInfo: { found: boolean },
    isAdaptive: boolean
  ): PatternVariant {
    if (globalPoolInfo.found) {
      if (maxPoolInfo.found) return 'global_max_pool' as PatternVariant;
      if (avgPoolInfo.found) return 'global_avg_pool' as PatternVariant;
      return 'global_avg_pool' as PatternVariant;
    }

    if (isAdaptive) {
      if (maxPoolInfo.found) return 'adaptive_max_pool' as PatternVariant;
      if (avgPoolInfo.found) return 'adaptive_avg_pool' as PatternVariant;
    }

    if (maxPoolInfo.found) return 'max_pool_2d' as PatternVariant;
    if (avgPoolInfo.found) return 'avg_pool_2d' as PatternVariant;

    return 'max_pool_2d' as PatternVariant; // default
  }
}

export const poolingMatcher = new PoolingMatcher();
