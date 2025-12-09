/**
 * Normalization Pattern Matcher
 * Detects LayerNorm, BatchNorm, GroupNorm, InstanceNorm, and RMSNorm patterns
 */

import { CudaKernelInfo, PatternMatch, Evidence, PatternVariant } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class NormalizationMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;

    // === Primary Indicators (high weight) ===

    // 1. Mean computation (sum / count)
    const meanInfo = this.detectMeanComputation(source);
    if (meanInfo.found) {
      addEvidence(evidence, 'mean_computation', 0.25,
        `Mean computation: ${meanInfo.type}`);
    }

    // 2. Variance computation (sum of squared differences)
    const varInfo = this.detectVarianceComputation(source);
    if (varInfo.found) {
      addEvidence(evidence, 'variance_computation', 0.25,
        `Variance computation: ${varInfo.type}`);
    }

    // 3. Normalization: (x - mean) / sqrt(var + eps)
    const normInfo = this.detectNormalization(source);
    if (normInfo.found) {
      addEvidence(evidence, 'normalize_pattern', 0.30,
        'Normalization formula detected');
    }

    // 4. Affine transform: gamma * normalized + beta
    const affineInfo = this.detectAffineTransform(source);
    if (affineInfo.found) {
      addEvidence(evidence, 'affine_transform', 0.15,
        'Affine transform (scale/shift)');
    }

    // === Normalization Type Detection ===

    // 5. LayerNorm: normalize over hidden/feature dimension
    const isLayerNorm = this.detectLayerNorm(source, kernel);
    if (isLayerNorm) {
      addEvidence(evidence, 'layernorm_pattern', 0.20,
        'LayerNorm pattern (normalize over features)');
    }

    // 6. BatchNorm: normalize over batch dimension
    const isBatchNorm = this.detectBatchNorm(source, kernel);
    if (isBatchNorm) {
      addEvidence(evidence, 'batchnorm_pattern', 0.20,
        'BatchNorm pattern (normalize over batch)');
    }

    // 7. GroupNorm: normalize over grouped channels
    const isGroupNorm = this.detectGroupNorm(source, kernel);
    if (isGroupNorm) {
      addEvidence(evidence, 'groupnorm_pattern', 0.20,
        'GroupNorm pattern');
    }

    // 8. InstanceNorm: per-sample, per-channel normalization
    const isInstanceNorm = this.detectInstanceNorm(source, kernel);
    if (isInstanceNorm) {
      addEvidence(evidence, 'instancenorm_pattern', 0.20,
        'InstanceNorm pattern');
    }

    // 9. RMSNorm: root mean square normalization (no mean subtraction)
    const isRMSNorm = this.detectRMSNorm(source);
    if (isRMSNorm) {
      addEvidence(evidence, 'rmsnorm_pattern', 0.25,
        'RMSNorm pattern (no mean subtraction)');
    }

    // === Secondary Indicators ===

    // 10. Epsilon usage (numerical stability)
    const hasEpsilon = /eps|epsilon|1e-[56]|1e-12|1e-8/i.test(source);
    if (hasEpsilon) {
      addEvidence(evidence, 'epsilon', 0.10,
        'Epsilon for numerical stability');
    }

    // 11. rsqrt intrinsic
    const hasRsqrt = /rsqrt|rsqrtf|__rsqrtf/.test(source);
    if (hasRsqrt) {
      addEvidence(evidence, 'rsqrt', 0.10,
        'Reciprocal square root');
    }

    // 12. Name hints
    if (/norm|layernorm|batchnorm|groupnorm|instancenorm|rmsnorm/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests normalization');
    }

    // 13. Gamma/beta parameters
    const hasGammaBeta = kernel.parameters.some(p =>
      /gamma|beta|scale|bias|weight/i.test(p.name)
    );
    if (hasGammaBeta) {
      addEvidence(evidence, 'gamma_beta_params', 0.10,
        'Scale/shift parameters');
    }

    // === Negative Indicators ===

    // Softmax pattern
    if (/exp\s*\([^)]*\)\s*\/\s*\w*sum/.test(source) && !meanInfo.found) {
      addEvidence(evidence, 'softmax_pattern', -0.20,
        'Exp/sum pattern (likely softmax)');
    }

    const match = createPatternMatch('normalization' as any, evidence, warnings);

    if (match.confidence > 0.3) {
      match.variant = this.determineVariant(
        isLayerNorm, isBatchNorm, isGroupNorm, isInstanceNorm, isRMSNorm
      );
    }

    return match;
  }

  /**
   * Detect mean computation
   */
  private detectMeanComputation(source: string): { found: boolean; type: string } {
    // Explicit mean calculation
    if (/mean\s*=\s*\w*sum\s*\/|mean\s*=\s*\w+\s*\*\s*(?:inv|1\.0\s*\/)/.test(source)) {
      return { found: true, type: 'explicit mean' };
    }

    // Sum accumulation for mean
    if (/sum\s*\+=|mean_sum\s*\+=/.test(source) && /\/\s*(?:n|size|count|dim)/i.test(source)) {
      return { found: true, type: 'sum and divide' };
    }

    // Welford's online mean
    if (/delta\s*=.*-\s*mean|mean\s*\+=.*delta\s*\//.test(source)) {
      return { found: true, type: 'Welford online' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect variance computation
   */
  private detectVarianceComputation(source: string): { found: boolean; type: string } {
    // Squared difference sum
    if (/\(\s*\w+\s*-\s*mean\s*\)\s*\*\s*\(\s*\w+\s*-\s*mean\s*\)|diff\s*\*\s*diff/.test(source)) {
      return { found: true, type: 'squared difference' };
    }

    // Variance accumulation
    if (/var\s*\+=|variance\s*\+=|var_sum\s*\+=/.test(source)) {
      return { found: true, type: 'variance accumulation' };
    }

    // E[x^2] - E[x]^2 form
    if (/mean_sq|sq_mean|sum_sq/.test(source) && /mean\s*\*\s*mean/.test(source)) {
      return { found: true, type: 'mean of squares formula' };
    }

    // Welford's online variance
    if (/M2\s*\+=|delta2\s*=/.test(source)) {
      return { found: true, type: 'Welford M2' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect normalization formula
   */
  private detectNormalization(source: string): { found: boolean } {
    // (x - mean) / sqrt(var + eps)
    const patterns = [
      /\(\s*\w+\s*-\s*mean\s*\)\s*\*\s*(?:rsqrt|rstd|inv_std)/,
      /\(\s*\w+\s*-\s*mean\s*\)\s*\/\s*(?:sqrt|std)/,
      /normalized\s*=\s*\w+\s*-\s*mean/,
      /\w+\s*\*\s*rsqrt\s*\(\s*var/,
    ];

    return { found: patterns.some(p => p.test(source)) };
  }

  /**
   * Detect affine transform
   */
  private detectAffineTransform(source: string): { found: boolean } {
    // gamma * x + beta
    const patterns = [
      /gamma\s*\*\s*\w+\s*\+\s*beta/,
      /scale\s*\*\s*\w+\s*\+\s*bias/,
      /weight\s*\*\s*normalized\s*\+/,
      /\w+\s*\*\s*\w+\s*\[\s*\w+\s*\]\s*\+\s*\w+\s*\[\s*\w+\s*\]/,
    ];

    return { found: patterns.some(p => p.test(source)) };
  }

  /**
   * Detect LayerNorm (normalize over last/feature dimension)
   */
  private detectLayerNorm(source: string, kernel: CudaKernelInfo): boolean {
    const hasLayerNormHint = /layernorm|layer_norm|ln/i.test(kernel.name);
    const hasHiddenDim = /hidden|feature|embed|dim/i.test(source);
    const normalizeOverFeatures = /for\s*\([^)]*\w*_?dim|for\s*\([^)]*hidden/i.test(source);

    return hasLayerNormHint || (hasHiddenDim && normalizeOverFeatures);
  }

  /**
   * Detect BatchNorm (normalize over batch dimension)
   */
  private detectBatchNorm(source: string, kernel: CudaKernelInfo): boolean {
    const hasBatchNormHint = /batchnorm|batch_norm|bn/i.test(kernel.name);
    const hasBatchDim = /batch|n_samples/i.test(source);
    const hasRunningStats = /running_mean|running_var|momentum/i.test(source);

    return hasBatchNormHint || (hasBatchDim && hasRunningStats);
  }

  /**
   * Detect GroupNorm (normalize over groups of channels)
   */
  private detectGroupNorm(source: string, kernel: CudaKernelInfo): boolean {
    const hasGroupNormHint = /groupnorm|group_norm|gn/i.test(kernel.name);
    const hasGroups = /groups?|num_groups|group_size/i.test(source);
    const hasChannelGroups = /channels?\s*\/\s*groups?|c\s*\/\s*g/i.test(source);

    return hasGroupNormHint || (hasGroups && hasChannelGroups);
  }

  /**
   * Detect InstanceNorm (per-sample, per-channel)
   */
  private detectInstanceNorm(source: string, kernel: CudaKernelInfo): boolean {
    const hasInstanceNormHint = /instancenorm|instance_norm|in/i.test(kernel.name);
    // Instance norm: batch idx and channel idx outside mean/var loop
    const hasSpatialLoop = /for\s*\([^)]*h\s*\)|for\s*\([^)]*w\s*\)/i.test(source);
    const perChannelPerSample = /\[\s*n\s*\]\s*\[\s*c\s*\]|\[\s*b\s*,\s*c\s*\]/i.test(source);

    return hasInstanceNormHint || (hasSpatialLoop && perChannelPerSample);
  }

  /**
   * Detect RMSNorm (no mean subtraction)
   */
  private detectRMSNorm(source: string): boolean {
    // RMS = sqrt(mean(x^2))
    const hasRMS = /rms|root_mean_square/i.test(source);
    const hasSquareSum = /x\s*\*\s*x|pow.*2|sq\s*=/.test(source);
    const noMeanSubtract = !/\(\s*x\s*-\s*mean\s*\)|x\s*-=\s*mean/.test(source);

    // RMSNorm: normalize by sqrt(mean(x^2) + eps), no mean subtraction
    if (hasRMS) return true;
    if (hasSquareSum && noMeanSubtract && /rsqrt|\/\s*sqrt/.test(source)) return true;

    return false;
  }

  /**
   * Determine normalization variant
   */
  private determineVariant(
    isLayerNorm: boolean,
    isBatchNorm: boolean,
    isGroupNorm: boolean,
    isInstanceNorm: boolean,
    isRMSNorm: boolean
  ): PatternVariant {
    if (isRMSNorm) return 'rmsnorm' as PatternVariant;
    if (isLayerNorm) return 'layernorm' as PatternVariant;
    if (isBatchNorm) return 'batchnorm' as PatternVariant;
    if (isGroupNorm) return 'groupnorm' as PatternVariant;
    if (isInstanceNorm) return 'instancenorm' as PatternVariant;

    return 'layernorm' as PatternVariant; // default
  }
}

export const normalizationMatcher = new NormalizationMatcher();
