/**
 * Rotary Position Embedding (RoPE) Pattern Matcher
 * Detects RoPE patterns used in LLaMA, GPT-NeoX, and other modern transformers
 */

import { CudaKernelInfo, PatternMatch, Evidence, PatternVariant } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class RoPEMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;

    // === Primary Indicators (high weight) ===

    // 1. Rotation matrix application (complex multiplication pattern)
    const rotationInfo = this.detectRotationPattern(source);
    if (rotationInfo.found) {
      addEvidence(evidence, 'rotation_pattern', 0.35,
        `Rotation: ${rotationInfo.type}`);
    }

    // 2. Frequency calculation (position * freq)
    const freqInfo = this.detectFrequencyPattern(source);
    if (freqInfo.found) {
      addEvidence(evidence, 'frequency_pattern', 0.30,
        `Frequency computation: ${freqInfo.type}`);
    }

    // 3. Sin/cos pair computation
    const sinCosInfo = this.detectSinCosPair(source);
    if (sinCosInfo.found) {
      addEvidence(evidence, 'sincos_pair', 0.25,
        'Sin/cos pair computation');
    }

    // === Secondary Indicators (medium weight) ===

    // 4. Position-based indexing
    const hasPosIndex = /pos\s*\*|position\s*\*|seq_?idx\s*\*/i.test(source);
    if (hasPosIndex) {
      addEvidence(evidence, 'position_index', 0.15,
        'Position-based indexing');
    }

    // 5. Head dimension pairing (x[..., 0::2], x[..., 1::2])
    const hasPairedDim = /2\s*\*\s*\w+|\/\s*2|\%\s*2|&\s*1|0::2|1::2/.test(source);
    if (hasPairedDim && sinCosInfo.found) {
      addEvidence(evidence, 'paired_dim', 0.15,
        'Paired dimension access');
    }

    // 6. Inverse frequency computation (1 / (10000^(2i/d)))
    const hasInvFreq = /10000|1e4|inv_?freq|1\.0\s*\/\s*\(\s*base\s*\^|pow.*base/.test(source);
    if (hasInvFreq) {
      addEvidence(evidence, 'inv_freq', 0.20,
        'Inverse frequency computation');
    }

    // 7. Name hints
    if (/rope|rotary|rot_?pos|position_?embed/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests RoPE');
    }

    // 8. Complex number rotation (a + bi) * (cos + i*sin)
    const hasComplexRot = /\*\s*cos.*-.*\*\s*sin|\*\s*sin.*\+.*\*\s*cos/.test(source);
    if (hasComplexRot) {
      addEvidence(evidence, 'complex_rotation', 0.20,
        'Complex rotation formula');
    }

    // 9. Interleaved or sequential layout handling
    const hasLayoutHandling = /interleave|sequential|paired|half_?dim/i.test(source);
    if (hasLayoutHandling) {
      addEvidence(evidence, 'layout_handling', 0.10,
        'RoPE layout handling');
    }

    // 10. Theta/angle computation
    const hasTheta = /theta|angle\s*=|freq\s*\*\s*pos/i.test(source);
    if (hasTheta) {
      addEvidence(evidence, 'theta_compute', 0.15,
        'Theta/angle computation');
    }

    // === Negative Indicators ===

    // Standard attention without rotation
    if (/softmax|attention/i.test(source) && !rotationInfo.found) {
      addEvidence(evidence, 'attention_no_rot', -0.15,
        'Attention pattern without rotation');
    }

    // Just sinusoidal position encoding (not RoPE)
    if (sinCosInfo.found && !rotationInfo.found && !hasComplexRot) {
      addEvidence(evidence, 'sinusoidal_only', -0.10,
        'Sinusoidal without rotation (may be positional encoding)');
    }

    const match = createPatternMatch('rope' as any, evidence, warnings);

    if (match.confidence > 0.3) {
      match.variant = this.determineVariant(rotationInfo, freqInfo);
    }

    return match;
  }

  /**
   * Detect rotation pattern (RoPE core operation)
   */
  private detectRotationPattern(source: string): { found: boolean; type: string } {
    // Standard RoPE rotation: x' = x * cos - x_rotate * sin
    //                        x'_rotate = x * sin + x_rotate * cos
    if (/\w+\s*\*\s*cos\s*-\s*\w+\s*\*\s*sin/.test(source) &&
        /\w+\s*\*\s*sin\s*\+\s*\w+\s*\*\s*cos/.test(source)) {
      return { found: true, type: 'standard rotation' };
    }

    // Pairwise rotation
    if (/x_?\d?\s*=.*cos.*sin.*x_?\d?|q\s*\*\s*cos\s*[\+\-]/.test(source)) {
      return { found: true, type: 'pairwise rotation' };
    }

    // Complex multiplication form
    if (/real.*imag|cos_.*sin_|rope_.*embed/i.test(source)) {
      return { found: true, type: 'complex form' };
    }

    // Rotation matrix application
    if (/rotate|rotation.*matrix|cos.*-sin.*sin.*cos/i.test(source)) {
      return { found: true, type: 'rotation matrix' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect frequency computation pattern
   */
  private detectFrequencyPattern(source: string): { found: boolean; type: string } {
    // Standard: freq_i = 1 / (base^(2i/d))
    if (/1\.0\s*\/\s*(?:pow|__powf)\s*\(\s*(?:10000|base|theta_base)/.test(source)) {
      return { found: true, type: 'standard inverse freq' };
    }

    // Precomputed frequency table access
    if (/freq\s*\[\s*\w+\s*\]|freqs_?cis\s*\[/.test(source)) {
      return { found: true, type: 'precomputed freq table' };
    }

    // Position * frequency
    if (/pos\s*\*\s*freq|position\s*\*\s*(?:freq|inv_freq)/i.test(source)) {
      return { found: true, type: 'pos * freq' };
    }

    // exp(-2i*log(base)/d) form
    if (/exp\s*\(.*log.*base|exp\s*\(.*-2/.test(source)) {
      return { found: true, type: 'exponential form' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect sin/cos pair computation
   */
  private detectSinCosPair(source: string): { found: boolean } {
    // Both sin and cos present
    const hasSin = /\bsin\s*\(|\bsinf\s*\(|__sinf\s*\(/i.test(source);
    const hasCos = /\bcos\s*\(|\bcosf\s*\(|__cosf\s*\(/i.test(source);

    if (hasSin && hasCos) {
      return { found: true };
    }

    // sincos intrinsic
    if (/__sincosf|sincosf/.test(source)) {
      return { found: true };
    }

    // Precomputed sin/cos table
    if (/sin_cache|cos_cache|sin_table|cos_table|sin_\[|cos_\[/i.test(source)) {
      return { found: true };
    }

    return { found: false };
  }

  /**
   * Determine RoPE variant
   */
  private determineVariant(
    rotationInfo: { found: boolean; type: string },
    freqInfo: { found: boolean; type: string }
  ): PatternVariant {
    // Standard RoPE (LLaMA style)
    if (rotationInfo.type === 'standard rotation') {
      return 'rope_standard' as PatternVariant;
    }

    // GPT-NeoX style (interleaved)
    if (rotationInfo.type === 'complex form') {
      return 'rope_neox' as PatternVariant;
    }

    // Precomputed (fused with attention)
    if (freqInfo.type === 'precomputed freq table') {
      return 'rope_cached' as PatternVariant;
    }

    return 'rope_standard' as PatternVariant; // default
  }
}

export const ropeMatcher = new RoPEMatcher();
