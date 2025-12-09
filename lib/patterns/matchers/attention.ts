// Attention Pattern Matcher
// Detects Flash Attention, Multi-Head Attention, and related attention mechanisms

import { CudaKernelInfo, PatternMatch, Evidence } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class AttentionMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;
    const sourceLower = source.toLowerCase();

    // === Primary Indicators (high weight) ===

    // 1. Q, K, V matrix parameters (critical for attention)
    const qkvAnalysis = this.analyzeQKVParameters(kernel);
    if (qkvAnalysis.hasAllQKV) {
      addEvidence(evidence, 'qkv_parameters', 0.30,
        'Found Q, K, V matrix parameters');
    } else if (qkvAnalysis.hasSomeQKV) {
      addEvidence(evidence, 'partial_qkv', 0.15,
        `Found ${qkvAnalysis.count}/3 QKV-like parameters`);
    }

    // 2. Matrix transpose pattern (K^T for QK^T)
    const hasTranspose = this.detectTransposePattern(kernel);
    if (hasTranspose) {
      addEvidence(evidence, 'transpose_pattern', 0.20,
        'Found matrix transpose pattern (K^T)');
    }

    // 3. Softmax computation (exp + sum + normalize)
    const softmaxAnalysis = this.analyzeSoftmaxPattern(kernel);
    if (softmaxAnalysis.hasFullSoftmax) {
      addEvidence(evidence, 'softmax_pattern', 0.30,
        'Found softmax computation (exp, sum, normalize)');
    } else if (softmaxAnalysis.hasPartialSoftmax) {
      addEvidence(evidence, 'partial_softmax', 0.15,
        'Found partial softmax pattern');
    }

    // 4. Online softmax (Flash Attention signature)
    const hasOnlineSoftmax = this.detectOnlineSoftmax(kernel);
    if (hasOnlineSoftmax) {
      addEvidence(evidence, 'online_softmax', 0.25,
        'Found online softmax (Flash Attention algorithm)');
    }

    // 5. Block-wise iteration over K/V
    const hasBlockwiseIteration = this.detectBlockwiseIteration(kernel);
    if (hasBlockwiseIteration) {
      addEvidence(evidence, 'blockwise_iteration', 0.20,
        'Found block-wise iteration over K/V blocks');
    }

    // === Secondary Indicators (medium weight) ===

    // 6. Scaling factor (1/sqrt(d_k))
    const hasScalingFactor = this.detectScalingFactor(kernel);
    if (hasScalingFactor) {
      addEvidence(evidence, 'scaling_factor', 0.15,
        'Found attention scaling factor (1/sqrt(d_k))');
    }

    // 7. Two matrix multiplications (QK^T and attention*V)
    const matmulCount = this.countMatmulPatterns(kernel);
    if (matmulCount >= 2) {
      addEvidence(evidence, 'two_matmuls', 0.20,
        `Found ${matmulCount} matrix multiplication patterns`);
    } else if (matmulCount === 1) {
      addEvidence(evidence, 'single_matmul', 0.05,
        'Found single matrix multiplication');
    }

    // 8. Causal mask pattern
    const hasCausalMask = this.detectCausalMask(kernel);
    if (hasCausalMask) {
      addEvidence(evidence, 'causal_mask', 0.15,
        'Found causal attention mask pattern');
    }

    // 9. Running max/sum accumulators
    const hasAccumulators = this.detectRunningAccumulators(kernel);
    if (hasAccumulators) {
      addEvidence(evidence, 'running_accumulators', 0.15,
        'Found running max/sum accumulators');
    }

    // 10. Name-based hints
    if (/attention|attn|flash|mha|multihead|multi_head|self_attn|cross_attn/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.15,
        'Kernel name suggests attention mechanism');
    }

    // 11. Head dimension parameter
    const hasHeadDim = this.detectHeadDimParameter(kernel);
    if (hasHeadDim) {
      addEvidence(evidence, 'head_dim_param', 0.10,
        'Found head dimension parameter');
    }

    // 12. Multi-sync pattern (multiple phases)
    const syncCount = kernel.syncPoints.filter(s => s.type === 'syncthreads').length;
    if (syncCount >= 3) {
      addEvidence(evidence, 'multi_phase_sync', 0.10,
        `Multiple sync points (${syncCount}) suggests multi-phase attention`);
    }

    // === Negative Indicators ===

    // Simple elementwise (no matrix structure)
    if (kernel.memoryAccesses.length <= 2 && kernel.loops.length <= 1) {
      addEvidence(evidence, 'too_simple', -0.20,
        'Too simple for attention pattern');
    }

    // Reduction-specific patterns without attention structure
    if (kernel.loops.some(l => l.hasStrideHalving) && !softmaxAnalysis.hasPartialSoftmax) {
      addEvidence(evidence, 'pure_reduction', -0.15,
        'Has reduction structure without softmax');
    }

    // Stencil patterns (neighbor access without attention structure)
    const neighborAccesses = kernel.memoryAccesses.filter(a => a.hasNeighborOffset);
    if (neighborAccesses.length > 5 && !qkvAnalysis.hasSomeQKV) {
      addEvidence(evidence, 'stencil_pattern', -0.15,
        'Multiple neighbor accesses suggest stencil not attention');
    }

    // Determine variant based on features
    const variant = this.determineVariant(hasOnlineSoftmax, hasCausalMask, qkvAnalysis);
    const match = createPatternMatch('attention', evidence, warnings);

    if (variant) {
      match.variant = variant;
    }

    return match;
  }

  private analyzeQKVParameters(kernel: CudaKernelInfo): {
    hasAllQKV: boolean;
    hasSomeQKV: boolean;
    count: number;
  } {
    const paramNames = kernel.parameters.map(p => p.name.toLowerCase());
    const source = kernel.sourceText.toLowerCase();

    // Check for Q, K, V parameters or arrays
    const qPatterns = ['q', 'query', 'queries'];
    const kPatterns = ['k', 'key', 'keys'];
    const vPatterns = ['v', 'value', 'values'];

    const hasQ = qPatterns.some(p => paramNames.some(n => n.includes(p))) ||
                 /\bq\s*\[|\bquery\s*\[|\bqueries\s*\[/.test(source);
    const hasK = kPatterns.some(p => paramNames.some(n => n.includes(p))) ||
                 /\bk\s*\[|\bkey\s*\[|\bkeys\s*\[/.test(source);
    const hasV = vPatterns.some(p => paramNames.some(n => n.includes(p))) ||
                 /\bv\s*\[|\bvalue\s*\[|\bvalues\s*\[/.test(source);

    const count = [hasQ, hasK, hasV].filter(Boolean).length;

    return {
      hasAllQKV: count === 3,
      hasSomeQKV: count >= 2,
      count,
    };
  }

  private detectTransposePattern(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // Explicit transpose in comments or variable names
    if (/k\s*\^?\s*t|key.*transpose|transpose.*key/i.test(source)) {
      return true;
    }

    // Transposed index pattern: K[j][i] when typical is K[i][j]
    // or K[col * stride + row] instead of K[row * stride + col]
    const transposedAccess = /\[\s*\w+\s*\]\s*\[\s*\w+\s*\].*\[\s*\w+\s*\]\s*\[\s*\w+\s*\]/;
    if (transposedAccess.test(source)) {
      return true;
    }

    // Transpose operations
    if (/transpose|trans\(|\.T\b|\.t\(\)/i.test(source)) {
      return true;
    }

    return false;
  }

  private analyzeSoftmaxPattern(kernel: CudaKernelInfo): {
    hasFullSoftmax: boolean;
    hasPartialSoftmax: boolean;
  } {
    const source = kernel.sourceText;

    // Key components of softmax
    const hasExp = /expf?\s*\(|__expf\s*\(/.test(source);
    const hasMax = /fmaxf?\s*\(|__fmaxf\s*\(|max\s*\(/.test(source);
    const hasSum = /\+=|sum|l_i|l_\w+/.test(source);
    const hasNormalize = /\/\s*\w+|__fdividef|__frcp/.test(source);

    // Check for explicit softmax
    if (/softmax/i.test(source)) {
      return { hasFullSoftmax: true, hasPartialSoftmax: true };
    }

    // Full softmax: exp, max (for numerical stability), sum, and normalize
    const hasFullSoftmax = hasExp && hasMax && hasSum && hasNormalize;

    // Partial: at least exp and either sum or normalize
    const hasPartialSoftmax = hasExp && (hasSum || hasNormalize);

    return { hasFullSoftmax, hasPartialSoftmax };
  }

  private detectOnlineSoftmax(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // Online softmax characteristics:
    // 1. Running max maintained across blocks: m_new = max(m_old, m_current)
    // 2. Rescaling: exp(m_old - m_new) to adjust previous values
    // 3. Running sum that gets rescaled

    const hasRunningMax = /m_\w*\s*=\s*(?:fmaxf?|max)\s*\(.*m_/.test(source) ||
                          /max_\w*\s*=\s*(?:fmaxf?|max)\s*\(/.test(source) ||
                          /m_new|m_ij|row_max/.test(source);

    const hasRescale = /\*\s*expf?\s*\(\s*m_|expf?\s*\(\s*\w+\s*-\s*m_/.test(source) ||
                       /alpha\s*=\s*expf?|rescale/.test(source);

    const hasRunningSum = /l_\w*\s*[+*]=|l_new|l_ij|row_sum/.test(source);

    // Need at least 2 of these characteristics
    const matchCount = [hasRunningMax, hasRescale, hasRunningSum].filter(Boolean).length;

    return matchCount >= 2;
  }

  private detectBlockwiseIteration(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // Look for block-wise iteration patterns
    // for (int block_kv = 0; block_kv < num_blocks; block_kv++)
    const hasBlockLoop = /for\s*\([^)]*block|for\s*\([^)]*BLOCK|for\s*\([^)]*tile/i.test(source);

    // Block-sized loading
    const hasBlockSizedLoad = /\[\s*\w+\s*\*\s*(?:BLOCK|block|TILE|tile)/i.test(source);

    // Sync inside loop (loading tiles)
    const hasSyncInLoop = kernel.loops.some(loop => loop.containsSyncthreads);

    return (hasBlockLoop || hasBlockSizedLoad) && hasSyncInLoop;
  }

  private detectScalingFactor(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // 1/sqrt(d_k) or sqrt(d_k) patterns
    const patterns = [
      /rsqrtf?\s*\(/,                    // rsqrt function
      /__rsqrtf\s*\(/,                   // CUDA fast rsqrt
      /\/\s*sqrtf?\s*\(/,                // 1/sqrt
      /scale\s*[=*]/i,                   // scale factor
      /\*\s*(?:0\.)?(?:125|0625|1767)/,  // Common scale values (1/sqrt(64), 1/sqrt(128))
      /sm_scale|softmax_scale/i,         // Named scale
    ];

    return patterns.some(p => p.test(source));
  }

  private countMatmulPatterns(kernel: CudaKernelInfo): number {
    const source = kernel.sourceText;
    let count = 0;

    // Count accumulation patterns: sum += a * b
    const accumPatterns = source.match(/\+=\s*\w+\s*\[[^\]]+\]\s*\*\s*\w+\s*\[[^\]]+\]/g);
    if (accumPatterns) {
      count += accumPatterns.length;
    }

    // Count FMA patterns
    const fmaPatterns = source.match(/fmaf?\s*\(|__fmaf\s*\(/g);
    if (fmaPatterns) {
      count += Math.ceil(fmaPatterns.length / 4); // FMA typically in inner loop
    }

    // Count nested loop structures
    const nestedLoops = kernel.loops.filter(l => l.nestLevel >= 1);
    count += nestedLoops.length;

    return Math.min(count, 3); // Cap at 3
  }

  private detectCausalMask(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // Causal mask patterns
    const patterns = [
      /causal/i,
      /mask\s*&&\s*\w+\s*[<>]/,           // mask && i < j
      /if\s*\(\s*\w+\s*>\s*\w+\s*\)/,     // if (row > col)
      /\?\s*-?\s*(?:INFINITY|inf|1e)/i,    // ternary with -inf
      /row\s*[<>]\s*col|i\s*[<>]\s*j/,     // row/col comparison
      /lower\s*triangle|upper\s*triangle/i,
    ];

    return patterns.some(p => p.test(source));
  }

  private detectRunningAccumulators(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // Running max: m = max(m, value)
    const hasRunningMax = /m_?\w*\s*=\s*(?:fmaxf?|max)\s*\(\s*m_?\w*\s*,/.test(source);

    // Running sum: l += value or l = l + value
    const hasRunningSum = /l_?\w*\s*\+=|l_?\w*\s*=\s*l_?\w*\s*\+/.test(source);

    // Output accumulator: out += attention * value
    const hasOutputAccum = /out_?\w*\s*\+=|acc_?\w*\s*\+=|o_?\w*\s*\+=/.test(source);

    return hasRunningMax || hasRunningSum || hasOutputAccum;
  }

  private detectHeadDimParameter(kernel: CudaKernelInfo): boolean {
    const paramNames = kernel.parameters.map(p => p.name.toLowerCase());
    const source = kernel.sourceText.toLowerCase();

    const headDimPatterns = ['head_dim', 'headdim', 'd_k', 'dk', 'd_head', 'dhead', 'head_size'];

    return headDimPatterns.some(p =>
      paramNames.some(n => n.includes(p)) || source.includes(p)
    );
  }

  private determineVariant(
    hasOnlineSoftmax: boolean,
    hasCausalMask: boolean,
    qkvAnalysis: { hasAllQKV: boolean; hasSomeQKV: boolean; count: number }
  ): 'flash_attention' | 'flash_attention_v2' | 'multi_head_attention' | 'causal_attention' | 'cross_attention' | undefined {
    if (hasOnlineSoftmax) {
      return hasCausalMask ? 'flash_attention_v2' : 'flash_attention';
    }

    if (hasCausalMask) {
      return 'causal_attention';
    }

    if (qkvAnalysis.hasAllQKV) {
      return 'multi_head_attention';
    }

    return undefined;
  }
}

export const attentionMatcher = new AttentionMatcher();
