// FFT (Fast Fourier Transform) Pattern Matcher
// Detects FFT kernels including Radix-2, Radix-4, and inverse FFT

import { CudaKernelInfo, PatternMatch, Evidence } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class FFTMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;
    const sourceLower = source.toLowerCase();

    // === Primary Indicators (high weight) ===

    // 1. Butterfly computation pattern (core of FFT)
    const butterflyAnalysis = this.analyzeButterflyPattern(kernel);
    if (butterflyAnalysis.hasFullButterfly) {
      addEvidence(evidence, 'butterfly_pattern', 0.35,
        'Found complete butterfly computation pattern');
    } else if (butterflyAnalysis.hasPartialButterfly) {
      addEvidence(evidence, 'partial_butterfly', 0.20,
        'Found partial butterfly pattern');
    }

    // 2. Bit-reversal indexing
    const hasBitReversal = this.detectBitReversal(kernel);
    if (hasBitReversal) {
      addEvidence(evidence, 'bit_reversal', 0.25,
        'Found bit-reversal permutation pattern');
    }

    // 3. Twiddle factor computation (W = exp(-2*pi*i*k/N))
    const hasTwiddleFactor = this.detectTwiddleFactor(kernel);
    if (hasTwiddleFactor) {
      addEvidence(evidence, 'twiddle_factor', 0.30,
        'Found twiddle factor computation (W_N)');
    }

    // 4. Complex number arithmetic
    const hasComplexArith = this.detectComplexArithmetic(kernel);
    if (hasComplexArith) {
      addEvidence(evidence, 'complex_arithmetic', 0.20,
        'Found complex number arithmetic');
    }

    // === Secondary Indicators (medium weight) ===

    // 5. Stride doubling pattern (typical FFT iteration)
    const hasStrideDoubling = kernel.loops.some(l => l.hasStrideDoubling);
    if (hasStrideDoubling) {
      addEvidence(evidence, 'stride_doubling', 0.15,
        'Found stride doubling pattern');
    }

    // 6. Log2 iterations (FFT stages)
    const hasLog2Iterations = this.detectLog2Iterations(kernel);
    if (hasLog2Iterations) {
      addEvidence(evidence, 'log2_iterations', 0.15,
        'Found log2(N) iteration pattern');
    }

    // 7. Synchronization between stages
    const syncCount = kernel.syncPoints.filter(s => s.type === 'syncthreads').length;
    if (syncCount >= 3) {
      addEvidence(evidence, 'stage_sync', 0.10,
        `Multiple sync points (${syncCount}) for FFT stages`);
    }

    // 8. Name-based hints
    if (/fft|fourier|dft|ifft|radix/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.15,
        'Kernel name suggests FFT operation');
    }

    // 9. Power-of-2 constants
    const hasPow2Constants = this.detectPowerOf2Constants(source);
    if (hasPow2Constants) {
      addEvidence(evidence, 'pow2_constants', 0.10,
        'Found power-of-2 constants');
    }

    // 10. Sin/Cos function calls (for twiddle factors)
    const hasSinCos = /__sincosf|__sinf.*__cosf|sincosf|sinf.*cosf/i.test(source);
    if (hasSinCos) {
      addEvidence(evidence, 'sincos_calls', 0.15,
        'Found sin/cos computations (twiddle factors)');
    }

    // 11. Real and imaginary part handling
    const hasRealImag = /real|imag|\.x|\.y|_re|_im/i.test(source);
    if (hasRealImag && hasComplexArith) {
      addEvidence(evidence, 'real_imag_parts', 0.10,
        'Found real/imaginary part handling');
    }

    // === Negative Indicators ===

    // No trig functions and no bit manipulation
    if (!hasTwiddleFactor && !hasBitReversal) {
      addEvidence(evidence, 'no_fft_core', -0.30,
        'Missing core FFT operations (twiddle/bit-reversal)');
    }

    // Simple reduction without complex arithmetic
    if (kernel.loops.some(l => l.hasStrideHalving) && !hasComplexArith) {
      addEvidence(evidence, 'simple_reduction', -0.20,
        'Stride halving without complex arithmetic suggests reduction');
    }

    // Softmax pattern (not FFT)
    if (/expf.*sum.*\/|softmax/i.test(source)) {
      addEvidence(evidence, 'softmax_not_fft', -0.25,
        'Softmax pattern detected instead of FFT');
    }

    // Determine variant
    const variant = this.determineVariant(source, hasBitReversal, butterflyAnalysis);
    const match = createPatternMatch('fft', evidence, warnings);

    if (variant) {
      match.variant = variant;
    }

    return match;
  }

  private analyzeButterflyPattern(kernel: CudaKernelInfo): {
    hasFullButterfly: boolean;
    hasPartialButterfly: boolean;
  } {
    const source = kernel.sourceText;

    // Butterfly pattern: a' = a + w*b, b' = a - w*b
    // Look for paired add/subtract with twiddle multiplication

    // Full butterfly: both addition and subtraction with twiddle
    const hasAddWithTwiddle = /\w+\s*=\s*\w+\s*\+\s*(?:\w+\s*\*\s*)?[wW]/.test(source) ||
                              /\w+\s*\+=\s*[wW]/.test(source);
    const hasSubWithTwiddle = /\w+\s*=\s*\w+\s*-\s*(?:\w+\s*\*\s*)?[wW]/.test(source) ||
                              /\w+\s*-=\s*[wW]/.test(source);

    // Alternative: temp = w * b; a' = a + temp; b' = a - temp
    const hasTempCompute = /temp|t\d*\s*=.*\*.*[wW]|[wW].*\*/i.test(source);
    const hasPairedAddSub = /\+\s*temp.*-\s*temp|-\s*temp.*\+\s*temp/i.test(source);

    // Butterfly with complex arithmetic
    const hasComplexButterfly = /\.x\s*=.*\.x.*[+-]|\.y\s*=.*\.y.*[+-]/.test(source);

    const hasFullButterfly = (hasAddWithTwiddle && hasSubWithTwiddle) ||
                             (hasTempCompute && hasPairedAddSub) ||
                             hasComplexButterfly;

    const hasPartialButterfly = hasAddWithTwiddle || hasSubWithTwiddle ||
                                hasTempCompute || hasComplexButterfly;

    return { hasFullButterfly, hasPartialButterfly };
  }

  private detectBitReversal(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // Explicit __brev intrinsic
    if (/__brev\s*\(|__brevll\s*\(/.test(source)) {
      return true;
    }

    // Bit reversal in variable name
    if (/bitrev|bit_rev|reversed_index|rev_idx/i.test(source)) {
      return true;
    }

    // Manual bit reversal pattern
    // j = j | ((i >> k) & 1) << (n - 1 - k)
    const manualBitRev = /\|\s*\(\s*\(\s*\w+\s*>>\s*\w+\s*\)\s*&\s*1\s*\)\s*<</.test(source);

    // Bit manipulation for index shuffling
    const bitShuffle = /&\s*\(\s*\w+\s*-\s*1\s*\).*>>|<<.*&\s*\w+/.test(source);

    return manualBitRev || bitShuffle;
  }

  private detectTwiddleFactor(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // Explicit twiddle naming
    if (/twiddle|w_r|w_i|omega|W_N/i.test(source)) {
      return true;
    }

    // Twiddle computation: exp(-2*pi*i*k/N) or sin/cos(-2*pi*k/N)
    const twiddleCompute = /2\.0?\s*\*\s*(?:3\.14|M_PI|PI).*\/\s*\w+|M_PI.*\*.*\/\s*N/i.test(source);

    // Sin/cos with fractional argument
    const sinCosFrac = /__sincosf\s*\(.*\*.*\/|__sinf\s*\(.*\/.*\*|cosf?\s*\(.*\*.*\//i.test(source);

    // Precomputed twiddle table access
    const twiddleTable = /twiddle\s*\[|W\s*\[\s*\w+\s*\*|omega\s*\[/i.test(source);

    return twiddleCompute || sinCosFrac || twiddleTable;
  }

  private detectComplexArithmetic(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // Complex struct access (.x, .y for real/imag)
    const hasComplexStruct = /\.\s*x\s*[+\-*/=]|\.\s*y\s*[+\-*/=]/.test(source);

    // Separate real/imaginary arrays
    const hasRealImagArrays = /\w+_re\s*\[|\w+_im\s*\[|\w+_real\s*\[|\w+_imag\s*\[/i.test(source);

    // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    const complexMult = /\*.*\s*-\s*.*\*.*\*.*\s*\+\s*.*\*/.test(source);

    // cuComplex type or float2
    const hasComplexType = /cuComplex|cuDoubleComplex|float2|double2|complex/i.test(source);

    return hasComplexStruct || hasRealImagArrays || complexMult || hasComplexType;
  }

  private detectLog2Iterations(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // log2 function call
    if (/log2f?\s*\(|__log2f\s*\(/i.test(source)) {
      return true;
    }

    // Counting leading zeros for log2
    if (/__clz\s*\(|__clzll\s*\(/.test(source)) {
      return true;
    }

    // Common FFT stage count patterns
    const stagePatterns = [
      /for\s*\([^)]*stage|for\s*\([^)]*s\s*=\s*0.*s\s*<\s*log/i,
      /while\s*\([^)]*>\s*1\s*\)/,
      /NUM_STAGES|numStages|n_stages/i,
    ];

    return stagePatterns.some(p => p.test(source));
  }

  private detectPowerOf2Constants(source: string): boolean {
    // Common FFT sizes: 256, 512, 1024, 2048, 4096
    const pow2Sizes = /\b(?:256|512|1024|2048|4096|8192|16384)\b/;

    // Log2 values: 8, 9, 10, 11, 12
    const log2Values = /LOG2_\w+|log_n|LOG_SIZE/i;

    return pow2Sizes.test(source) || log2Values.test(source);
  }

  private determineVariant(
    source: string,
    hasBitReversal: boolean,
    butterflyAnalysis: { hasFullButterfly: boolean; hasPartialButterfly: boolean }
  ): 'fft_radix2' | 'fft_radix4' | 'fft_radix8' | 'inverse_fft' | 'real_fft' | undefined {
    const sourceLower = source.toLowerCase();

    // Inverse FFT
    if (/ifft|inverse|backward/i.test(sourceLower)) {
      return 'inverse_fft';
    }

    // Real FFT
    if (/real_fft|rfft|real_to_complex/i.test(sourceLower)) {
      return 'real_fft';
    }

    // Radix-4 (processes 4 elements per butterfly)
    if (/radix.?4|radix_4/i.test(sourceLower) ||
        /4\s*\*\s*stride|stride\s*\*\s*4/.test(source)) {
      return 'fft_radix4';
    }

    // Radix-8
    if (/radix.?8|radix_8/i.test(sourceLower)) {
      return 'fft_radix8';
    }

    // Default to radix-2 if butterfly pattern detected
    if (butterflyAnalysis.hasFullButterfly || hasBitReversal) {
      return 'fft_radix2';
    }

    return undefined;
  }
}

export const fftMatcher = new FFTMatcher();
