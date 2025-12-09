/**
 * Quantization Pattern Matcher
 * Detects INT8, INT4, FP8 quantization/dequantization patterns
 */

import { CudaKernelInfo, PatternMatch, Evidence, PatternVariant } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class QuantizationMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;

    // === Primary Indicators (high weight) ===

    // 1. Scale and zero-point operations (affine quantization)
    const affineInfo = this.detectAffineQuantization(source);
    if (affineInfo.found) {
      addEvidence(evidence, 'affine_quant', 0.35,
        `Affine quantization: ${affineInfo.type}`);
    }

    // 2. INT8 operations
    const int8Info = this.detectInt8Ops(source, kernel);
    if (int8Info.found) {
      addEvidence(evidence, 'int8_ops', 0.30,
        `INT8 operations: ${int8Info.type}`);
    }

    // 3. INT4 / packed format operations
    const int4Info = this.detectInt4Ops(source);
    if (int4Info.found) {
      addEvidence(evidence, 'int4_ops', 0.30,
        `INT4 operations: ${int4Info.type}`);
    }

    // 4. FP8 operations
    const fp8Info = this.detectFp8Ops(source);
    if (fp8Info.found) {
      addEvidence(evidence, 'fp8_ops', 0.30,
        `FP8 operations: ${fp8Info.type}`);
    }

    // === Secondary Indicators (medium weight) ===

    // 5. Clipping/saturation (quantization range)
    const hasClipping = /clip|clamp|saturate|min\s*\(\s*max|max\s*\(\s*min/i.test(source);
    if (hasClipping) {
      addEvidence(evidence, 'clipping', 0.15,
        'Clipping/saturation detected');
    }

    // 6. Rounding mode
    const hasRounding = /round|nearbyint|rint|floor|ceil|\+\s*0\.5f?/.test(source);
    if (hasRounding) {
      addEvidence(evidence, 'rounding', 0.10,
        'Rounding operation');
    }

    // 7. Type conversion (float to int or vice versa)
    const hasTypeConv = /\(\s*(?:int8_t|int4|uint8_t|char)\s*\)|\(\s*float\s*\)\s*\(?\s*int/i.test(source);
    if (hasTypeConv) {
      addEvidence(evidence, 'type_conversion', 0.15,
        'Type conversion for quantization');
    }

    // 8. Scale/zero-point parameters
    const quantParams = kernel.parameters.filter(p =>
      /scale|zero_?point|qzero|quant|dequant/i.test(p.name)
    );
    if (quantParams.length >= 1) {
      addEvidence(evidence, 'quant_params', 0.15,
        `Quantization parameters: ${quantParams.map(p => p.name).join(', ')}`);
    }

    // 9. Name hints
    if (/quant|dequant|int8|int4|fp8|w8a8|w4a16/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests quantization');
    }

    // 10. Per-channel or per-group quantization
    const hasPerChannel = /per_?channel|per_?token|per_?group|group_?size/i.test(source);
    if (hasPerChannel) {
      addEvidence(evidence, 'per_channel', 0.15,
        'Per-channel/group quantization');
    }

    // 11. Symmetric vs asymmetric
    const isSymmetric = /symmetric|no_?zero_?point|zero_?point\s*=\s*0/.test(source);
    if (isSymmetric) {
      addEvidence(evidence, 'symmetric', 0.05,
        'Symmetric quantization');
    }

    // 12. GPTQ/AWQ style weight-only quantization
    const hasWeightOnly = /gptq|awq|weight_?quant|w4|w8/i.test(source);
    if (hasWeightOnly) {
      addEvidence(evidence, 'weight_only', 0.15,
        'Weight-only quantization');
    }

    // === Negative Indicators ===

    // Standard floating-point only operations
    if (!affineInfo.found && !int8Info.found && !int4Info.found && !fp8Info.found) {
      const onlyFloat = kernel.parameters.every(p => /float|double|half/i.test(p.type));
      if (onlyFloat) {
        addEvidence(evidence, 'float_only', -0.25,
          'All floating-point parameters');
      }
    }

    const match = createPatternMatch('quantization' as any, evidence, warnings);

    if (match.confidence > 0.3) {
      match.variant = this.determineVariant(affineInfo, int8Info, int4Info, fp8Info);
    }

    return match;
  }

  /**
   * Detect affine quantization (scale * int + zero_point)
   */
  private detectAffineQuantization(source: string): { found: boolean; type: string } {
    // Dequantization: (q - zero_point) * scale
    if (/\(\s*\w+\s*-\s*zero_?point\s*\)\s*\*\s*scale/i.test(source)) {
      return { found: true, type: 'dequantization' };
    }

    // Quantization: round(x / scale + zero_point)
    if (/round.*\/\s*scale|\/\s*scale.*\+\s*zero_?point/i.test(source)) {
      return { found: true, type: 'quantization' };
    }

    // Scale multiplication with type cast
    if (/scale\s*\*.*\(\s*(?:int|float)/i.test(source)) {
      return { found: true, type: 'scaled conversion' };
    }

    // Per-tensor scaling
    if (/\*\s*scale\s*\[|\/\s*scale\s*\[/.test(source)) {
      return { found: true, type: 'per-tensor scale' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect INT8 operations
   */
  private detectInt8Ops(source: string, kernel: CudaKernelInfo): { found: boolean; type: string } {
    // INT8 parameters
    const hasInt8Param = kernel.parameters.some(p =>
      /int8|char|i8/i.test(p.type)
    );

    // INT8 GEMM (dp4a instruction)
    if (/dp4a|__dp4a|imma/i.test(source)) {
      return { found: true, type: 'INT8 GEMM (dp4a)' };
    }

    // INT8 tensor core
    if (/wmma.*s8|mma.*int8/i.test(source)) {
      return { found: true, type: 'INT8 tensor core' };
    }

    // Basic INT8 arithmetic
    if (hasInt8Param && /\*.*scale|\/.*scale/.test(source)) {
      return { found: true, type: 'INT8 with scaling' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect INT4 / packed format operations
   */
  private detectInt4Ops(source: string): { found: boolean; type: string } {
    // Packed INT4 (two values in one byte)
    if (/&\s*0x[fF0]|\s*>>\s*4|<<\s*4\s*\|/i.test(source)) {
      if (/quant|scale|weight/i.test(source)) {
        return { found: true, type: 'packed INT4' };
      }
    }

    // INT4 weight unpacking
    if (/unpack|nibble|int4|w4/i.test(source)) {
      return { found: true, type: 'INT4 unpack' };
    }

    // GPTQ/AWQ style 4-bit
    if (/qweight.*qzeros|zeros.*scales.*g_idx/i.test(source)) {
      return { found: true, type: 'GPTQ/AWQ 4-bit' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect FP8 operations
   */
  private detectFp8Ops(source: string): { found: boolean; type: string } {
    // FP8 types (E4M3, E5M2)
    if (/fp8|__nv_fp8|e4m3|e5m2|float8/i.test(source)) {
      return { found: true, type: 'FP8 format' };
    }

    // FP8 tensor core
    if (/wmma.*fp8|mma.*e4m3|mma.*e5m2/i.test(source)) {
      return { found: true, type: 'FP8 tensor core' };
    }

    // FP8 scaling
    if (/fp8_?scale|e4m3.*scale|e5m2.*scale/i.test(source)) {
      return { found: true, type: 'FP8 with scaling' };
    }

    return { found: false, type: '' };
  }

  /**
   * Determine quantization variant
   */
  private determineVariant(
    affineInfo: { found: boolean; type: string },
    int8Info: { found: boolean; type: string },
    int4Info: { found: boolean; type: string },
    fp8Info: { found: boolean; type: string }
  ): PatternVariant {
    if (fp8Info.found) return 'quant_fp8' as PatternVariant;
    if (int4Info.found) return 'quant_int4' as PatternVariant;
    if (int8Info.found) return 'quant_int8' as PatternVariant;
    if (affineInfo.type === 'dequantization') return 'dequantize' as PatternVariant;
    if (affineInfo.type === 'quantization') return 'quantize' as PatternVariant;

    return 'quant_int8' as PatternVariant; // default
  }
}

export const quantizationMatcher = new QuantizationMatcher();
