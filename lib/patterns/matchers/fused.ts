// Fused Pattern Matcher
// Detects fused kernels that combine multiple operations (matmul+activation, conv+batchnorm, etc.)

import { CudaKernelInfo, PatternMatch, Evidence, KernelArchetype } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';
import { PhaseAnalyzer, MultiPhaseAnalysis } from '../../ast/phase-analyzer';

interface FusedOperation {
  type: 'matmul' | 'activation' | 'bias' | 'normalization' | 'residual' | 'reduction' | 'elementwise';
  confidence: number;
  line?: number;
}

export class FusedMatcher implements PatternMatcher {
  private phaseAnalyzer = new PhaseAnalyzer();

  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;
    const sourceLower = source.toLowerCase();

    // Step 1: Analyze phases
    const phaseAnalysis = this.phaseAnalyzer.analyze(kernel);

    // Step 2: Detect individual operations
    const operations = this.detectOperations(kernel);

    // Step 3: Score based on operation combinations
    const hasMatmul = operations.some(op => op.type === 'matmul');
    const hasActivation = operations.some(op => op.type === 'activation');
    const hasBias = operations.some(op => op.type === 'bias');
    const hasNormalization = operations.some(op => op.type === 'normalization');
    const hasResidual = operations.some(op => op.type === 'residual');

    // === Fused Pattern Indicators ===

    // 1. Multiple distinct operations
    const uniqueOpTypes = new Set(operations.map(op => op.type));
    if (uniqueOpTypes.size >= 3) {
      addEvidence(evidence, 'multi_operation', 0.30,
        `Found ${uniqueOpTypes.size} distinct operation types`);
    } else if (uniqueOpTypes.size === 2) {
      addEvidence(evidence, 'two_operations', 0.20,
        `Found 2 distinct operation types`);
    }

    // 2. Matmul + Activation (common fused pattern)
    if (hasMatmul && hasActivation) {
      addEvidence(evidence, 'matmul_activation', 0.25,
        'Found matmul + activation fusion');
    }

    // 3. Matmul + Bias + Activation
    if (hasMatmul && hasBias && hasActivation) {
      addEvidence(evidence, 'matmul_bias_activation', 0.30,
        'Found matmul + bias + activation fusion (common in transformers)');
    }

    // 4. Normalization + Residual (LayerNorm pattern)
    if (hasNormalization && hasResidual) {
      addEvidence(evidence, 'norm_residual', 0.20,
        'Found normalization + residual connection');
    }

    // 5. Multi-phase kernel detected
    if (phaseAnalysis.isMultiPhaseKernel) {
      addEvidence(evidence, 'multi_phase', 0.20,
        `Detected ${phaseAnalysis.phaseCount} computation phases`);
    }

    // 6. Multiple sync points suggesting phase boundaries
    const syncCount = kernel.syncPoints.filter(s => s.type === 'syncthreads').length;
    if (syncCount >= 2) {
      addEvidence(evidence, 'phase_boundaries', 0.10,
        `Multiple sync points (${syncCount}) suggest fused phases`);
    }

    // 7. Name-based hints
    if (/fused|fusion|_act|_relu|_gelu|_silu|bias_act|linear_/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests fused operation');
    }

    // 8. Complex output pattern (multiple writes to different arrays)
    const writeArrays = new Set(
      kernel.memoryAccesses.filter(a => a.accessType === 'write').map(a => a.array)
    );
    if (writeArrays.size >= 2) {
      addEvidence(evidence, 'multi_output', 0.10,
        `Multiple output arrays (${writeArrays.size})`);
    }

    // 9. Back-to-back operations without intermediate global stores
    const hasBackToBack = this.detectBackToBackOperations(kernel);
    if (hasBackToBack) {
      addEvidence(evidence, 'back_to_back', 0.15,
        'Back-to-back operations without intermediate stores');
    }

    // === Negative Indicators ===

    // Single operation only
    if (uniqueOpTypes.size <= 1) {
      addEvidence(evidence, 'single_op', -0.30,
        'Only single operation type detected');
    }

    // No meaningful combinations
    if (!hasMatmul && !hasNormalization && operations.length < 3) {
      addEvidence(evidence, 'weak_fusion', -0.20,
        'No strong fusion patterns detected');
    }

    // Determine variant
    const variant = this.determineVariant(hasMatmul, hasActivation, hasBias, hasNormalization, hasResidual);
    const match = createPatternMatch('fused', evidence, warnings);

    if (variant) {
      match.variant = variant;
    }

    return match;
  }

  private detectOperations(kernel: CudaKernelInfo): FusedOperation[] {
    const operations: FusedOperation[] = [];
    const source = kernel.sourceText;

    // Detect matmul
    if (this.hasMatmulPattern(source)) {
      operations.push({ type: 'matmul', confidence: 0.8 });
    }

    // Detect activations
    const activations = this.detectActivations(source);
    if (activations.length > 0) {
      operations.push({ type: 'activation', confidence: 0.9 });
    }

    // Detect bias add
    if (this.hasBiasPattern(source)) {
      operations.push({ type: 'bias', confidence: 0.8 });
    }

    // Detect normalization
    if (this.hasNormalizationPattern(source)) {
      operations.push({ type: 'normalization', confidence: 0.85 });
    }

    // Detect residual connection
    if (this.hasResidualPattern(source)) {
      operations.push({ type: 'residual', confidence: 0.8 });
    }

    // Detect reduction
    if (this.hasReductionPattern(kernel)) {
      operations.push({ type: 'reduction', confidence: 0.7 });
    }

    // Detect elementwise
    if (this.hasElementwisePattern(source) && operations.length === 0) {
      operations.push({ type: 'elementwise', confidence: 0.5 });
    }

    return operations;
  }

  private hasMatmulPattern(source: string): boolean {
    // Accumulation patterns
    const accumPattern = /\w+\s*\+=\s*\w+\s*\[[^\]]+\]\s*\*\s*\w+\s*\[[^\]]+\]/;
    const fmaPattern = /fmaf?\s*\(|__fmaf\s*\(/;

    return accumPattern.test(source) || fmaPattern.test(source);
  }

  private detectActivations(source: string): string[] {
    const activations: string[] = [];

    // ReLU variants
    if (/fmaxf?\s*\(\s*\w+\s*,\s*0|max\s*\(\s*\w+\s*,\s*0|relu/i.test(source)) {
      activations.push('relu');
    }

    // GELU
    if (/gelu|0\.5.*erf|erf.*0\.5|tanh.*0\.797/i.test(source)) {
      activations.push('gelu');
    }

    // SiLU / Swish
    if (/silu|swish|\*\s*sigmoid|sigmoid\s*\*|__fdividef.*1\.0.*expf/i.test(source)) {
      activations.push('silu');
    }

    // Sigmoid
    if (/sigmoid|1\.0.*\+.*expf.*-|__frcp.*1\.0.*expf/i.test(source)) {
      activations.push('sigmoid');
    }

    // Tanh
    if (/tanhf?\s*\(|__tanf\s*\(/i.test(source)) {
      activations.push('tanh');
    }

    // Leaky ReLU
    if (/fmaxf?\s*\(\s*\w+\s*,\s*\w+\s*\*\s*\w+|leaky/i.test(source)) {
      activations.push('leaky_relu');
    }

    return activations;
  }

  private hasBiasPattern(source: string): boolean {
    // Look for bias addition patterns
    const biasPatterns = [
      /\+\s*bias\s*\[/i,
      /\+\s*b\s*\[/,
      /bias\s*\[.*\]\s*;/i,
      /\w+\s*=\s*\w+\s*\+\s*\w+\s*\[\s*\w+\s*\]/,  // out = val + bias[idx]
    ];

    return biasPatterns.some(p => p.test(source));
  }

  private hasNormalizationPattern(source: string): boolean {
    const source_lower = source.toLowerCase();

    // Layer normalization
    if (/layernorm|layer_norm|ln_/.test(source_lower)) {
      return true;
    }

    // Batch normalization
    if (/batchnorm|batch_norm|bn_/.test(source_lower)) {
      return true;
    }

    // RMS normalization
    if (/rmsnorm|rms_norm/.test(source_lower)) {
      return true;
    }

    // Normalization computation patterns
    // mean = sum / n; var = sum_sq / n - mean^2; normalized = (x - mean) / sqrt(var + eps)
    const hasMean = /mean|\/\s*\w+\s*;.*sum/i.test(source);
    const hasVariance = /var|variance|std|rsqrt/i.test(source);
    const hasEpsilon = /eps|epsilon|1e-|0\.000001/i.test(source);

    return hasMean && hasVariance && hasEpsilon;
  }

  private hasResidualPattern(source: string): boolean {
    const source_lower = source.toLowerCase();

    // Explicit residual
    if (/residual|skip_connection|shortcut/.test(source_lower)) {
      return true;
    }

    // Pattern: out = normalized + input (residual add)
    // or: out = layer_output + residual
    const addBackPattern = /\w+\s*=\s*\w+\s*\+\s*(?:input|x|residual)/i;
    const skipPattern = /out\s*\+=\s*\w+\s*\[/;

    return addBackPattern.test(source) || skipPattern.test(source);
  }

  private hasReductionPattern(kernel: CudaKernelInfo): boolean {
    // Check for reduction-specific patterns
    const hasStrideHalving = kernel.loops.some(l => l.hasStrideHalving);
    const hasWarpShuffle = /__shfl_down_sync|__shfl_xor_sync/.test(kernel.sourceText);

    return hasStrideHalving || hasWarpShuffle;
  }

  private hasElementwisePattern(source: string): boolean {
    // Simple elementwise operations
    const elementwiseOps = /\[\s*idx\s*\]\s*[+\-*/]=|\[\s*tid\s*\]\s*=/;
    return elementwiseOps.test(source);
  }

  private detectBackToBackOperations(kernel: CudaKernelInfo): boolean {
    // Check if operations chain without global memory stores in between
    const source = kernel.sourceText;
    const lines = source.split('\n');

    let lastWriteGlobal = -1;
    let operationAfterWrite = false;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Check for global memory write
      if (/\w+\s*\[\s*(?:g_|global_)?\w+\s*\]\s*=/.test(line) && !/shared/.test(line)) {
        lastWriteGlobal = i;
      }

      // Check for significant operation after last write
      if (lastWriteGlobal >= 0 && i > lastWriteGlobal) {
        if (/expf|fmaxf|tanhf|__expf|__fmaf|\+=.*\*/.test(line)) {
          operationAfterWrite = true;
        }
      }
    }

    // Multiple operations before any global write
    const syncPoints = kernel.syncPoints.filter(s => s.type === 'syncthreads');
    const operationsBetweenSyncs = syncPoints.length >= 1 && lastWriteGlobal === -1;

    return operationAfterWrite || operationsBetweenSyncs;
  }

  private determineVariant(
    hasMatmul: boolean,
    hasActivation: boolean,
    hasBias: boolean,
    hasNormalization: boolean,
    hasResidual: boolean
  ): 'matmul_activation' | 'matmul_bias_activation' | 'conv_batchnorm' | 'layernorm_residual' | 'multi_phase_fused' | undefined {
    if (hasMatmul && hasBias && hasActivation) {
      return 'matmul_bias_activation';
    }

    if (hasMatmul && hasActivation) {
      return 'matmul_activation';
    }

    if (hasNormalization && hasResidual) {
      return 'layernorm_residual';
    }

    // Multiple operations that don't fit other patterns
    const opCount = [hasMatmul, hasActivation, hasBias, hasNormalization, hasResidual].filter(Boolean).length;
    if (opCount >= 3) {
      return 'multi_phase_fused';
    }

    return undefined;
  }
}

export const fusedMatcher = new FusedMatcher();
