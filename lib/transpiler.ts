// CUDA SIMT to cuTile Transpiler
// Production-grade pattern-based code transformation with semantic analysis

import { TranspileResult, PatternMatch, KernelIR, ValidationResult, PatternVariant } from './ast/types';
import { astExtractor } from './ast/extractor';
import { semanticAnalyzer, SemanticAnalysisResult } from './ast/semantic-analyzer';
import { memoryAnalyzer, MemoryAnalysisResult } from './ast/memory-analyzer';
import { cudaParser, EnhancedParseResult } from './parser';
import { patternMatcher, PatternAnalysis } from './patterns/matcher';
import { irBuilder } from './ir/builder';
import { irOptimizer, EnhancedKernelIR } from './ir';
import { codeGenerator } from './codegen/generator';
import { getTemplateGenerator } from './codegen/templates';
import { semanticValidator } from './validation/validator';
import {
  DiagnosticsCollector,
  createDiagnosticsCollector,
  analyzeSemanticDiagnostics,
  analyzeMemoryDiagnostics,
  analyzePatternDiagnostics,
  Diagnostic,
} from './validation/diagnostics';

/**
 * Enhanced transpile result with additional analysis info
 */
export interface EnhancedTranspileResult extends TranspileResult {
  enhancedIR?: EnhancedKernelIR;
  semanticAnalysis?: SemanticAnalysisResult;
  memoryAnalysis?: MemoryAnalysisResult;
  diagnostics?: Diagnostic[];
  variant?: PatternVariant;
}

/**
 * Main transpile function - converts CUDA SIMT code to cuTile Python
 * Enhanced with semantic analysis, memory analysis, and variant detection
 */
export async function transpile(code: string): Promise<EnhancedTranspileResult> {
  const diagnostics = createDiagnosticsCollector();

  try {
    // Step 1: Extract kernel information from CUDA source
    const kernels = astExtractor.extract(code);

    if (kernels.length === 0) {
      return createErrorResult('No CUDA kernel found in source code');
    }

    // Use the first kernel found
    const kernel = kernels[0];

    // Step 2: Enhanced parsing for additional analysis
    const enhancedParse = cudaParser.parse(code);

    // Step 3: Semantic analysis
    const semanticResult = semanticAnalyzer.analyze(kernel, enhancedParse);

    // Step 4: Memory analysis
    const memoryResult = memoryAnalyzer.analyze(kernel, enhancedParse);

    // Step 5: Pattern detection (including new patterns)
    let pattern = detectBestPattern(kernel, enhancedParse);

    // Add diagnostics for pattern confidence
    analyzePatternDiagnostics(pattern.archetype, pattern.variant, pattern.confidence, diagnostics);

    // Step 6: Semantic and memory diagnostics
    analyzeSemanticDiagnostics(semanticResult, diagnostics);
    analyzeMemoryDiagnostics(memoryResult, diagnostics);

    // Step 7: Build intermediate representation
    const ir = irBuilder.build(kernel, pattern);

    // Step 8: Optimize IR with memory analysis
    const enhancedIR = irOptimizer.optimize(ir, memoryResult, pattern.variant);

    // Step 9: Generate cuTile Python code (use variant template if available)
    let tileCode: string;
    const templateGenerator = getTemplateGenerator(pattern.archetype, pattern.variant);

    if (templateGenerator) {
      tileCode = templateGenerator(enhancedIR);
    } else {
      tileCode = codeGenerator.generate(ir);
    }

    // Clean up the generated code - remove leading/trailing whitespace
    tileCode = tileCode.trim();

    // Step 10: Validate generated code
    const validation = semanticValidator.validate(kernel, ir, tileCode);

    // Check for errors
    if (diagnostics.hasErrors()) {
      validation.warnings.push(...diagnostics.getBySeverity('error').map(d => d.message));
    }

    // Build final result
    return {
      tileCode,
      pattern,
      ir,
      validation,
      enhancedIR,
      semanticAnalysis: semanticResult,
      memoryAnalysis: memoryResult,
      diagnostics: diagnostics.getDiagnostics(),
      variant: pattern.variant,
    };
  } catch (error) {
    return createErrorResult(`Transpilation error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

/**
 * Detect the best pattern using the unified pattern matcher orchestrator
 * Supports all 18 archetypes with 60+ variants
 */
function detectBestPattern(kernel: any, enhanced: EnhancedParseResult): PatternMatch {
  // Use the pattern matcher orchestrator which includes all 18 pattern matchers:
  // - Core: elementwise, gemm, reduction, scan, stencil
  // - ML/DL: convolution, pooling, normalization, fused
  // - LLM: attention, rope, kvcache, embedding, quantization
  // - Specialized: sparse, histogram, sorting, fft
  const best = patternMatcher.match(kernel);

  // Boost confidence based on enhanced analysis signals
  let boostedConfidence = best.confidence;

  // Use pattern signals from enhanced parse
  for (const signal of enhanced.patternSignals) {
    if (signal.pattern === best.archetype) {
      boostedConfidence = Math.min(boostedConfidence + signal.weight * 0.5, 1.0);
    }
  }

  return {
    ...best,
    confidence: boostedConfidence,
  };
}

/**
 * Get detailed analysis of a CUDA kernel without generating code
 */
export function analyze(code: string): PatternAnalysis | null {
  const kernels = astExtractor.extract(code);

  if (kernels.length === 0) {
    return null;
  }

  return patternMatcher.analyze(kernels[0]);
}

/**
 * Get all pattern matches for debugging/UI
 */
export function getAllPatternMatches(code: string): PatternMatch[] {
  const kernels = astExtractor.extract(code);

  if (kernels.length === 0) {
    return [];
  }

  return patternMatcher.matchAll(kernels[0]);
}

/**
 * Create an error result when transpilation fails
 */
function createErrorResult(errorMessage: string): TranspileResult {
  return {
    tileCode: `# Error: ${errorMessage}\n# Please check your CUDA code and try again.`,
    pattern: {
      archetype: 'elementwise',
      confidence: 0,
      evidence: [],
      warnings: [errorMessage],
    },
    ir: {
      name: 'error',
      originalName: 'unknown',
      archetype: 'elementwise',
      confidence: 0,
      parameters: [],
      loads: [],
      operations: [],
      stores: [],
      tileConfig: { tileSize: 256 },
    },
    validation: {
      isValid: false,
      errors: [errorMessage],
      warnings: [],
      adjustedConfidence: 0,
    },
  };
}

// Re-export types for convenience
export type {
  TranspileResult,
  PatternMatch,
  KernelIR,
  ValidationResult,
  PatternAnalysis,
  PatternVariant,
};

// Export enhanced types
export type { EnhancedKernelIR } from './ir';
export type { SemanticAnalysisResult } from './ast/semantic-analyzer';
export type { MemoryAnalysisResult } from './ast/memory-analyzer';
export type { Diagnostic } from './validation/diagnostics';
