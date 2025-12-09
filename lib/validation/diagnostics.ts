/**
 * Diagnostics System
 * Comprehensive error, warning, and info reporting for transpilation
 */

import { KernelArchetype, PatternVariant } from '../ast/types';
import { SemanticAnalysisResult } from '../ast/semantic-analyzer';
import { MemoryAnalysisResult } from '../ast/memory-analyzer';

// Diagnostic severity levels
export type DiagnosticSeverity = 'error' | 'warning' | 'info' | 'hint';

// Diagnostic categories
export type DiagnosticCategory =
  | 'parse'
  | 'pattern'
  | 'semantic'
  | 'memory'
  | 'codegen'
  | 'performance'
  | 'correctness';

/**
 * Diagnostic message interface
 */
export interface Diagnostic {
  code: string;
  severity: DiagnosticSeverity;
  category: DiagnosticCategory;
  message: string;
  line?: number;
  column?: number;
  source?: string;
  suggestions?: string[];
  relatedInfo?: RelatedDiagnostic[];
}

export interface RelatedDiagnostic {
  message: string;
  line?: number;
  source?: string;
}

/**
 * Diagnostic codes with their descriptions
 */
export const DIAGNOSTIC_CODES = {
  // Parse errors (E1xx)
  E100: 'Parse error',
  E101: 'Invalid kernel signature',
  E102: 'Unrecognized CUDA construct',
  E103: 'Malformed expression',
  E104: 'Missing kernel parameters',

  // Pattern detection warnings (W2xx)
  W200: 'Low pattern confidence',
  W201: 'Ambiguous pattern detected',
  W202: 'Multiple patterns match with similar confidence',
  W203: 'Pattern variant could not be determined',
  W204: 'Unsupported pattern feature',

  // Semantic issues (W3xx / E3xx)
  E300: 'Possible race condition detected',
  E301: 'Barrier divergence detected',
  W300: 'Potential data dependency issue',
  W301: 'Uninitialized reduction variable',
  W302: 'Missing synchronization',
  W303: 'Indirect memory access detected',

  // Memory issues (W4xx / I4xx)
  W400: 'Poor memory coalescing',
  W401: 'Potential bank conflicts',
  W402: 'High shared memory usage',
  W403: 'Register spilling likely',
  I400: 'Suboptimal tile size',
  I401: 'Consider vectorized loads',
  I402: 'Memory access pattern not optimal',

  // Code generation issues (W5xx / E5xx)
  E500: 'Cannot generate code for pattern',
  E501: 'Unsupported operation',
  W500: 'Generated code may differ from original',
  W501: 'Some optimizations not applied',
  I500: 'Code generation hint',

  // Performance hints (I6xx)
  I600: 'Performance optimization available',
  I601: 'Consider using variant-specific template',
  I602: 'Tile size could be optimized',
  I603: 'Loop unrolling recommended',
};

/**
 * Diagnostics collector class
 */
export class DiagnosticsCollector {
  private diagnostics: Diagnostic[] = [];
  private errorCount = 0;
  private warningCount = 0;

  /**
   * Add a diagnostic
   */
  add(diagnostic: Diagnostic): void {
    this.diagnostics.push(diagnostic);
    if (diagnostic.severity === 'error') {
      this.errorCount++;
    } else if (diagnostic.severity === 'warning') {
      this.warningCount++;
    }
  }

  /**
   * Add an error
   */
  addError(
    code: keyof typeof DIAGNOSTIC_CODES,
    category: DiagnosticCategory,
    message: string,
    options?: { line?: number; source?: string; suggestions?: string[] }
  ): void {
    this.add({
      code,
      severity: 'error',
      category,
      message: `${DIAGNOSTIC_CODES[code]}: ${message}`,
      line: options?.line,
      source: options?.source,
      suggestions: options?.suggestions,
    });
  }

  /**
   * Add a warning
   */
  addWarning(
    code: keyof typeof DIAGNOSTIC_CODES,
    category: DiagnosticCategory,
    message: string,
    options?: { line?: number; source?: string; suggestions?: string[] }
  ): void {
    this.add({
      code,
      severity: 'warning',
      category,
      message: `${DIAGNOSTIC_CODES[code]}: ${message}`,
      line: options?.line,
      source: options?.source,
      suggestions: options?.suggestions,
    });
  }

  /**
   * Add an info message
   */
  addInfo(
    code: keyof typeof DIAGNOSTIC_CODES,
    category: DiagnosticCategory,
    message: string,
    options?: { line?: number; source?: string; suggestions?: string[] }
  ): void {
    this.add({
      code,
      severity: 'info',
      category,
      message: `${DIAGNOSTIC_CODES[code]}: ${message}`,
      line: options?.line,
      source: options?.source,
      suggestions: options?.suggestions,
    });
  }

  /**
   * Check if there are any errors
   */
  hasErrors(): boolean {
    return this.errorCount > 0;
  }

  /**
   * Get error count
   */
  getErrorCount(): number {
    return this.errorCount;
  }

  /**
   * Get warning count
   */
  getWarningCount(): number {
    return this.warningCount;
  }

  /**
   * Get all diagnostics
   */
  getDiagnostics(): Diagnostic[] {
    return [...this.diagnostics];
  }

  /**
   * Get diagnostics by severity
   */
  getBySeverity(severity: DiagnosticSeverity): Diagnostic[] {
    return this.diagnostics.filter(d => d.severity === severity);
  }

  /**
   * Get diagnostics by category
   */
  getByCategory(category: DiagnosticCategory): Diagnostic[] {
    return this.diagnostics.filter(d => d.category === category);
  }

  /**
   * Clear all diagnostics
   */
  clear(): void {
    this.diagnostics = [];
    this.errorCount = 0;
    this.warningCount = 0;
  }

  /**
   * Format diagnostics for display
   */
  format(): string {
    const sorted = this.diagnostics.sort((a, b) => {
      const severityOrder = { error: 0, warning: 1, info: 2, hint: 3 };
      return severityOrder[a.severity] - severityOrder[b.severity];
    });

    return sorted.map(d => {
      const prefix = d.severity.toUpperCase();
      const location = d.line ? ` (line ${d.line})` : '';
      let result = `[${prefix}] ${d.code}${location}: ${d.message}`;

      if (d.suggestions && d.suggestions.length > 0) {
        result += '\n  Suggestions:';
        d.suggestions.forEach(s => {
          result += `\n    - ${s}`;
        });
      }

      return result;
    }).join('\n\n');
  }
}

/**
 * Analyze semantic results for diagnostics
 */
export function analyzeSemanticDiagnostics(
  semantic: SemanticAnalysisResult,
  collector: DiagnosticsCollector
): void {
  // Check for race conditions
  for (const race of semantic.possibleRaces) {
    collector.add({
      code: 'E300',
      severity: race.severity,
      category: 'correctness',
      message: `${DIAGNOSTIC_CODES.E300}: ${race.type} race on ${race.array}`,
      line: race.line1,
      suggestions: [race.suggestion],
      relatedInfo: [{ message: `Second access`, line: race.line2 }],
    });
  }

  // Check for barrier divergence
  if (semantic.hasBarrierDivergence) {
    collector.addError('E301', 'correctness',
      '__syncthreads() in divergent control flow - may cause deadlock',
      { suggestions: ['Ensure all threads reach the barrier or use __syncwarp() for warp-level sync'] }
    );
  }

  // Check for data dependencies
  if (semantic.dataFlow.hasAntiDependency) {
    collector.addWarning('W300', 'semantic',
      'Anti-dependency detected - write-after-read may cause issues',
      { suggestions: ['Consider using double buffering or separate arrays'] }
    );
  }

  // Check reduction variables
  for (const rv of semantic.reductionVariables) {
    if (!rv.usesAtomic && !rv.usesWarpShuffle && rv.scope === 'global') {
      collector.addWarning('W301', 'correctness',
        `Reduction variable '${rv.name}' may need atomic operations for global scope`,
        { suggestions: ['Use atomicAdd or similar for inter-block reduction'] }
      );
    }
  }
}

/**
 * Analyze memory results for diagnostics
 */
export function analyzeMemoryDiagnostics(
  memory: MemoryAnalysisResult,
  collector: DiagnosticsCollector
): void {
  // Check coalescing
  if (memory.globalMemory.coalescingScore < 0.5) {
    collector.addWarning('W400', 'memory',
      `Poor memory coalescing (${Math.round(memory.globalMemory.coalescingScore * 100)}% efficiency)`,
      { suggestions: [
        'Reorganize data layout for coalesced access',
        'Consider transposing data or using shared memory staging',
      ]}
    );
  }

  // Check bank conflicts
  if (memory.sharedMemory.isUsed && memory.sharedMemory.bankConflictRisk > 0.3) {
    collector.addWarning('W401', 'memory',
      `High bank conflict risk (${Math.round(memory.sharedMemory.bankConflictRisk * 100)}%)`,
      { suggestions: [
        'Add padding to shared memory arrays (+1 element per row)',
        'Reorganize access pattern to avoid stride conflicts',
      ]}
    );
  }

  // Check shared memory usage
  if (memory.sharedMemory.totalBytes > 40 * 1024) {
    collector.addWarning('W402', 'memory',
      `High shared memory usage (${Math.round(memory.sharedMemory.totalBytes / 1024)}KB)`,
      { suggestions: ['Reduce tile size to improve occupancy'] }
    );
  }

  // Check for optimization opportunities
  if (memory.accessSummary.hasSpatialLocality && !memory.sharedMemory.isUsed) {
    collector.addInfo('I401', 'performance',
      'Spatial locality detected - vectorized loads could improve throughput',
      { suggestions: ['Use float4/int4 for aligned memory accesses'] }
    );
  }

  // Tile size optimization
  for (const hint of memory.optimizationHints) {
    if (hint.category === 'tiling') {
      collector.addInfo('I400', 'performance', hint.message, {
        suggestions: [hint.suggestion],
      });
    }
  }
}

/**
 * Analyze pattern confidence for diagnostics
 */
export function analyzePatternDiagnostics(
  archetype: KernelArchetype,
  variant: PatternVariant | undefined,
  confidence: number,
  collector: DiagnosticsCollector
): void {
  if (confidence < 0.5) {
    collector.addWarning('W200', 'pattern',
      `Low pattern confidence (${Math.round(confidence * 100)}%) for ${archetype}`,
      { suggestions: [
        'The kernel may not match this pattern well',
        'Consider manual review of the generated code',
      ]}
    );
  } else if (confidence < 0.7) {
    collector.addInfo('W201', 'pattern',
      `Moderate pattern confidence (${Math.round(confidence * 100)}%) for ${archetype}`,
      { suggestions: ['Some features may not be captured correctly'] }
    );
  }

  if (!variant) {
    collector.addInfo('W203', 'pattern',
      `Pattern variant could not be determined for ${archetype}`,
      { suggestions: ['Using default template - may not be optimal'] }
    );
  }
}

/**
 * Create a new diagnostics collector
 */
export function createDiagnosticsCollector(): DiagnosticsCollector {
  return new DiagnosticsCollector();
}

// Export singleton for convenience
export const diagnostics = new DiagnosticsCollector();
