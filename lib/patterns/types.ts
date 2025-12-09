import { CudaKernelInfo, KernelArchetype, PatternMatch, Evidence } from '../ast/types';

export interface PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch;
}

export function createPatternMatch(
  archetype: KernelArchetype,
  evidence: Evidence[],
  warnings: string[] = []
): PatternMatch {
  const confidence = calculateConfidence(evidence);
  return { archetype, confidence, evidence, warnings };
}

function calculateConfidence(evidence: Evidence[]): number {
  const total = evidence.reduce((sum, e) => sum + e.weight, 0);
  return Math.min(1.0, Math.max(0, total));
}

export function addEvidence(
  evidence: Evidence[],
  type: string,
  weight: number,
  description: string,
  line?: number
): void {
  evidence.push({ type, weight, description, line });
}
