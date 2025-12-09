// Main Pattern Matcher - Orchestrates all pattern detectors

import { CudaKernelInfo, PatternMatch, KernelArchetype } from '../ast/types';
import { elementwiseMatcher } from './matchers/elementwise';
import { gemmMatcher } from './matchers/gemm';
import { reductionMatcher } from './matchers/reduction';
import { scanMatcher } from './matchers/scan';
import { stencilMatcher } from './matchers/stencil';

export class PatternMatcherOrchestrator {
  /**
   * Match a kernel against all pattern archetypes and return the best match
   */
  match(kernel: CudaKernelInfo): PatternMatch {
    // Run all pattern matchers
    const matches = new Map<KernelArchetype, PatternMatch>();

    matches.set('elementwise', elementwiseMatcher.match(kernel));
    matches.set('gemm', gemmMatcher.match(kernel));
    matches.set('reduction', reductionMatcher.match(kernel));
    matches.set('scan', scanMatcher.match(kernel));
    matches.set('stencil', stencilMatcher.match(kernel));

    // Find the best match
    let bestMatch: PatternMatch | null = null;
    let bestConfidence = -1;

    for (const [archetype, match] of matches) {
      if (match.confidence > bestConfidence) {
        bestConfidence = match.confidence;
        bestMatch = match;
      }
    }

    // If no clear winner, default to elementwise
    if (!bestMatch || bestMatch.confidence < 0.3) {
      const elementwise = matches.get('elementwise')!;
      elementwise.warnings.push('Low confidence detection - defaulting to elementwise');
      return elementwise;
    }

    // Check for potential hybrid kernel (two patterns with similar confidence)
    const sortedMatches = Array.from(matches.values())
      .sort((a, b) => b.confidence - a.confidence);

    if (sortedMatches.length >= 2) {
      const [first, second] = sortedMatches;
      const gap = first.confidence - second.confidence;

      if (gap < 0.15 && second.confidence > 0.4) {
        first.warnings.push(
          `Possible hybrid kernel: also matches ${second.archetype} (${Math.round(second.confidence * 100)}%)`
        );
      }
    }

    return bestMatch;
  }

  /**
   * Get all matches sorted by confidence (for debugging/UI)
   */
  matchAll(kernel: CudaKernelInfo): PatternMatch[] {
    const matches: PatternMatch[] = [
      elementwiseMatcher.match(kernel),
      gemmMatcher.match(kernel),
      reductionMatcher.match(kernel),
      scanMatcher.match(kernel),
      stencilMatcher.match(kernel),
    ];

    return matches.sort((a, b) => b.confidence - a.confidence);
  }

  /**
   * Get a detailed analysis of all patterns for a kernel
   */
  analyze(kernel: CudaKernelInfo): PatternAnalysis {
    const allMatches = this.matchAll(kernel);
    const bestMatch = allMatches[0];

    return {
      bestMatch,
      allMatches,
      kernelInfo: {
        name: kernel.name,
        paramCount: kernel.parameters.length,
        loopCount: kernel.loops.length,
        syncCount: kernel.syncPoints.filter(s => s.type === 'syncthreads').length,
        sharedMemCount: kernel.sharedMemoryDecls.length,
        memoryAccessCount: kernel.memoryAccesses.length,
        hasNeighborAccess: kernel.memoryAccesses.some(a => a.hasNeighborOffset),
        hasStridePattern: kernel.loops.some(l => l.hasStrideHalving || l.hasStrideDoubling),
      },
    };
  }
}

export interface PatternAnalysis {
  bestMatch: PatternMatch;
  allMatches: PatternMatch[];
  kernelInfo: {
    name: string;
    paramCount: number;
    loopCount: number;
    syncCount: number;
    sharedMemCount: number;
    memoryAccessCount: number;
    hasNeighborAccess: boolean;
    hasStridePattern: boolean;
  };
}

export const patternMatcher = new PatternMatcherOrchestrator();
