// Main Pattern Matcher - Orchestrates all pattern detectors

import { CudaKernelInfo, PatternMatch, KernelArchetype } from '../ast/types';
import { PhaseAnalyzer, MultiPhaseAnalysis } from '../ast/phase-analyzer';

// Core matchers
import { elementwiseMatcher } from './matchers/elementwise';
import { gemmMatcher } from './matchers/gemm';
import { reductionMatcher } from './matchers/reduction';
import { scanMatcher } from './matchers/scan';
import { stencilMatcher } from './matchers/stencil';

// Complex/LLM-specific matchers
import { attentionMatcher } from './matchers/attention';
import { fusedMatcher } from './matchers/fused';
import { fftMatcher } from './matchers/fft';

// Sparse and specialized matchers
import { sparseMatcher } from './matchers/sparse';
import { histogramMatcher } from './matchers/histogram';
import { convolutionMatcher } from './matchers/convolution';

// ML/DL-specific matchers
import { sortingMatcher } from './matchers/sorting';
import { poolingMatcher } from './matchers/pooling';
import { normalizationMatcher } from './matchers/normalization';
import { embeddingMatcher } from './matchers/embedding';
import { ropeMatcher } from './matchers/rope';
import { kvcacheMatcher } from './matchers/kvcache';
import { quantizationMatcher } from './matchers/quantization';

export class PatternMatcherOrchestrator {
  private phaseAnalyzer = new PhaseAnalyzer();

  /**
   * Match a kernel against all pattern archetypes and return the best match
   */
  match(kernel: CudaKernelInfo): PatternMatch {
    // First, run phase analysis to detect multi-phase kernels
    const phaseAnalysis = this.phaseAnalyzer.analyze(kernel);

    // Run all pattern matchers
    const matches = new Map<KernelArchetype, PatternMatch>();

    // Priority matchers for complex kernels (run first)
    // These are LLM-specific or complex multi-phase patterns
    matches.set('attention', attentionMatcher.match(kernel));
    matches.set('fft', fftMatcher.match(kernel));
    matches.set('fused', fusedMatcher.match(kernel));

    // LLM/Transformer-specific matchers
    matches.set('rope', ropeMatcher.match(kernel));
    matches.set('kvcache', kvcacheMatcher.match(kernel));
    matches.set('embedding', embeddingMatcher.match(kernel));
    matches.set('quantization', quantizationMatcher.match(kernel));

    // ML/DL layer matchers
    matches.set('normalization', normalizationMatcher.match(kernel));
    matches.set('pooling', poolingMatcher.match(kernel));
    matches.set('convolution', convolutionMatcher.match(kernel));

    // Standard/Classic matchers
    matches.set('gemm', gemmMatcher.match(kernel));
    matches.set('reduction', reductionMatcher.match(kernel));
    matches.set('scan', scanMatcher.match(kernel));
    matches.set('stencil', stencilMatcher.match(kernel));

    // Specialized matchers
    matches.set('sparse', sparseMatcher.match(kernel));
    matches.set('histogram', histogramMatcher.match(kernel));
    matches.set('sorting', sortingMatcher.match(kernel));

    // Fallback matcher
    matches.set('elementwise', elementwiseMatcher.match(kernel));

    // Boost scores based on phase analysis
    this.adjustScoresFromPhaseAnalysis(matches, phaseAnalysis);

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

    // Add phase analysis info to best match
    if (phaseAnalysis.isMultiPhaseKernel) {
      bestMatch.warnings.push(
        `Multi-phase kernel detected: ${phaseAnalysis.phaseCount} phases`
      );
    }

    return bestMatch;
  }

  /**
   * Adjust pattern scores based on phase analysis results
   */
  private adjustScoresFromPhaseAnalysis(
    matches: Map<KernelArchetype, PatternMatch>,
    phaseAnalysis: MultiPhaseAnalysis
  ): void {
    // Boost attention if online softmax detected
    if (phaseAnalysis.hasOnlineSoftmax) {
      const attention = matches.get('attention');
      if (attention) {
        attention.confidence = Math.min(1.0, attention.confidence + 0.15);
        attention.evidence.push({
          type: 'phase_analysis_boost',
          weight: 0.15,
          description: 'Online softmax detected by phase analyzer',
        });
      }
    }

    // Boost attention if dominant pattern is attention
    if (phaseAnalysis.dominantPattern === 'attention') {
      const attention = matches.get('attention');
      if (attention) {
        attention.confidence = Math.min(1.0, attention.confidence + 0.10);
      }
    }

    // Boost fused if multiple patterns detected in phases
    if (phaseAnalysis.isFusedKernel && phaseAnalysis.fusedPatterns.length >= 2) {
      const fused = matches.get('fused');
      if (fused) {
        fused.confidence = Math.min(1.0, fused.confidence + 0.20);
        fused.evidence.push({
          type: 'phase_analysis_fused',
          weight: 0.20,
          description: `Fused patterns detected: ${phaseAnalysis.fusedPatterns.join(', ')}`,
        });
      }
    }

    // Boost FFT if FFT phase detected
    if (phaseAnalysis.dominantPattern === 'fft') {
      const fft = matches.get('fft');
      if (fft) {
        fft.confidence = Math.min(1.0, fft.confidence + 0.15);
      }
    }

    // Boost normalization if dominant pattern detected
    if (phaseAnalysis.dominantPattern === 'normalization') {
      const norm = matches.get('normalization');
      if (norm) {
        norm.confidence = Math.min(1.0, norm.confidence + 0.10);
      }
    }

    // Boost convolution if dominant pattern detected
    if (phaseAnalysis.dominantPattern === 'convolution') {
      const conv = matches.get('convolution');
      if (conv) {
        conv.confidence = Math.min(1.0, conv.confidence + 0.10);
      }
    }

    // Boost sorting if dominant pattern detected
    if (phaseAnalysis.dominantPattern === 'sorting') {
      const sort = matches.get('sorting');
      if (sort) {
        sort.confidence = Math.min(1.0, sort.confidence + 0.10);
      }
    }

    // Penalize simple patterns for multi-phase kernels
    if (phaseAnalysis.isMultiPhaseKernel && phaseAnalysis.phaseCount >= 3) {
      const elementwise = matches.get('elementwise');
      if (elementwise) {
        elementwise.confidence = Math.max(0, elementwise.confidence - 0.15);
      }
    }
  }

  /**
   * Get all matches sorted by confidence (for debugging/UI)
   */
  matchAll(kernel: CudaKernelInfo): PatternMatch[] {
    const matches: PatternMatch[] = [
      // Complex/LLM-specific kernel matchers
      attentionMatcher.match(kernel),
      fftMatcher.match(kernel),
      fusedMatcher.match(kernel),

      // LLM/Transformer-specific matchers
      ropeMatcher.match(kernel),
      kvcacheMatcher.match(kernel),
      embeddingMatcher.match(kernel),
      quantizationMatcher.match(kernel),

      // ML/DL layer matchers
      normalizationMatcher.match(kernel),
      poolingMatcher.match(kernel),
      convolutionMatcher.match(kernel),

      // Standard matchers
      elementwiseMatcher.match(kernel),
      gemmMatcher.match(kernel),
      reductionMatcher.match(kernel),
      scanMatcher.match(kernel),
      stencilMatcher.match(kernel),

      // Specialized matchers
      sparseMatcher.match(kernel),
      histogramMatcher.match(kernel),
      sortingMatcher.match(kernel),
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
