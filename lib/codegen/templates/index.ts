/**
 * Code Generation Templates Index
 * Exports all variant-specific templates for all supported archetypes
 */

// Re-export all templates
export * from './reduction';
export * from './stencil';
export * from './attention';
export * from './fused';
export * from './sparse';
export * from './histogram';
export * from './convolution';
export * from './sorting';
export * from './pooling';
export * from './normalization';
export * from './embedding';
export * from './rope';
export * from './kvcache';
export * from './quantization';

// Import template functions for use in enhanced generator
import {
  generateTreeReduction,
  generateWarpShuffleReduction,
  generateMultiBlockReduction,
  generateSegmentedReduction,
} from './reduction';

import {
  generateStencil1D3Point,
  generateStencil1D5Point,
  generateStencil2D5Point,
  generateStencil2D9Point,
  generateStencil3D,
} from './stencil';

import { getAttentionGenerator } from './attention';
import { getFusedGenerator } from './fused';
import { getSparseGenerator } from './sparse';
import { getHistogramGenerator } from './histogram';
import { getConvolutionGenerator } from './convolution';
import { getSortingGenerator } from './sorting';
import { getPoolingGenerator } from './pooling';
import { getNormalizationGenerator } from './normalization';
import { getEmbeddingGenerator } from './embedding';
import { getRoPEGenerator } from './rope';
import { getKVCacheGenerator } from './kvcache';
import { getQuantizationGenerator } from './quantization';

import { EnhancedKernelIR } from '../../ir/types';
import { PatternVariant } from '../../ast/types';

/**
 * Get the appropriate template generator for a variant
 */
export function getTemplateGenerator(
  archetype: string,
  variant?: PatternVariant
): ((ir: EnhancedKernelIR) => string) | null {
  // Reduction variants
  if (archetype === 'reduction') {
    switch (variant) {
      case 'tree_reduction':
        return generateTreeReduction;
      case 'warp_shuffle':
        return generateWarpShuffleReduction;
      case 'multi_block':
        return generateMultiBlockReduction;
      case 'segmented':
        return generateSegmentedReduction;
      default:
        return generateTreeReduction;
    }
  }

  // Stencil variants
  if (archetype === 'stencil') {
    switch (variant) {
      case 'stencil_1d_3pt':
        return generateStencil1D3Point;
      case 'stencil_1d_5pt':
        return generateStencil1D5Point;
      case 'stencil_2d_5pt':
        return generateStencil2D5Point;
      case 'stencil_2d_9pt':
        return generateStencil2D9Point;
      case 'stencil_3d':
        return generateStencil3D;
      default:
        return generateStencil2D5Point;
    }
  }

  // Attention variants
  if (archetype === 'attention') {
    return getAttentionGenerator(variant as string) as any;
  }

  // Fused variants
  if (archetype === 'fused') {
    return getFusedGenerator(variant as string) as any;
  }

  // Sparse variants
  if (archetype === 'sparse') {
    return getSparseGenerator(variant as string) as any;
  }

  // Histogram variants
  if (archetype === 'histogram') {
    return getHistogramGenerator(variant as string) as any;
  }

  // Convolution variants
  if (archetype === 'convolution') {
    return getConvolutionGenerator(variant as string) as any;
  }

  // Sorting variants
  if (archetype === 'sorting') {
    return getSortingGenerator(variant as string) as any;
  }

  // Pooling variants
  if (archetype === 'pooling') {
    return getPoolingGenerator(variant as string) as any;
  }

  // Normalization variants
  if (archetype === 'normalization') {
    return getNormalizationGenerator(variant as string) as any;
  }

  // Embedding variants
  if (archetype === 'embedding') {
    return getEmbeddingGenerator(variant as string) as any;
  }

  // RoPE variants
  if (archetype === 'rope') {
    return getRoPEGenerator(variant as string) as any;
  }

  // KV Cache variants
  if (archetype === 'kvcache') {
    return getKVCacheGenerator(variant as string) as any;
  }

  // Quantization variants
  if (archetype === 'quantization') {
    return getQuantizationGenerator(variant as string) as any;
  }

  return null;
}
