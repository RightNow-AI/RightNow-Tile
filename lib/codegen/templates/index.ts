/**
 * Code Generation Templates Index
 * Exports all variant-specific templates
 */

export * from './reduction';
export * from './stencil';

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

  return null;
}
