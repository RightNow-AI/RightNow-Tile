/**
 * Convolution Pattern Matcher
 * Detects various convolution patterns including 1D, 2D, 3D, depthwise, and grouped convolutions
 */

import { CudaKernelInfo, PatternMatch, Evidence, PatternVariant } from '../../ast/types';

export interface ConvolutionAnalysis {
  convType: 'conv1d' | 'conv2d' | 'conv3d' | 'depthwise' | 'grouped' | 'transposed';
  kernelSize: number[];
  stride: number[];
  padding: number[];
  dilation: number[];
  groups: number;
  hasIm2col: boolean;
  hasWinograd: boolean;
  usesFFT: boolean;
  implicitGemm: boolean;
}

class ConvolutionPatternMatcher {
  /**
   * Match convolution patterns in a CUDA kernel
   */
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    let confidence = 0;

    // Check for convolution-related parameter names
    const paramScore = this.analyzeParameters(kernel, evidence);
    confidence += paramScore;

    // Check for sliding window pattern
    const slidingWindowScore = this.detectSlidingWindow(kernel, evidence);
    confidence += slidingWindowScore;

    // Check for kernel/filter iteration
    const filterIterScore = this.detectFilterIteration(kernel, evidence);
    confidence += filterIterScore;

    // Check for im2col transformation
    const im2colScore = this.detectIm2col(kernel, evidence);
    confidence += im2colScore;

    // Check for Winograd pattern
    const winogradScore = this.detectWinograd(kernel, evidence);
    confidence += winogradScore;

    // Check for implicit GEMM pattern
    const implicitGemmScore = this.detectImplicitGemm(kernel, evidence);
    confidence += implicitGemmScore;

    // Check for channel/batch indexing
    const channelScore = this.detectChannelIndexing(kernel, evidence);
    confidence += channelScore;

    // Check for padding handling
    const paddingScore = this.detectPaddingHandling(kernel, evidence);
    confidence += paddingScore;

    // Normalize confidence
    confidence = Math.min(confidence, 1.0);

    // Determine variant
    const variant = this.determineVariant(kernel, evidence);

    return {
      archetype: 'convolution' as any,
      variant: variant as PatternVariant,
      confidence,
      evidence,
      warnings: this.generateWarnings(kernel, confidence),
    };
  }

  /**
   * Analyze parameter names for convolution indicators
   */
  private analyzeParameters(kernel: CudaKernelInfo, evidence: Evidence[]): number {
    let score = 0;
    const source = kernel.sourceText.toLowerCase();
    const params = kernel.parameters.map(p => p.name.toLowerCase());

    // Input/output feature map parameters
    const hasInput = params.some(p =>
      /^(input|x|data|in|src|image|img|feat)/.test(p)
    );
    const hasOutput = params.some(p =>
      /^(output|y|out|dst|result)/.test(p)
    );
    const hasWeight = params.some(p =>
      /^(weight|w|filter|kernel|filt)/.test(p)
    );
    const hasBias = params.some(p =>
      /^(bias|b)$/.test(p)
    );

    if (hasInput && hasOutput && hasWeight) {
      score += 0.20;
      evidence.push({
        type: 'parameter_names',
        weight: 0.20,
        description: 'Input/Output/Weight parameters suggest convolution',
      });
    }

    if (hasBias) {
      score += 0.05;
      evidence.push({
        type: 'bias_parameter',
        weight: 0.05,
        description: 'Bias parameter detected',
      });
    }

    // Dimension parameters
    const dimParams = [
      /kernel_size|ksize|ks/,
      /stride|step/,
      /padding|pad/,
      /dilation|dilat/,
      /in_channels|ic|c_in|input_channels/,
      /out_channels|oc|c_out|output_channels/,
      /groups|group/,
    ];

    const matchedDims = dimParams.filter(pattern =>
      params.some(p => pattern.test(p)) || pattern.test(source)
    );

    if (matchedDims.length >= 3) {
      score += 0.15;
      evidence.push({
        type: 'dimension_params',
        weight: 0.15,
        description: `Found ${matchedDims.length} convolution dimension parameters`,
      });
    }

    return score;
  }

  /**
   * Detect sliding window access pattern
   */
  private detectSlidingWindow(kernel: CudaKernelInfo, evidence: Evidence[]): number {
    let score = 0;
    const source = kernel.sourceText;

    // Pattern: output[oh][ow] depends on input[oh*stride+kh][ow*stride+kw]
    const slidingWindowPatterns = [
      // 2D sliding window
      /\[\s*\w+\s*\*\s*stride[_h]?\s*\+\s*\w+\s*\]/i,
      /\[\s*\(\s*\w+\s*\*\s*\d+\s*\+\s*\w+\s*\)\s*\]/,
      // Kernel iteration with image offset
      /ih\s*\+\s*kh|iy\s*\+\s*ky|oh\s*\*\s*stride\s*\+\s*kh/i,
      /iw\s*\+\s*kw|ix\s*\+\s*kx|ow\s*\*\s*stride\s*\+\s*kw/i,
    ];

    const matches = slidingWindowPatterns.filter(p => p.test(source));
    if (matches.length >= 2) {
      score += 0.20;
      evidence.push({
        type: 'sliding_window',
        weight: 0.20,
        description: 'Sliding window access pattern detected',
      });
    } else if (matches.length >= 1) {
      score += 0.10;
      evidence.push({
        type: 'sliding_window',
        weight: 0.10,
        description: 'Partial sliding window pattern detected',
      });
    }

    return score;
  }

  /**
   * Detect filter/kernel iteration
   */
  private detectFilterIteration(kernel: CudaKernelInfo, evidence: Evidence[]): number {
    let score = 0;
    const source = kernel.sourceText;

    // Check for filter size loops
    const filterLoopPatterns = [
      /for\s*\([^)]*kh?\s*[=<]/i,
      /for\s*\([^)]*kw?\s*[=<]/i,
      /for\s*\([^)]*kernel/i,
      /for\s*\([^)]*filter/i,
      /for\s*\([^)]*ksize/i,
    ];

    const filterLoops = filterLoopPatterns.filter(p => p.test(source));

    if (filterLoops.length >= 2) {
      score += 0.15;
      evidence.push({
        type: 'filter_iteration',
        weight: 0.15,
        description: '2D filter iteration detected (kernel size loops)',
      });
    } else if (filterLoops.length >= 1) {
      score += 0.08;
      evidence.push({
        type: 'filter_iteration',
        weight: 0.08,
        description: '1D filter iteration detected',
      });
    }

    // Check for channel iteration
    if (/for\s*\([^)]*c_in|for\s*\([^)]*ic|for\s*\([^)]*in_channel/i.test(source)) {
      score += 0.10;
      evidence.push({
        type: 'channel_iteration',
        weight: 0.10,
        description: 'Input channel iteration detected',
      });
    }

    return score;
  }

  /**
   * Detect im2col transformation
   */
  private detectIm2col(kernel: CudaKernelInfo, evidence: Evidence[]): number {
    let score = 0;
    const source = kernel.sourceText.toLowerCase();

    // Direct im2col function name
    if (/im2col|image2col|patch2col/.test(source)) {
      score += 0.25;
      evidence.push({
        type: 'im2col',
        weight: 0.25,
        description: 'im2col transformation detected',
      });
      return score;
    }

    // im2col pattern: extracting patches to column format
    const im2colPatterns = [
      // Linear index from 2D coordinates
      /col\s*\*\s*\w+\s*\+\s*row/,
      // Patch extraction with channel interleaving
      /\[\s*\(?\s*c\s*\*\s*\w+\s*\+\s*kh\s*\)\s*\*\s*\w+\s*\+\s*kw/i,
      // Column major storage of patches
      /col_idx\s*=|col_offset\s*=/,
    ];

    const im2colMatches = im2colPatterns.filter(p => p.test(source));
    if (im2colMatches.length >= 2) {
      score += 0.15;
      evidence.push({
        type: 'im2col_pattern',
        weight: 0.15,
        description: 'im2col-like data layout pattern',
      });
    }

    return score;
  }

  /**
   * Detect Winograd convolution
   */
  private detectWinograd(kernel: CudaKernelInfo, evidence: Evidence[]): number {
    let score = 0;
    const source = kernel.sourceText.toLowerCase();

    // Winograd indicators
    const winogradPatterns = [
      /winograd/,
      /F\s*\(\s*\d+\s*,\s*\d+\s*\)/,  // F(m,r) notation
      /BtdB|GgGt|AtdA/,  // Winograd matrix names
      /tile_size\s*=\s*[46]/,  // Common Winograd tile sizes
    ];

    const winogradMatches = winogradPatterns.filter(p => p.test(source));
    if (winogradMatches.length >= 2) {
      score += 0.20;
      evidence.push({
        type: 'winograd',
        weight: 0.20,
        description: 'Winograd convolution pattern detected',
      });
    } else if (winogradMatches.length >= 1) {
      score += 0.10;
      evidence.push({
        type: 'winograd',
        weight: 0.10,
        description: 'Possible Winograd convolution',
      });
    }

    return score;
  }

  /**
   * Detect implicit GEMM convolution
   */
  private detectImplicitGemm(kernel: CudaKernelInfo, evidence: Evidence[]): number {
    let score = 0;
    const source = kernel.sourceText.toLowerCase();

    // Implicit GEMM: convolution as matrix multiplication without explicit im2col
    const implicitGemmPatterns = [
      /implicit\s*gemm/,
      /mma\s*\.sync|wmma::mma_sync/,  // Tensor core usage
      /acc\s*\+=\s*\w+\s*\*\s*\w+/,   // Accumulation pattern
      /__syncwarp|__syncthreads/,     // Synchronization for tiled access
    ];

    // Check for tensor core MMA
    if (/wmma|mma\.sync|tensor\s*core/i.test(source)) {
      score += 0.15;
      evidence.push({
        type: 'tensor_core_conv',
        weight: 0.15,
        description: 'Tensor core acceleration detected',
      });
    }

    // Check for blocked/tiled convolution
    const hasBlockedAccess = /\[\s*ty\s*\*/.test(source) && /\[\s*tx\s*\*/.test(source);
    if (hasBlockedAccess) {
      score += 0.10;
      evidence.push({
        type: 'blocked_conv',
        weight: 0.10,
        description: 'Blocked/tiled convolution pattern',
      });
    }

    return score;
  }

  /**
   * Detect channel and batch indexing
   */
  private detectChannelIndexing(kernel: CudaKernelInfo, evidence: Evidence[]): number {
    let score = 0;
    const source = kernel.sourceText;

    // NCHW format indexing
    const nchwPatterns = [
      /\[\s*n\s*\]\s*\[\s*c\s*\]\s*\[\s*h\s*\]\s*\[\s*w\s*\]/i,
      /n\s*\*\s*c\s*\*\s*h\s*\*\s*w|nc\s*\*\s*hw/i,
      /batch\s*\*\s*channels/i,
    ];

    // NHWC format indexing
    const nhwcPatterns = [
      /\[\s*n\s*\]\s*\[\s*h\s*\]\s*\[\s*w\s*\]\s*\[\s*c\s*\]/i,
      /\(\s*h\s*\*\s*w\s*\+\s*w\s*\)\s*\*\s*c\s*\+\s*c/i,
    ];

    if (nchwPatterns.some(p => p.test(source))) {
      score += 0.10;
      evidence.push({
        type: 'nchw_layout',
        weight: 0.10,
        description: 'NCHW data layout detected',
      });
    } else if (nhwcPatterns.some(p => p.test(source))) {
      score += 0.10;
      evidence.push({
        type: 'nhwc_layout',
        weight: 0.10,
        description: 'NHWC data layout detected',
      });
    }

    // Check for group convolution
    if (/groups?\s*>?\s*1|grouped\s*conv|group_size/i.test(source)) {
      score += 0.08;
      evidence.push({
        type: 'grouped_conv',
        weight: 0.08,
        description: 'Grouped convolution detected',
      });
    }

    // Check for depthwise convolution
    if (/depthwise|depth_multiplier|dw_conv/i.test(source)) {
      score += 0.08;
      evidence.push({
        type: 'depthwise_conv',
        weight: 0.08,
        description: 'Depthwise convolution detected',
      });
    }

    return score;
  }

  /**
   * Detect padding handling
   */
  private detectPaddingHandling(kernel: CudaKernelInfo, evidence: Evidence[]): number {
    let score = 0;
    const source = kernel.sourceText;

    // Zero padding checks
    const paddingPatterns = [
      /if\s*\([^)]*<\s*0\s*\|\|[^)]*>=\s*\w+/,  // Bounds check
      /pad_h|pad_w|padding/i,
      /\?\s*0\s*:/,  // Ternary for zero padding
      /ih\s*<\s*0|iw\s*<\s*0/,
    ];

    const paddingMatches = paddingPatterns.filter(p => p.test(source));
    if (paddingMatches.length >= 2) {
      score += 0.10;
      evidence.push({
        type: 'padding_handling',
        weight: 0.10,
        description: 'Padding boundary handling detected',
      });
    }

    return score;
  }

  /**
   * Determine the specific convolution variant
   */
  private determineVariant(kernel: CudaKernelInfo, evidence: Evidence[]): string {
    const source = kernel.sourceText.toLowerCase();

    // Check for specific variants based on evidence
    const hasWinograd = evidence.some(e => e.type === 'winograd');
    const hasIm2col = evidence.some(e => e.type === 'im2col' || e.type === 'im2col_pattern');
    const hasDepthwise = evidence.some(e => e.type === 'depthwise_conv');
    const hasGrouped = evidence.some(e => e.type === 'grouped_conv');
    const hasTensorCore = evidence.some(e => e.type === 'tensor_core_conv');

    if (hasWinograd) return 'conv_winograd';
    if (hasDepthwise) return 'conv_depthwise';
    if (hasGrouped) return 'conv_grouped';
    if (hasTensorCore) return 'conv_implicit_gemm';
    if (hasIm2col) return 'conv_im2col';

    // Check for dimensionality
    const has3d = /conv3d|\[\s*d\s*\]|\[\s*depth/i.test(source);
    const has1d = /conv1d|seq_len|length/.test(source);

    if (has3d) return 'conv_3d';
    if (has1d) return 'conv_1d';

    return 'conv_2d';
  }

  /**
   * Generate warnings based on analysis
   */
  private generateWarnings(kernel: CudaKernelInfo, confidence: number): string[] {
    const warnings: string[] = [];
    const source = kernel.sourceText;

    if (confidence < 0.5) {
      warnings.push('Low confidence convolution detection - verify pattern');
    }

    // Check for potential inefficiencies
    if (!/shared|__shared__/i.test(source)) {
      warnings.push('No shared memory usage detected - consider tiled approach');
    }

    if (/atomicAdd/i.test(source)) {
      warnings.push('Atomic operations detected - may limit performance');
    }

    return warnings;
  }

  /**
   * Suggest tile configuration for the convolution variant
   */
  private suggestTileConfig(variant: string): {
    tileSize?: number;
    blockM?: number;
    blockN?: number;
    blockK?: number;
  } {
    switch (variant) {
      case 'conv_winograd':
        return { tileSize: 4, blockM: 32, blockN: 32 };
      case 'conv_implicit_gemm':
        return { blockM: 128, blockN: 128, blockK: 32 };
      case 'conv_depthwise':
        return { tileSize: 32, blockM: 8, blockN: 8 };
      case 'conv_im2col':
        return { blockM: 128, blockN: 64, blockK: 32 };
      case 'conv_3d':
        return { tileSize: 8, blockM: 8, blockN: 8, blockK: 8 };
      case 'conv_1d':
        return { tileSize: 256 };
      default:
        return { tileSize: 16, blockM: 16, blockN: 16 };
    }
  }

  /**
   * Perform detailed convolution analysis
   */
  analyzeConvolution(kernel: CudaKernelInfo): ConvolutionAnalysis {
    const source = kernel.sourceText;
    const match = this.match(kernel);

    // Extract kernel size
    const kernelSizeMatch = source.match(/kernel_size\s*[=:]\s*(\d+)/i) ||
                            source.match(/ksize\s*[=:]\s*(\d+)/i);
    const kernelSize = kernelSizeMatch ? [parseInt(kernelSizeMatch[1])] : [3];

    // Extract stride
    const strideMatch = source.match(/stride\s*[=:]\s*(\d+)/i);
    const stride = strideMatch ? [parseInt(strideMatch[1])] : [1];

    // Extract padding
    const paddingMatch = source.match(/padding\s*[=:]\s*(\d+)/i);
    const padding = paddingMatch ? [parseInt(paddingMatch[1])] : [0];

    // Extract dilation
    const dilationMatch = source.match(/dilation\s*[=:]\s*(\d+)/i);
    const dilation = dilationMatch ? [parseInt(dilationMatch[1])] : [1];

    // Extract groups
    const groupsMatch = source.match(/groups?\s*[=:]\s*(\d+)/i);
    const groups = groupsMatch ? parseInt(groupsMatch[1]) : 1;

    return {
      convType: this.extractConvType(match.variant as string || 'conv_2d'),
      kernelSize,
      stride,
      padding,
      dilation,
      groups,
      hasIm2col: match.evidence.some(e => e.type.includes('im2col')),
      hasWinograd: match.evidence.some(e => e.type === 'winograd'),
      usesFFT: /fft|cufft/i.test(source),
      implicitGemm: match.evidence.some(e => e.type === 'tensor_core_conv' || e.type === 'blocked_conv'),
    };
  }

  private extractConvType(variant: string | undefined): ConvolutionAnalysis['convType'] {
    if (!variant) return 'conv2d';
    if (variant.includes('depthwise')) return 'depthwise';
    if (variant.includes('grouped')) return 'grouped';
    if (variant.includes('3d')) return 'conv3d';
    if (variant.includes('1d')) return 'conv1d';
    if (variant.includes('transposed')) return 'transposed';
    return 'conv2d';
  }
}

export const convolutionMatcher = new ConvolutionPatternMatcher();
