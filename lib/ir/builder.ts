// IR Builder - Transforms CUDA kernel info into Intermediate Representation

import {
  CudaKernelInfo,
  PatternMatch,
  KernelIR,
  IRParameter,
  IRLoad,
  IROperation,
  IRStore,
  TileConfig,
  KernelArchetype,
} from '../ast/types';
import { PhaseAnalyzer } from '../ast/phase-analyzer';

import { PatternVariant } from '../ast/types';

// Extended IR types for complex kernels
export interface AttentionIR extends KernelIR {
  variant?: PatternVariant;
  attentionConfig: {
    headDim: number;
    numHeads: number;
    seqLenQ: number;
    seqLenKV: number;
    blockSizeQ: number;
    blockSizeKV: number;
    useCausalMask: boolean;
    useFlashAlgorithm: boolean;
    softmaxScale: number;
  };
}

export interface FusedIR extends KernelIR {
  variant?: PatternVariant;
  fusedOperations: Array<{
    type: string;
    order: number;
  }>;
}

export interface FFTIR extends KernelIR {
  variant?: PatternVariant;
  fftConfig: {
    size: number;
    radix: number;
    isInverse: boolean;
    isReal: boolean;
  };
}

export interface SparseIR extends KernelIR {
  variant?: PatternVariant;
  sparseConfig: {
    format: 'csr' | 'coo' | 'ell' | 'hybrid' | 'bsr';
    vectorWidth: number;
    warpsPerRow: number;
  };
}

export interface HistogramIR extends KernelIR {
  variant?: PatternVariant;
  histogramConfig: {
    numBins: number;
    usePrivatization: boolean;
  };
}

export interface ConvolutionIR extends KernelIR {
  variant?: PatternVariant;
  convConfig: {
    convType: 'conv1d' | 'conv2d' | 'conv3d' | 'depthwise' | 'grouped';
    kernelSize: number[];
    stride: number[];
    padding: number[];
    groups: number;
  };
}

export interface SortingIR extends KernelIR {
  variant?: PatternVariant;
  sortConfig: {
    sortType: 'bitonic' | 'radix' | 'merge';
    direction: 'ascending' | 'descending';
  };
}

export interface PoolingIR extends KernelIR {
  variant?: PatternVariant;
  poolConfig: {
    poolType: 'max' | 'avg' | 'global_max' | 'global_avg';
    kernelSize: number[];
    stride: number[];
    padding: number[];
  };
}

export interface NormalizationIR extends KernelIR {
  variant?: PatternVariant;
  normConfig: {
    normType: 'layernorm' | 'batchnorm' | 'groupnorm' | 'instancenorm' | 'rmsnorm';
    epsilon: number;
    affine: boolean;
    numGroups?: number;
  };
}

export interface EmbeddingIR extends KernelIR {
  variant?: PatternVariant;
  embeddingConfig: {
    vocabSize: number;
    embeddingDim: number;
  };
}

export interface RoPEIR extends KernelIR {
  variant?: PatternVariant;
  ropeConfig: {
    headDim: number;
    rotaryDim: number;
    base: number;
  };
}

export interface KVCacheIR extends KernelIR {
  variant?: PatternVariant;
  kvConfig: {
    isPaged: boolean;
    blockSize: number;
    numKVHeads: number;
  };
}

export interface QuantizationIR extends KernelIR {
  variant?: PatternVariant;
  quantConfig: {
    bits: number;
    isSymmetric: boolean;
    perChannel: boolean;
  };
}

export class IRBuilder {
  private phaseAnalyzer = new PhaseAnalyzer();

  /**
   * Build IR from kernel info and pattern match
   */
  build(kernel: CudaKernelInfo, pattern: PatternMatch): KernelIR {
    const tileConfig = this.determineTileConfig(kernel, pattern.archetype);

    switch (pattern.archetype) {
      case 'attention':
        return this.buildAttentionIR(kernel, pattern, tileConfig);
      case 'fft':
        return this.buildFFTIR(kernel, pattern, tileConfig);
      case 'fused':
        return this.buildFusedIR(kernel, pattern, tileConfig);
      case 'gemm':
        return this.buildGEMMIR(kernel, pattern, tileConfig);
      case 'reduction':
        return this.buildReductionIR(kernel, pattern, tileConfig);
      case 'scan':
        return this.buildScanIR(kernel, pattern, tileConfig);
      case 'stencil':
        return this.buildStencilIR(kernel, pattern, tileConfig);
      case 'sparse':
        return this.buildSparseIR(kernel, pattern, tileConfig);
      case 'histogram':
        return this.buildHistogramIR(kernel, pattern, tileConfig);
      case 'convolution':
        return this.buildConvolutionIR(kernel, pattern, tileConfig);
      case 'sorting':
        return this.buildSortingIR(kernel, pattern, tileConfig);
      case 'pooling':
        return this.buildPoolingIR(kernel, pattern, tileConfig);
      case 'normalization':
        return this.buildNormalizationIR(kernel, pattern, tileConfig);
      case 'embedding':
        return this.buildEmbeddingIR(kernel, pattern, tileConfig);
      case 'rope':
        return this.buildRoPEIR(kernel, pattern, tileConfig);
      case 'kvcache':
        return this.buildKVCacheIR(kernel, pattern, tileConfig);
      case 'quantization':
        return this.buildQuantizationIR(kernel, pattern, tileConfig);
      case 'elementwise':
      default:
        return this.buildElementwiseIR(kernel, pattern, tileConfig);
    }
  }

  private determineTileConfig(kernel: CudaKernelInfo, archetype: KernelArchetype): TileConfig {
    // Default tile sizes based on pattern
    switch (archetype) {
      case 'attention':
        return {
          tileSize: 64,
          blockM: 64,   // Block size for Q
          blockN: 64,   // Block size for K/V
          blockK: 64,   // Head dimension
        };
      case 'fft':
        return {
          tileSize: 256,
          blockM: 256,
        };
      case 'fused':
        return {
          tileSize: 128,
          blockM: 128,
          blockN: 128,
          blockK: 32,
        };
      case 'gemm':
        return {
          tileSize: 128,
          blockM: 128,
          blockN: 128,
          blockK: 32,
        };
      case 'reduction':
        return { tileSize: 256 };
      case 'scan':
        return { tileSize: 256 };
      case 'stencil':
        return { tileSize: 16 }; // 2D tiles typically smaller
      case 'sparse':
        return { tileSize: 256 };
      case 'histogram':
        return { tileSize: 256 };
      case 'convolution':
        return { tileSize: 256, blockM: 16, blockN: 16 };
      case 'sorting':
        return { tileSize: 512 }; // Larger for sorting efficiency
      case 'pooling':
        return { tileSize: 256 };
      case 'normalization':
        return { tileSize: 256 };
      case 'embedding':
        return { tileSize: 256 };
      case 'rope':
        return { tileSize: 128 };
      case 'kvcache':
        return { tileSize: 128, blockM: 64, blockN: 64 };
      case 'quantization':
        return { tileSize: 256 };
      case 'elementwise':
      default:
        return { tileSize: 256 };
    }
  }

  private buildAttentionIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): AttentionIR {
    const parameters = this.mapParameters(kernel);
    const source = kernel.sourceText.toLowerCase();

    // Identify Q, K, V, O arrays
    const { q, k, v, o } = this.identifyAttentionArrays(kernel);

    // Detect attention configuration
    const attentionConfig = this.extractAttentionConfig(kernel, pattern);

    // Build loads for Q, K, V
    const loads: IRLoad[] = [
      {
        source: q,
        target: 'q_tile',
        index: '(head_idx, block_q)',
        shape: `(${tileConfig.blockM}, ${tileConfig.blockK})`,
      },
      {
        source: k,
        target: 'k_tile',
        index: '(head_idx, block_kv)',
        shape: `(${tileConfig.blockN}, ${tileConfig.blockK})`,
      },
      {
        source: v,
        target: 'v_tile',
        index: '(head_idx, block_kv)',
        shape: `(${tileConfig.blockN}, ${tileConfig.blockK})`,
      },
    ];

    // Attention operations: QK^T -> softmax -> AV
    const operations: IROperation[] = [
      {
        type: 'matmul',
        inputs: ['q_tile', 'k_tile'],
        output: 'qk',
        op: 'matmul_transpose_b',
      },
      {
        type: 'elementwise',
        inputs: ['qk'],
        output: 'qk_scaled',
        op: 'scale',
      },
      {
        type: 'reduce',
        inputs: ['qk_scaled'],
        output: 'softmax_out',
        op: 'softmax',
        axis: 1,
      },
      {
        type: 'matmul',
        inputs: ['softmax_out', 'v_tile'],
        output: 'out',
        op: 'matmul',
      },
      {
        type: 'accumulate',
        inputs: ['out', 'acc'],
        output: 'acc',
        op: 'rescale_accumulate',
      },
    ];

    // Store output
    const stores: IRStore[] = [
      {
        source: 'acc',
        target: o,
        index: '(head_idx, block_q)',
      },
    ];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'attention',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
      attentionConfig,
    };
  }

  private buildFFTIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): FFTIR {
    const parameters = this.mapParameters(kernel);
    const source = kernel.sourceText.toLowerCase();

    // Detect FFT configuration
    const fftConfig = this.extractFFTConfig(kernel, pattern);

    // Identify input/output arrays
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);
    const inputName = inputArrays[0] || 'data';
    const outputName = outputArrays[0] || inputName;

    const loads: IRLoad[] = [
      {
        source: inputName,
        target: 'data_tile',
        index: '(pid,)',
        shape: `(${tileConfig.tileSize},)`,
      },
    ];

    // FFT operations depend on radix
    const operations: IROperation[] = [
      {
        type: 'elementwise',
        inputs: ['data_tile'],
        output: 'bit_reversed',
        op: 'bit_reversal',
      },
      {
        type: 'elementwise',
        inputs: ['bit_reversed'],
        output: 'result',
        op: fftConfig.isInverse ? 'ifft' : 'fft',
      },
    ];

    const stores: IRStore[] = [
      {
        source: 'result',
        target: outputName,
        index: '(pid,)',
      },
    ];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'fft',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
      fftConfig,
    };
  }

  private buildFusedIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): FusedIR {
    const parameters = this.mapParameters(kernel);
    const phaseAnalysis = this.phaseAnalyzer.analyze(kernel);

    // Identify arrays
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    // Build loads based on detected phases
    const loads: IRLoad[] = inputArrays.map((arr, i) => ({
      source: arr,
      target: `input_${i}_tile`,
      index: '(pid,)',
      shape: `(${tileConfig.tileSize},)`,
    }));

    // Build operations based on fused patterns
    const operations: IROperation[] = [];
    const fusedOperations: Array<{ type: string; order: number }> = [];

    let currentInput = loads.length > 0 ? loads[0].target : 'input';
    let opOrder = 0;

    // Process each phase
    for (const phase of phaseAnalysis.phases) {
      switch (phase.type) {
        case 'matmul':
          operations.push({
            type: 'matmul',
            inputs: [currentInput, loads.length > 1 ? loads[1].target : 'weight'],
            output: `matmul_out_${opOrder}`,
            op: 'matmul',
          });
          fusedOperations.push({ type: 'matmul', order: opOrder });
          currentInput = `matmul_out_${opOrder}`;
          break;

        case 'softmax':
          operations.push({
            type: 'reduce',
            inputs: [currentInput],
            output: `softmax_out_${opOrder}`,
            op: 'softmax',
            axis: 1,
          });
          fusedOperations.push({ type: 'softmax', order: opOrder });
          currentInput = `softmax_out_${opOrder}`;
          break;

        case 'reduce':
          operations.push({
            type: 'reduce',
            inputs: [currentInput],
            output: `reduce_out_${opOrder}`,
            op: 'sum',
          });
          fusedOperations.push({ type: 'reduction', order: opOrder });
          currentInput = `reduce_out_${opOrder}`;
          break;

        case 'elementwise':
        default:
          operations.push({
            type: 'elementwise',
            inputs: [currentInput],
            output: `elem_out_${opOrder}`,
            op: 'activation',
          });
          fusedOperations.push({ type: 'elementwise', order: opOrder });
          currentInput = `elem_out_${opOrder}`;
          break;
      }
      opOrder++;
    }

    // If no phases detected, default to elementwise
    if (operations.length === 0) {
      operations.push({
        type: 'elementwise',
        inputs: [currentInput],
        output: 'result',
        op: 'identity',
      });
      currentInput = 'result';
    }

    // Store output
    const stores: IRStore[] = outputArrays.map((arr, i) => ({
      source: i === 0 ? currentInput : `output_${i}`,
      target: arr,
      index: '(pid,)',
    }));

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'fused',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
      fusedOperations,
    };
  }

  private identifyAttentionArrays(kernel: CudaKernelInfo): {
    q: string;
    k: string;
    v: string;
    o: string;
  } {
    const paramNames = kernel.parameters.map(p => p.name.toLowerCase());
    const source = kernel.sourceText.toLowerCase();

    // Try to identify Q, K, V, O arrays from parameters or usage
    let q = 'q', k = 'k', v = 'v', o = 'out';

    // Check parameter names
    for (const name of paramNames) {
      if (/^q$|query|queries/.test(name)) q = name;
      else if (/^k$|key|keys/.test(name)) k = name;
      else if (/^v$|value|values/.test(name)) v = name;
      else if (/^o$|out|output/.test(name)) o = name;
    }

    // If standard names not found, use first 4 pointer params
    const pointerParams = kernel.parameters.filter(p => p.isPointer);
    if (pointerParams.length >= 4 && q === 'q') {
      [q, k, v, o] = pointerParams.slice(0, 4).map(p => p.name);
    } else if (pointerParams.length >= 3 && q === 'q') {
      [q, k, v] = pointerParams.slice(0, 3).map(p => p.name);
      o = q; // In-place output
    }

    return { q, k, v, o };
  }

  private extractAttentionConfig(
    kernel: CudaKernelInfo,
    pattern: PatternMatch
  ): AttentionIR['attentionConfig'] {
    const source = kernel.sourceText;
    const variant = pattern.variant;

    // Try to detect head dimension from parameters or constants
    let headDim = 64; // Default
    const headDimMatch = source.match(/head_dim\s*=?\s*(\d+)|HEAD_DIM\s*=?\s*(\d+)/i);
    if (headDimMatch) {
      headDim = parseInt(headDimMatch[1] || headDimMatch[2]);
    }

    // Detect causal mask
    const useCausalMask = /causal|mask.*<|lower.*triangle/i.test(source);

    // Detect Flash Attention (online softmax)
    const useFlashAlgorithm = variant === 'flash_attention' || variant === 'flash_attention_v2' ||
                              /m_new|m_ij|rescale|online/i.test(source);

    return {
      headDim,
      numHeads: 1, // Will be determined at runtime
      seqLenQ: 0,  // Will be determined at runtime
      seqLenKV: 0, // Will be determined at runtime
      blockSizeQ: 64,
      blockSizeKV: 64,
      useCausalMask,
      useFlashAlgorithm,
      softmaxScale: 1.0 / Math.sqrt(headDim),
    };
  }

  private extractFFTConfig(
    kernel: CudaKernelInfo,
    pattern: PatternMatch
  ): FFTIR['fftConfig'] {
    const source = kernel.sourceText.toLowerCase();
    const variant = pattern.variant;

    // Detect size
    let size = 256; // Default
    const sizeMatch = source.match(/fft_size\s*=?\s*(\d+)|N\s*=\s*(\d+)/);
    if (sizeMatch) {
      size = parseInt(sizeMatch[1] || sizeMatch[2]);
    }

    // Detect radix
    let radix = 2; // Default radix-2
    if (variant === 'fft_radix4' || /radix.?4/i.test(source)) radix = 4;
    else if (variant === 'fft_radix8' || /radix.?8/i.test(source)) radix = 8;

    // Detect inverse FFT
    const isInverse = variant === 'inverse_fft' || /ifft|inverse|backward/i.test(source);

    // Detect real FFT
    const isReal = variant === 'real_fft' || /rfft|real_to_complex/i.test(source);

    return { size, radix, isInverse, isReal };
  }

  private buildElementwiseIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): KernelIR {
    // Map parameters
    const parameters = this.mapParameters(kernel);

    // Identify input arrays (read-only) vs output arrays (written)
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    // Build loads
    const loads: IRLoad[] = inputArrays.map(arr => ({
      source: arr,
      target: `${arr}_tile`,
      index: '(pid,)',
      shape: '(tile_size,)',
    }));

    // Detect the operation being performed
    const operations = this.extractElementwiseOperations(kernel, loads);

    // Build stores
    const stores: IRStore[] = outputArrays.map(arr => ({
      source: operations.length > 0 ? operations[operations.length - 1].output : 'result',
      target: arr,
      index: '(pid,)',
    }));

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'elementwise',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
    };
  }

  private buildGEMMIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): KernelIR {
    const parameters = this.mapParameters(kernel);

    // For GEMM, we need A, B input and C output
    const pointerParams = kernel.parameters.filter(p => p.isPointer);
    const matrixNames = pointerParams.map(p => p.name.toLowerCase());

    // Standard A, B, C naming
    const [aName, bName, cName] = matrixNames.length >= 3
      ? matrixNames
      : ['a', 'b', 'c'];

    const loads: IRLoad[] = [
      {
        source: aName,
        target: 'a_tile',
        index: '(pid_m, k)',
        shape: '(block_m, block_k)',
      },
      {
        source: bName,
        target: 'b_tile',
        index: '(k, pid_n)',
        shape: '(block_k, block_n)',
      },
    ];

    const operations: IROperation[] = [
      {
        type: 'matmul',
        inputs: ['a_tile', 'b_tile', 'acc'],
        output: 'acc',
        op: 'matmul',
      },
    ];

    const stores: IRStore[] = [
      {
        source: 'acc',
        target: cName,
        index: '(pid_m, pid_n)',
      },
    ];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'gemm',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
    };
  }

  private buildReductionIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): KernelIR {
    const parameters = this.mapParameters(kernel);

    // Identify input and output arrays
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const inputName = inputArrays[0] || 'input_arr';
    const outputName = outputArrays[0] || 'output_arr';

    const loads: IRLoad[] = [
      {
        source: inputName,
        target: 'tile',
        index: '(pid,)',
        shape: '(tile_size,)',
      },
    ];

    // Detect reduction operation (sum, max, min, etc.)
    const reductionOp = this.detectReductionOp(kernel);

    const operations: IROperation[] = [
      {
        type: 'reduce',
        inputs: ['tile'],
        output: 'result',
        op: reductionOp,
      },
      {
        type: 'atomic',
        inputs: [outputName, 'result'],
        output: outputName,
        op: 'add',
      },
    ];

    const stores: IRStore[] = []; // Atomic handles the store

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'reduction',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
    };
  }

  private buildScanIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): KernelIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const inputName = inputArrays[0] || 'data';
    const outputName = outputArrays[0] || inputName;

    const loads: IRLoad[] = [
      {
        source: inputName,
        target: 'tile',
        index: '(pid,)',
        shape: '(tile_size,)',
      },
    ];

    const operations: IROperation[] = [
      {
        type: 'reduce', // cuTile handles scan via cumsum
        inputs: ['tile'],
        output: 'scanned',
        op: 'cumsum', // Cumulative sum = inclusive scan
      },
    ];

    const stores: IRStore[] = [
      {
        source: 'scanned',
        target: outputName,
        index: '(pid,)',
      },
    ];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'scan',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
    };
  }

  private buildStencilIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): KernelIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const inputName = inputArrays[0] || 'input_arr';
    const outputName = outputArrays[0] || 'output_arr';

    // For stencil, we need to load a tile with halo
    const loads: IRLoad[] = [
      {
        source: inputName,
        target: 'tile',
        index: '(pid_y, pid_x)',
        shape: '(tile_size + 2, tile_size + 2)', // +2 for halo
      },
    ];

    // Stencil operation (simplified as average of neighbors)
    const operations: IROperation[] = [
      {
        type: 'elementwise',
        inputs: ['tile'],
        output: 'result',
        op: 'stencil_apply',
      },
    ];

    const stores: IRStore[] = [
      {
        source: 'result',
        target: outputName,
        index: '(pid_y, pid_x)',
      },
    ];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'stencil',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
    };
  }

  private mapParameters(kernel: CudaKernelInfo): IRParameter[] {
    return kernel.parameters.map(p => {
      const isConstant = !p.isPointer && (p.type === 'int' || p.type === 'size_t');
      return {
        cudaName: p.name,
        cuTileName: p.name.toLowerCase(),
        type: p.type,
        isConstant,
        constantAnnotation: isConstant ? `ct.Constant[${this.mapType(p.type)}]` : undefined,
      };
    });
  }

  private mapType(cudaType: string): string {
    const typeMap: Record<string, string> = {
      'int': 'int',
      'unsigned int': 'int',
      'size_t': 'int',
      'float': 'float',
      'double': 'float',
      'float*': 'float',
      'double*': 'float',
      'int*': 'int',
    };
    return typeMap[cudaType.toLowerCase()] || 'int';
  }

  private generateCuTileName(cudaName: string): string {
    // Convert camelCase or snake_case to a clean name
    return cudaName
      .replace(/([A-Z])/g, '_$1')
      .toLowerCase()
      .replace(/^_/, '')
      .replace(/__+/g, '_');
  }

  private identifyInputArrays(kernel: CudaKernelInfo): string[] {
    const reads = new Set(
      kernel.memoryAccesses
        .filter(a => a.accessType === 'read')
        .map(a => a.array)
    );

    const writes = new Set(
      kernel.memoryAccesses
        .filter(a => a.accessType === 'write')
        .map(a => a.array)
    );

    // Input arrays are read but NOT written (pure inputs)
    return Array.from(reads).filter(arr => !writes.has(arr));
  }

  private identifyOutputArrays(kernel: CudaKernelInfo): string[] {
    const writes = kernel.memoryAccesses
      .filter(a => a.accessType === 'write')
      .map(a => a.array);

    return [...new Set(writes)];
  }

  private extractElementwiseOperations(kernel: CudaKernelInfo, loads: IRLoad[]): IROperation[] {
    const operations: IROperation[] = [];
    const source = kernel.sourceText;

    // Try to detect the operation being performed
    const tileVars = loads.map(l => l.target);

    // Look for common operations
    if (/\+/.test(source) && tileVars.length >= 2) {
      operations.push({
        type: 'elementwise',
        inputs: tileVars.slice(0, 2),
        output: 'result',
        op: '+',
      });
    } else if (/\*/.test(source) && !/\+=/.test(source) && tileVars.length >= 2) {
      operations.push({
        type: 'elementwise',
        inputs: tileVars.slice(0, 2),
        output: 'result',
        op: '*',
      });
    } else if (/-/.test(source) && tileVars.length >= 2) {
      operations.push({
        type: 'elementwise',
        inputs: tileVars.slice(0, 2),
        output: 'result',
        op: '-',
      });
    } else if (tileVars.length === 1) {
      // Unary operation or copy
      operations.push({
        type: 'elementwise',
        inputs: [tileVars[0]],
        output: 'result',
        op: 'identity',
      });
    } else if (tileVars.length >= 2) {
      // Default to addition
      operations.push({
        type: 'elementwise',
        inputs: tileVars.slice(0, 2),
        output: 'result',
        op: '+',
      });
    }

    return operations;
  }

  private detectReductionOp(kernel: CudaKernelInfo): string {
    const source = kernel.sourceText.toLowerCase();

    if (/atomicmax|max\s*\(/.test(source)) return 'max';
    if (/atomicmin|min\s*\(/.test(source)) return 'min';
    if (/\*=|\*\s*=/.test(source) && !/\+/.test(source)) return 'prod';
    // Default to sum
    return 'sum';
  }

  // ============= New Pattern IR Builders =============

  private buildSparseIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): SparseIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const loads: IRLoad[] = inputArrays.map((arr, i) => ({
      source: arr,
      target: `${arr}_tile`,
      index: '(pid,)',
      shape: `(${tileConfig.tileSize},)`,
    }));

    const operations: IROperation[] = [{
      type: 'matmul',
      inputs: ['values_tile', 'x_tile'],
      output: 'result',
      op: 'spmv',
    }];

    const stores: IRStore[] = outputArrays.map(arr => ({
      source: 'result',
      target: arr,
      index: '(pid,)',
    }));

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'sparse',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
      variant: pattern.variant,
      sparseConfig: {
        format: this.detectSparseFormat(kernel, pattern),
        vectorWidth: 1,
        warpsPerRow: 1,
      },
    };
  }

  private detectSparseFormat(kernel: CudaKernelInfo, pattern: PatternMatch): SparseIR['sparseConfig']['format'] {
    if (pattern.variant === 'spmv_coo') return 'coo';
    if (pattern.variant === 'spmv_ell') return 'ell';
    return 'csr';
  }

  private buildHistogramIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): HistogramIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const loads: IRLoad[] = [{
      source: inputArrays[0] || 'input_arr',
      target: 'input_tile',
      index: '(pid,)',
      shape: `(${tileConfig.tileSize},)`,
    }];

    const operations: IROperation[] = [{
      type: 'atomic',
      inputs: ['input_tile'],
      output: 'histogram',
      op: 'histogram_atomic',
    }];

    const stores: IRStore[] = [];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'histogram',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
      variant: pattern.variant,
      histogramConfig: {
        numBins: 256,
        usePrivatization: pattern.variant === 'histogram_privatized',
      },
    };
  }

  private buildConvolutionIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): ConvolutionIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const loads: IRLoad[] = [
      {
        source: inputArrays[0] || 'input_arr',
        target: 'input_tile',
        index: '(b, c, h, w)',
        shape: '(tile_h, tile_w)',
      },
      {
        source: inputArrays[1] || 'weight',
        target: 'weight_tile',
        index: '(oc, ic, kh, kw)',
        shape: '(block_oc, block_ic, kernel_h, kernel_w)',
      },
    ];

    const operations: IROperation[] = [{
      type: 'matmul',
      inputs: ['input_tile', 'weight_tile'],
      output: 'conv_out',
      op: 'conv2d',
    }];

    const stores: IRStore[] = [{
      source: 'conv_out',
      target: outputArrays[0] || 'output',
      index: '(b, oc, oh, ow)',
    }];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'convolution',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
      variant: pattern.variant,
      convConfig: {
        convType: this.detectConvType(pattern),
        kernelSize: [3, 3],
        stride: [1, 1],
        padding: [0, 0],
        groups: 1,
      },
    };
  }

  private detectConvType(pattern: PatternMatch): ConvolutionIR['convConfig']['convType'] {
    if (pattern.variant?.includes('1d')) return 'conv1d';
    if (pattern.variant?.includes('3d')) return 'conv3d';
    if (pattern.variant?.includes('depthwise')) return 'depthwise';
    if (pattern.variant?.includes('grouped')) return 'grouped';
    return 'conv2d';
  }

  private buildSortingIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): SortingIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const loads: IRLoad[] = [{
      source: inputArrays[0] || 'data',
      target: 'data_tile',
      index: '(pid,)',
      shape: `(${tileConfig.tileSize},)`,
    }];

    const operations: IROperation[] = [{
      type: 'elementwise',
      inputs: ['data_tile'],
      output: 'sorted',
      op: 'bitonic_sort',
    }];

    const stores: IRStore[] = [{
      source: 'sorted',
      target: outputArrays[0] || inputArrays[0] || 'data',
      index: '(pid,)',
    }];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'sorting',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
      variant: pattern.variant,
      sortConfig: {
        sortType: this.detectSortType(pattern),
        direction: 'ascending',
      },
    };
  }

  private detectSortType(pattern: PatternMatch): SortingIR['sortConfig']['sortType'] {
    if (pattern.variant === 'radix_sort') return 'radix';
    if (pattern.variant === 'merge_sort') return 'merge';
    return 'bitonic';
  }

  private buildPoolingIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): PoolingIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const loads: IRLoad[] = [{
      source: inputArrays[0] || 'input_arr',
      target: 'input_tile',
      index: '(b, c, h, w)',
      shape: '(tile_h + pool_h, tile_w + pool_w)',
    }];

    const poolType = this.detectPoolType(pattern);
    const operations: IROperation[] = [{
      type: 'reduce',
      inputs: ['input_tile'],
      output: 'pooled',
      op: poolType === 'max' || poolType === 'global_max' ? 'max' : 'mean',
    }];

    const stores: IRStore[] = [{
      source: 'pooled',
      target: outputArrays[0] || 'output',
      index: '(b, c, oh, ow)',
    }];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'pooling',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
      variant: pattern.variant,
      poolConfig: {
        poolType,
        kernelSize: [2, 2],
        stride: [2, 2],
        padding: [0, 0],
      },
    };
  }

  private detectPoolType(pattern: PatternMatch): PoolingIR['poolConfig']['poolType'] {
    if (pattern.variant?.includes('max')) return pattern.variant.includes('global') ? 'global_max' : 'max';
    if (pattern.variant?.includes('global')) return 'global_avg';
    return 'avg';
  }

  private buildNormalizationIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): NormalizationIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const loads: IRLoad[] = [
      {
        source: inputArrays[0] || 'input_arr',
        target: 'input_tile',
        index: '(batch, hidden)',
        shape: `(1, ${tileConfig.tileSize})`,
      },
    ];

    const operations: IROperation[] = [
      {
        type: 'reduce',
        inputs: ['input_tile'],
        output: 'mean',
        op: 'mean',
        axis: 1,
      },
      {
        type: 'reduce',
        inputs: ['input_tile', 'mean'],
        output: 'var',
        op: 'variance',
        axis: 1,
      },
      {
        type: 'elementwise',
        inputs: ['input_tile', 'mean', 'var'],
        output: 'normalized',
        op: 'normalize',
      },
    ];

    const stores: IRStore[] = [{
      source: 'normalized',
      target: outputArrays[0] || 'output',
      index: '(batch, hidden)',
    }];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'normalization',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
      variant: pattern.variant,
      normConfig: {
        normType: this.detectNormType(pattern),
        epsilon: 1e-5,
        affine: true,
      },
    };
  }

  private detectNormType(pattern: PatternMatch): NormalizationIR['normConfig']['normType'] {
    if (pattern.variant === 'rmsnorm') return 'rmsnorm';
    if (pattern.variant === 'batchnorm') return 'batchnorm';
    if (pattern.variant === 'groupnorm') return 'groupnorm';
    if (pattern.variant === 'instancenorm') return 'instancenorm';
    return 'layernorm';
  }

  private buildEmbeddingIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): EmbeddingIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const loads: IRLoad[] = [
      {
        source: inputArrays[0] || 'indices',
        target: 'indices_tile',
        index: '(pid,)',
        shape: `(${tileConfig.tileSize},)`,
      },
      {
        source: inputArrays[1] || 'embedding_table',
        target: 'embed_tile',
        index: '(idx, dim)',
        shape: '(1, embed_dim)',
      },
    ];

    const operations: IROperation[] = [{
      type: 'elementwise',
      inputs: ['indices_tile', 'embed_tile'],
      output: 'output',
      op: 'gather',
    }];

    const stores: IRStore[] = [{
      source: 'output',
      target: outputArrays[0] || 'output',
      index: '(pid, dim)',
    }];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'embedding',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
      variant: pattern.variant,
      embeddingConfig: {
        vocabSize: 32000,
        embeddingDim: 4096,
      },
    };
  }

  private buildRoPEIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): RoPEIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const loads: IRLoad[] = [{
      source: inputArrays[0] || 'x',
      target: 'x_tile',
      index: '(batch, head, seq, dim)',
      shape: `(1, 1, 1, ${tileConfig.tileSize})`,
    }];

    const operations: IROperation[] = [{
      type: 'elementwise',
      inputs: ['x_tile'],
      output: 'rotated',
      op: 'rope_rotate',
    }];

    const stores: IRStore[] = [{
      source: 'rotated',
      target: outputArrays[0] || inputArrays[0] || 'x',
      index: '(batch, head, seq, dim)',
    }];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'rope',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
      variant: pattern.variant,
      ropeConfig: {
        headDim: 128,
        rotaryDim: 128,
        base: 10000,
      },
    };
  }

  private buildKVCacheIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): KVCacheIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const loads: IRLoad[] = [
      {
        source: inputArrays[0] || 'k_new',
        target: 'k_tile',
        index: '(batch, head, seq, dim)',
        shape: `(1, 1, 1, ${tileConfig.tileSize})`,
      },
      {
        source: inputArrays[1] || 'v_new',
        target: 'v_tile',
        index: '(batch, head, seq, dim)',
        shape: `(1, 1, 1, ${tileConfig.tileSize})`,
      },
    ];

    const operations: IROperation[] = [{
      type: 'elementwise',
      inputs: ['k_tile', 'v_tile'],
      output: 'cached',
      op: 'cache_append',
    }];

    const stores: IRStore[] = [
      {
        source: 'k_tile',
        target: outputArrays[0] || 'k_cache',
        index: '(batch, head, pos, dim)',
      },
      {
        source: 'v_tile',
        target: outputArrays[1] || 'v_cache',
        index: '(batch, head, pos, dim)',
      },
    ];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'kvcache',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
      variant: pattern.variant,
      kvConfig: {
        isPaged: pattern.variant === 'kvcache_paged',
        blockSize: 16,
        numKVHeads: 8,
      },
    };
  }

  private buildQuantizationIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): QuantizationIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const loads: IRLoad[] = [{
      source: inputArrays[0] || 'input_arr',
      target: 'input_tile',
      index: '(pid,)',
      shape: `(${tileConfig.tileSize},)`,
    }];

    const isQuantize = pattern.variant === 'quantize' || pattern.variant?.includes('quant');
    const operations: IROperation[] = [{
      type: 'elementwise',
      inputs: ['input_tile'],
      output: 'quant_out',
      op: isQuantize ? 'quantize' : 'dequantize',
    }];

    const stores: IRStore[] = [{
      source: 'quant_out',
      target: outputArrays[0] || 'output',
      index: '(pid,)',
    }];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'quantization',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
      variant: pattern.variant,
      quantConfig: {
        bits: this.detectQuantBits(pattern),
        isSymmetric: true,
        perChannel: false,
      },
    };
  }

  private detectQuantBits(pattern: PatternMatch): number {
    if (pattern.variant?.includes('int4') || pattern.variant?.includes('4')) return 4;
    if (pattern.variant?.includes('fp8') || pattern.variant?.includes('8')) return 8;
    return 8;
  }
}

export const irBuilder = new IRBuilder();
