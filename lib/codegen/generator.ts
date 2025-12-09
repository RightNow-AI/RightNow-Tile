// cuTile Code Generator - Transforms IR to cuTile Python code

import { KernelIR, KernelArchetype } from '../ast/types';
import {
  AttentionIR,
  FusedIR,
  FFTIR,
  SparseIR,
  HistogramIR,
  ConvolutionIR,
  SortingIR,
  PoolingIR,
  NormalizationIR,
  EmbeddingIR,
  RoPEIR,
  KVCacheIR,
  QuantizationIR,
} from '../ir/builder';
import { getAttentionGenerator } from './templates/attention';
import { getFusedGenerator } from './templates/fused';
import { getSparseGenerator } from './templates/sparse';
import { getHistogramGenerator } from './templates/histogram';
import { getConvolutionGenerator } from './templates/convolution';
import { getSortingGenerator } from './templates/sorting';
import { getPoolingGenerator } from './templates/pooling';
import { getNormalizationGenerator } from './templates/normalization';
import { getEmbeddingGenerator } from './templates/embedding';
import { getRoPEGenerator } from './templates/rope';
import { getKVCacheGenerator } from './templates/kvcache';
import { getQuantizationGenerator } from './templates/quantization';
import { EnhancedKernelIR } from '../ir/types';

export class CuTileCodeGenerator {
  /**
   * Generate cuTile Python code from IR
   */
  generate(ir: KernelIR): string {
    switch (ir.archetype) {
      case 'attention':
        return this.generateAttention(ir as unknown as AttentionIR);
      case 'fused':
        return this.generateFused(ir as unknown as FusedIR);
      case 'fft':
        return this.generateFFT(ir as unknown as FFTIR);
      case 'gemm':
        return this.generateGEMM(ir);
      case 'reduction':
        return this.generateReduction(ir);
      case 'scan':
        return this.generateScan(ir);
      case 'stencil':
        return this.generateStencil(ir);
      case 'sparse':
        return this.generateSparse(ir as unknown as SparseIR);
      case 'histogram':
        return this.generateHistogram(ir as unknown as HistogramIR);
      case 'convolution':
        return this.generateConvolution(ir as unknown as ConvolutionIR);
      case 'sorting':
        return this.generateSorting(ir as unknown as SortingIR);
      case 'pooling':
        return this.generatePooling(ir as unknown as PoolingIR);
      case 'normalization':
        return this.generateNormalization(ir as unknown as NormalizationIR);
      case 'embedding':
        return this.generateEmbedding(ir as unknown as EmbeddingIR);
      case 'rope':
        return this.generateRoPE(ir as unknown as RoPEIR);
      case 'kvcache':
        return this.generateKVCache(ir as unknown as KVCacheIR);
      case 'quantization':
        return this.generateQuantization(ir as unknown as QuantizationIR);
      case 'elementwise':
      default:
        return this.generateElementwise(ir);
    }
  }

  /**
   * Generate attention kernel code
   */
  private generateAttention(ir: AttentionIR): string {
    const enhancedIR = this.toEnhancedIR(ir) as EnhancedKernelIR;
    const generator = getAttentionGenerator(ir.variant as string | undefined);
    return generator({ ...enhancedIR, attentionConfig: ir.attentionConfig } as any);
  }

  /**
   * Generate fused kernel code
   */
  private generateFused(ir: FusedIR): string {
    const enhancedIR = this.toEnhancedIR(ir) as EnhancedKernelIR;
    const generator = getFusedGenerator(ir.variant as string | undefined);
    return generator({ ...enhancedIR, fusedOperations: ir.fusedOperations } as any);
  }

  /**
   * Generate sparse kernel code
   */
  private generateSparse(ir: SparseIR): string {
    const enhancedIR = this.toEnhancedIR(ir) as EnhancedKernelIR;
    const generator = getSparseGenerator(ir.variant as string | undefined);
    return generator({ ...enhancedIR, sparseConfig: ir.sparseConfig } as any);
  }

  /**
   * Generate histogram kernel code
   */
  private generateHistogram(ir: HistogramIR): string {
    const enhancedIR = this.toEnhancedIR(ir) as EnhancedKernelIR;
    const generator = getHistogramGenerator(ir.variant as string | undefined);
    return generator({ ...enhancedIR, histogramConfig: ir.histogramConfig } as any);
  }

  /**
   * Generate convolution kernel code
   */
  private generateConvolution(ir: ConvolutionIR): string {
    const enhancedIR = this.toEnhancedIR(ir) as EnhancedKernelIR;
    const generator = getConvolutionGenerator(ir.variant as string | undefined);
    return generator({ ...enhancedIR, convConfig: ir.convConfig } as any);
  }

  /**
   * Generate sorting kernel code
   */
  private generateSorting(ir: SortingIR): string {
    const enhancedIR = this.toEnhancedIR(ir) as EnhancedKernelIR;
    const generator = getSortingGenerator(ir.variant as string | undefined);
    return generator({ ...enhancedIR, sortConfig: ir.sortConfig } as any);
  }

  /**
   * Generate pooling kernel code
   */
  private generatePooling(ir: PoolingIR): string {
    const enhancedIR = this.toEnhancedIR(ir) as EnhancedKernelIR;
    const generator = getPoolingGenerator(ir.variant as string | undefined);
    return generator({ ...enhancedIR, poolConfig: ir.poolConfig } as any);
  }

  /**
   * Generate normalization kernel code
   */
  private generateNormalization(ir: NormalizationIR): string {
    const enhancedIR = this.toEnhancedIR(ir) as EnhancedKernelIR;
    const generator = getNormalizationGenerator(ir.variant as string | undefined);
    return generator({ ...enhancedIR, normConfig: ir.normConfig } as any);
  }

  /**
   * Generate embedding kernel code
   */
  private generateEmbedding(ir: EmbeddingIR): string {
    const enhancedIR = this.toEnhancedIR(ir) as EnhancedKernelIR;
    const generator = getEmbeddingGenerator(ir.variant as string | undefined);
    return generator({ ...enhancedIR, embeddingConfig: ir.embeddingConfig } as any);
  }

  /**
   * Generate RoPE kernel code
   */
  private generateRoPE(ir: RoPEIR): string {
    const enhancedIR = this.toEnhancedIR(ir) as EnhancedKernelIR;
    const generator = getRoPEGenerator(ir.variant as string | undefined);
    return generator({ ...enhancedIR, ropeConfig: ir.ropeConfig } as any);
  }

  /**
   * Generate KV cache kernel code
   */
  private generateKVCache(ir: KVCacheIR): string {
    const enhancedIR = this.toEnhancedIR(ir) as EnhancedKernelIR;
    const generator = getKVCacheGenerator(ir.variant as string | undefined);
    return generator({ ...enhancedIR, kvConfig: ir.kvConfig } as any);
  }

  /**
   * Generate quantization kernel code
   */
  private generateQuantization(ir: QuantizationIR): string {
    const enhancedIR = this.toEnhancedIR(ir) as EnhancedKernelIR;
    const generator = getQuantizationGenerator(ir.variant as string | undefined);
    return generator({ ...enhancedIR, quantConfig: ir.quantConfig } as any);
  }

  /**
   * Generate FFT kernel code
   */
  private generateFFT(ir: FFTIR): string {
    const tileSize = ir.tileConfig.tileSize || 256;
    const isInverse = ir.fftConfig?.isInverse || false;
    const radix = ir.fftConfig?.radix || 2;

    const inputName = ir.loads[0]?.source || 'data';
    const outputName = ir.stores[0]?.target || inputName;

    return `import cuda_tile as ct
import cupy
import math

TILE_SIZE = ${tileSize}
RADIX = ${radix}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName},
    n: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    ${isInverse ? 'Inverse ' : ''}FFT kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: fft_radix${radix}

    Performs ${isInverse ? 'inverse ' : ''}Fast Fourier Transform
    using radix-${radix} algorithm.
    """
    pid = ct.bid(0)
    tid = ct.tid(0)

    # Load input data
    tile_real = ct.load(${inputName}, index=(pid * 2,), shape=(tile_size,))
    tile_imag = ct.load(${inputName}, index=(pid * 2 + 1,), shape=(tile_size,))

    # Bit-reversal permutation
    reversed_idx = ct.bit_reverse(tid, ct.log2(tile_size))
    if tid < reversed_idx:
        # Swap real parts
        temp_r = tile_real[tid]
        tile_real[tid] = tile_real[reversed_idx]
        tile_real[reversed_idx] = temp_r
        # Swap imag parts
        temp_i = tile_imag[tid]
        tile_imag[tid] = tile_imag[reversed_idx]
        tile_imag[reversed_idx] = temp_i

    ct.sync_threads()

    # FFT butterfly stages
    stage = 1
    while stage < tile_size:
        # Twiddle factor angle
        angle = ${isInverse ? '' : '-'}2.0 * math.pi * ct.float32(tid % stage) / ct.float32(stage * 2)
        w_r = ct.cos(angle)
        w_i = ct.sin(angle)

        # Butterfly indices
        even_idx = (tid // stage) * stage * 2 + (tid % stage)
        odd_idx = even_idx + stage

        if even_idx < tile_size and odd_idx < tile_size:
            # Load even and odd values
            e_r = tile_real[even_idx]
            e_i = tile_imag[even_idx]
            o_r = tile_real[odd_idx]
            o_i = tile_imag[odd_idx]

            # Complex multiply: (o_r + i*o_i) * (w_r + i*w_i)
            t_r = o_r * w_r - o_i * w_i
            t_i = o_r * w_i + o_i * w_r

            # Butterfly
            tile_real[even_idx] = e_r + t_r
            tile_imag[even_idx] = e_i + t_i
            tile_real[odd_idx] = e_r - t_r
            tile_imag[odd_idx] = e_i - t_i

        ct.sync_threads()
        stage *= 2

    ${isInverse ? `
    # Scale by 1/N for inverse FFT
    scale = 1.0 / ct.float32(tile_size)
    tile_real[tid] = tile_real[tid] * scale
    tile_imag[tid] = tile_imag[tid] * scale
    ` : ''}

    # Store results
    ct.store(${outputName}, index=(pid * 2,), tile=tile_real)
    ct.store(${outputName}, index=(pid * 2 + 1,), tile=tile_imag)


def launch_${ir.name}(${inputName}, ${outputName}=None):
    """Launch the ${ir.name} FFT kernel"""
    if ${outputName} is None:
        ${outputName} = ${inputName}  # In-place transform
    n = ${inputName}.shape[0] // 2  # Complex pairs
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, n, TILE_SIZE))`;
  }

  /**
   * Convert basic KernelIR to EnhancedKernelIR format for templates
   */
  private toEnhancedIR(ir: KernelIR): Partial<EnhancedKernelIR> {
    return {
      name: ir.name,
      originalName: ir.originalName,
      archetype: ir.archetype,
      variant: (ir as any).variant,
      confidence: ir.confidence,
      parameters: ir.parameters.map(p => ({
        cudaName: p.cudaName,
        cuTileName: p.cuTileName,
        cudaType: p.type,
        cuTileType: p.type.includes('float') ? 'ct.float32' : 'ct.int32',
        isPointer: !p.isConstant,
        isConstant: p.isConstant,
        constantAnnotation: p.constantAnnotation,
      })),
      loads: ir.loads.map(l => ({
        ...l,
        dtype: 'ct.float32',
        accessPattern: 'coalesced' as const,
      })),
      operations: ir.operations.map(op => ({
        ...op,
        dtype: 'ct.float32',
      })),
      stores: ir.stores.map(s => ({
        ...s,
        accessPattern: 'coalesced' as const,
      })),
      tileConfig: ir.tileConfig,
      tileStrategy: {
        approach: 'blocked' as const,
        dimensions: [],
        justification: 'Default tiling strategy',
      },
      semanticInfo: {
        dataTypes: new Map(),
        inputArrays: ir.loads.map(l => l.source),
        outputArrays: ir.stores.map(s => s.target),
        intermediates: [],
        hasDataDependency: false,
        isThreadSafe: true,
      },
      optimizationHints: [],
      memoryLayout: {
        totalSharedMemory: 0,
        registersPerThread: 32,
        globalMemoryReads: ir.loads.length,
        globalMemoryWrites: ir.stores.length,
        sharedMemoryBankConflictFree: true,
      },
    };
  }

  private generateElementwise(ir: KernelIR): string {
    const tileSize = ir.tileConfig.tileSize;

    // Get input parameter names (pointers)
    const pointerParams = ir.parameters.filter(p => !p.isConstant);
    const constantParams = ir.parameters.filter(p => p.isConstant);

    const paramList = [
      ...pointerParams.map(p => p.cuTileName),
      ...constantParams.map(p => `${p.cuTileName}: ${p.constantAnnotation}`),
      `tile_size: ct.Constant[int]`,
    ].join(', ');

    // Generate loads - only for actual inputs (not outputs)
    const loadLines = ir.loads.length > 0
      ? ir.loads.map(load =>
          `    ${load.target.toLowerCase()} = ct.load(${load.source.toLowerCase()}, index=(pid,), shape=(tile_size,))`
        ).join('\n')
      : '    # No input loads needed';

    // Generate operations
    const opLines = ir.operations.length > 0
      ? ir.operations.map(op => {
          if (op.op === 'identity') {
            return `    result = ${op.inputs[0].toLowerCase()}`;
          } else if (op.inputs.length === 2) {
            return `    result = ${op.inputs[0].toLowerCase()} ${op.op} ${op.inputs[1].toLowerCase()}`;
          } else {
            return `    result = ${op.inputs[0].toLowerCase()}`;
          }
        }).join('\n')
      : '    result = a_tile  # Default pass-through';

    // Generate stores - should only store to output arrays
    const storeLines = ir.stores.length > 0
      ? ir.stores.map(store =>
          `    ct.store(${store.target.toLowerCase()}, index=(pid,), tile=${store.source.toLowerCase()})`
        ).join('\n')
      : '    ct.store(c, index=(pid,), tile=result)';

    const firstPointer = pointerParams[0]?.cuTileName || 'a';

    return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(${paramList}):
    """
    Elementwise kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    """
    pid = ct.bid(0)

    # Load input tiles
${loadLines}

    # Compute
${opLines}

    # Store result
${storeLines}


def launch_${ir.name}(${pointerParams.map(p => p.cuTileName).join(', ')}):
    """Launch the ${ir.name} kernel"""
    n = ${firstPointer}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${[...pointerParams.map(p => p.cuTileName), 'TILE_SIZE'].join(', ')}))`;
  }

  private generateGEMM(ir: KernelIR): string {
    const { blockM = 128, blockN = 128, blockK = 32 } = ir.tileConfig;

    // Extract matrix names from parameters
    const pointerParams = ir.parameters.filter(p => !p.isConstant);
    const [aParam, bParam, cParam] = pointerParams.length >= 3
      ? pointerParams.map(p => p.cuTileName)
      : ['a', 'b', 'c'];

    return `import cuda_tile as ct
import cupy

# Tile configuration
BLOCK_M = ${blockM}
BLOCK_N = ${blockN}
BLOCK_K = ${blockK}

@ct.kernel
def ${ir.name}(
    ${aParam}, ${bParam}, ${cParam},
    M: ct.Constant[int], N: ct.Constant[int], K: ct.Constant[int],
    block_m: ct.Constant[int], block_n: ct.Constant[int], block_k: ct.Constant[int]
):
    """
    GEMM kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%

    Computes C = A @ B
    """
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    # Initialize accumulator tile
    acc = ct.zeros((block_m, block_n), dtype=ct.float32)

    # Loop over K dimension in tiles
    for k in range(0, K, block_k):
        # Load A tile: (pid_m * block_m, k) -> (block_m, block_k)
        a_tile = ct.load(${aParam}, index=(pid_m, k // block_k), shape=(block_m, block_k))

        # Load B tile: (k, pid_n * block_n) -> (block_k, block_n)
        b_tile = ct.load(${bParam}, index=(k // block_k, pid_n), shape=(block_k, block_n))

        # Accumulate: acc += A @ B
        acc = ct.tile_matmul(a_tile, b_tile, acc)

    # Store result tile
    ct.store(${cParam}, index=(pid_m, pid_n), tile=acc)


def launch_${ir.name}(${aParam}, ${bParam}, ${cParam}):
    """Launch the ${ir.name} GEMM kernel"""
    M, K = ${aParam}.shape
    K2, N = ${bParam}.shape
    assert K == K2, "Matrix dimensions must match"

    grid = (ct.cdiv(M, BLOCK_M), ct.cdiv(N, BLOCK_N), 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${aParam}, ${bParam}, ${cParam}, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K))`;
  }

  private generateReduction(ir: KernelIR): string {
    const tileSize = ir.tileConfig.tileSize;

    // Extract input/output names
    const pointerParams = ir.parameters.filter(p => !p.isConstant);
    const inputName = pointerParams[0]?.cuTileName || 'input_arr';
    const outputName = pointerParams[1]?.cuTileName || 'output_arr';

    // Get reduction operation
    const reductionOp = ir.operations.find(op => op.type === 'reduce')?.op || 'sum';

    return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(${inputName}, ${outputName}, tile_size: ct.Constant[int]):
    """
    Reduction kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%

    Performs parallel ${reductionOp} reduction
    """
    pid = ct.bid(0)

    # Load tile from global memory
    tile = ct.load(${inputName}, index=(pid,), shape=(tile_size,))

    # Reduce tile to scalar
    result = ct.reduce(tile, op=ct.${reductionOp})

    # Atomic accumulation to global result
    ct.atomic_add(${outputName}, 0, result)


def launch_${ir.name}(${inputName}, ${outputName}):
    """Launch the ${ir.name} reduction kernel"""
    n = ${inputName}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, TILE_SIZE))`;
  }

  private generateScan(ir: KernelIR): string {
    const tileSize = ir.tileConfig.tileSize;

    const pointerParams = ir.parameters.filter(p => !p.isConstant);
    const inputName = pointerParams[0]?.cuTileName || 'data';
    const outputName = pointerParams[1]?.cuTileName || inputName;

    return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(${inputName}, ${outputName}, tile_size: ct.Constant[int]):
    """
    Scan (prefix sum) kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%

    Performs parallel inclusive prefix sum
    """
    pid = ct.bid(0)

    # Load tile
    tile = ct.load(${inputName}, index=(pid,), shape=(tile_size,))

    # Perform inclusive scan (cumulative sum)
    scanned = ct.cumsum(tile, axis=0)

    # Store scanned result
    ct.store(${outputName}, index=(pid,), tile=scanned)


def launch_${ir.name}(${inputName}${inputName !== outputName ? `, ${outputName}` : ''}):
    """Launch the ${ir.name} scan kernel"""
    n = ${inputName}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, TILE_SIZE))`;
  }

  private generateStencil(ir: KernelIR): string {
    const tileSize = ir.tileConfig.tileSize;

    const pointerParams = ir.parameters.filter(p => !p.isConstant);
    const inputName = pointerParams[0]?.cuTileName || 'input_arr';
    const outputName = pointerParams[1]?.cuTileName || 'output_arr';

    return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
HALO = 1  # Halo region for stencil neighbors

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName},
    width: ct.Constant[int], height: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Stencil kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%

    Applies stencil pattern to 2D grid
    """
    pid_x = ct.bid(0)
    pid_y = ct.bid(1)

    # Calculate global coordinates
    gx = pid_x * tile_size
    gy = pid_y * tile_size

    # Load tile with halo region
    tile = ct.load(
        ${inputName},
        index=(gy - 1, gx - 1),
        shape=(tile_size + 2, tile_size + 2)
    )

    # Apply 5-point stencil (up, down, left, right)
    center = tile[1:-1, 1:-1]
    up = tile[:-2, 1:-1]
    down = tile[2:, 1:-1]
    left = tile[1:-1, :-2]
    right = tile[1:-1, 2:]

    # Weighted average (Jacobi-like stencil)
    result = 0.25 * (up + down + left + right)

    # Store result
    ct.store(${outputName}, index=(gy, gx), tile=result)


def launch_${ir.name}(${inputName}, ${outputName}, width, height):
    """Launch the ${ir.name} stencil kernel"""
    grid_x = ct.cdiv(width, TILE_SIZE)
    grid_y = ct.cdiv(height, TILE_SIZE)
    grid = (grid_x, grid_y, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, width, height, TILE_SIZE))`;
  }
}

export const codeGenerator = new CuTileCodeGenerator();
