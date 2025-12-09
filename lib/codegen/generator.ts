// cuTile Code Generator - Transforms IR to cuTile Python code

import { KernelIR, KernelArchetype } from '../ast/types';

export class CuTileCodeGenerator {
  /**
   * Generate cuTile Python code from IR
   */
  generate(ir: KernelIR): string {
    switch (ir.archetype) {
      case 'gemm':
        return this.generateGEMM(ir);
      case 'reduction':
        return this.generateReduction(ir);
      case 'scan':
        return this.generateScan(ir);
      case 'stencil':
        return this.generateStencil(ir);
      case 'elementwise':
      default:
        return this.generateElementwise(ir);
    }
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
