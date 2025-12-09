/**
 * Stencil Pattern Templates
 * Specialized code generation for stencil variants
 */

import { EnhancedKernelIR } from '../../ir/types';

export function generateStencil1D3Point(ir: EnhancedKernelIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const inputName = ir.semanticInfo.inputArrays[0] || 'input_arr';
  const outputName = ir.semanticInfo.outputArrays[0] || 'output_arr';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
HALO = 1

@ct.kernel
def ${ir.name}(${inputName}, ${outputName}, n: ct.Constant[int], tile_size: ct.Constant[int]):
    """
    1D 3-Point Stencil kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: stencil_1d_3pt
    """
    pid = ct.bid(0)
    tid = ct.tid(0)

    # Global index
    gid = pid * tile_size + tid

    # Load tile with halo
    tile = ct.load(${inputName}, index=(pid * tile_size - HALO,), shape=(tile_size + 2 * HALO,))

    # Apply 3-point stencil
    left = tile[:-2]
    center = tile[1:-1]
    right = tile[2:]

    result = 0.25 * left + 0.5 * center + 0.25 * right

    # Store result
    ct.store(${outputName}, index=(pid * tile_size,), tile=result, mask=(gid < n - 1) & (gid > 0))


def launch_${ir.name}(${inputName}, ${outputName}):
    """Launch the ${ir.name} 1D stencil kernel"""
    n = ${inputName}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, n, TILE_SIZE))`;
}

export function generateStencil1D5Point(ir: EnhancedKernelIR): string {
  const tileSize = ir.tileConfig.tileSize || 128;
  const inputName = ir.semanticInfo.inputArrays[0] || 'input_arr';
  const outputName = ir.semanticInfo.outputArrays[0] || 'output_arr';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
HALO = 2

@ct.kernel
def ${ir.name}(${inputName}, ${outputName}, n: ct.Constant[int], tile_size: ct.Constant[int]):
    """
    1D 5-Point Stencil kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: stencil_1d_5pt
    """
    pid = ct.bid(0)

    # Load tile with extended halo
    tile = ct.load(${inputName}, index=(pid * tile_size - HALO,), shape=(tile_size + 2 * HALO,))

    # Apply 5-point stencil
    p2 = tile[:-4]
    p1 = tile[1:-3]
    c = tile[2:-2]
    n1 = tile[3:-1]
    n2 = tile[4:]

    result = (-p2 + 16*p1 - 30*c + 16*n1 - n2) / 12.0

    ct.store(${outputName}, index=(pid * tile_size,), tile=result)


def launch_${ir.name}(${inputName}, ${outputName}):
    """Launch the ${ir.name} 5-point stencil kernel"""
    n = ${inputName}.shape[0]
    grid = (ct.cdiv(n - 4, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, n, TILE_SIZE))`;
}

export function generateStencil2D5Point(ir: EnhancedKernelIR): string {
  const blockM = ir.tileConfig.blockM || 32;
  const blockN = ir.tileConfig.blockN || 8;
  const inputName = ir.semanticInfo.inputArrays[0] || 'input_arr';
  const outputName = ir.semanticInfo.outputArrays[0] || 'output_arr';

  return `import cuda_tile as ct
import cupy

BLOCK_M = ${blockM}
BLOCK_N = ${blockN}
HALO = 1

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName},
    width: ct.Constant[int], height: ct.Constant[int],
    block_m: ct.Constant[int], block_n: ct.Constant[int]
):
    """
    2D 5-Point Stencil kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: stencil_2d_5pt
    """
    pid_x = ct.bid(0)
    pid_y = ct.bid(1)

    gx = pid_x * block_m
    gy = pid_y * block_n

    # Load tile with halo
    tile = ct.load(
        ${inputName},
        index=(gy - HALO, gx - HALO),
        shape=(block_n + 2 * HALO, block_m + 2 * HALO)
    )

    # Extract stencil points
    center = tile[HALO:-HALO, HALO:-HALO]
    up = tile[:-2*HALO, HALO:-HALO]
    down = tile[2*HALO:, HALO:-HALO]
    left = tile[HALO:-HALO, :-2*HALO]
    right = tile[HALO:-HALO, 2*HALO:]

    result = 0.25 * (up + down + left + right)

    ct.store(${outputName}, index=(gy, gx), tile=result)


def launch_${ir.name}(${inputName}, ${outputName}, width, height):
    """Launch the ${ir.name} 2D stencil kernel"""
    grid_x = ct.cdiv(width - 2, BLOCK_M)
    grid_y = ct.cdiv(height - 2, BLOCK_N)
    grid = (grid_x, grid_y, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, width, height, BLOCK_M, BLOCK_N))`;
}

export function generateStencil2D9Point(ir: EnhancedKernelIR): string {
  const blockM = ir.tileConfig.blockM || 16;
  const blockN = ir.tileConfig.blockN || 16;
  const inputName = ir.semanticInfo.inputArrays[0] || 'input_arr';
  const outputName = ir.semanticInfo.outputArrays[0] || 'output_arr';

  return `import cuda_tile as ct
import cupy

BLOCK_M = ${blockM}
BLOCK_N = ${blockN}
HALO = 1

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName},
    width: ct.Constant[int], height: ct.Constant[int],
    block_m: ct.Constant[int], block_n: ct.Constant[int]
):
    """
    2D 9-Point Stencil kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: stencil_2d_9pt
    """
    pid_x = ct.bid(0)
    pid_y = ct.bid(1)

    gx = pid_x * block_m
    gy = pid_y * block_n

    tile = ct.load(
        ${inputName},
        index=(gy - HALO, gx - HALO),
        shape=(block_n + 2 * HALO, block_m + 2 * HALO)
    )

    # Extract all 9 points
    nw = tile[:-2, :-2]
    n = tile[:-2, 1:-1]
    ne = tile[:-2, 2:]
    w = tile[1:-1, :-2]
    c = tile[1:-1, 1:-1]
    e = tile[1:-1, 2:]
    sw = tile[2:, :-2]
    s = tile[2:, 1:-1]
    se = tile[2:, 2:]

    result = 0.5 * (nw + ne + sw + se) + (n + s + w + e) - 6.0 * c

    ct.store(${outputName}, index=(gy, gx), tile=result)


def launch_${ir.name}(${inputName}, ${outputName}, width, height):
    """Launch the ${ir.name} 9-point stencil kernel"""
    grid_x = ct.cdiv(width - 2, BLOCK_M)
    grid_y = ct.cdiv(height - 2, BLOCK_N)
    grid = (grid_x, grid_y, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, width, height, BLOCK_M, BLOCK_N))`;
}

export function generateStencil3D(ir: EnhancedKernelIR): string {
  const blockSize = ir.tileConfig.tileSize || 8;
  const inputName = ir.semanticInfo.inputArrays[0] || 'input_arr';
  const outputName = ir.semanticInfo.outputArrays[0] || 'output_arr';

  return `import cuda_tile as ct
import cupy

BLOCK_SIZE = ${blockSize}
HALO = 1

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName},
    nx: ct.Constant[int], ny: ct.Constant[int], nz: ct.Constant[int],
    block_size: ct.Constant[int]
):
    """
    3D 7-Point Stencil kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: stencil_3d
    """
    pid_x = ct.bid(0)
    pid_y = ct.bid(1)
    pid_z = ct.bid(2)

    gx = pid_x * block_size
    gy = pid_y * block_size
    gz = pid_z * block_size

    tile = ct.load(
        ${inputName},
        index=(gz - HALO, gy - HALO, gx - HALO),
        shape=(block_size + 2*HALO, block_size + 2*HALO, block_size + 2*HALO)
    )

    # Extract 7 stencil points
    center = tile[HALO:-HALO, HALO:-HALO, HALO:-HALO]
    front = tile[:-2*HALO, HALO:-HALO, HALO:-HALO]
    back = tile[2*HALO:, HALO:-HALO, HALO:-HALO]
    top = tile[HALO:-HALO, :-2*HALO, HALO:-HALO]
    bottom = tile[HALO:-HALO, 2*HALO:, HALO:-HALO]
    left = tile[HALO:-HALO, HALO:-HALO, :-2*HALO]
    right = tile[HALO:-HALO, HALO:-HALO, 2*HALO:]

    result = (front + back + top + bottom + left + right - 6.0 * center)

    ct.store(${outputName}, index=(gz, gy, gx), tile=result)


def launch_${ir.name}(${inputName}, ${outputName}, nx, ny, nz):
    """Launch the ${ir.name} 3D stencil kernel"""
    grid = (ct.cdiv(nx-2, BLOCK_SIZE), ct.cdiv(ny-2, BLOCK_SIZE), ct.cdiv(nz-2, BLOCK_SIZE))
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, nx, ny, nz, BLOCK_SIZE))`;
}
