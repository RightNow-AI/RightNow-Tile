/**
 * Sparse Matrix Pattern Templates
 * Specialized code generation for SpMV, SpMM, and related sparse operations
 * Supports CSR, COO, ELL, and hybrid formats
 */

import { EnhancedKernelIR } from '../../ir/types';

/**
 * Sparse IR extension for sparse-specific configuration
 */
export interface SparseIR extends EnhancedKernelIR {
  sparseConfig?: {
    format: 'csr' | 'coo' | 'ell' | 'hybrid' | 'bsr';
    vectorWidth?: number;
    warpsPerRow?: number;
    avgNnzPerRow?: number;
    useTexture?: boolean;
  };
}

/**
 * Generate SpMV CSR kernel (thread-per-row variant)
 * Sparse Matrix-Vector multiplication using Compressed Sparse Row format
 */
export function generateSpMVCSR(ir: SparseIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const vectorWidth = ir.sparseConfig?.vectorWidth || 1;

  const rowPtrName = ir.loads[0]?.source || 'row_ptr';
  const colIdxName = ir.loads[1]?.source || 'col_idx';
  const valuesName = ir.loads[2]?.source || 'values';
  const xName = ir.loads[3]?.source || 'x';
  const yName = ir.stores[0]?.target || 'y';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
WARP_SIZE = 32

@ct.kernel
def ${ir.name}(
    ${rowPtrName}, ${colIdxName}, ${valuesName}, ${xName}, ${yName},
    num_rows: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    SpMV CSR kernel (thread-per-row) - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: spmv_csr

    Each thread processes one row of the sparse matrix.
    Good for matrices with uniform row lengths.
    """
    row = ct.bid(0) * tile_size + ct.tid(0)

    if row >= num_rows:
        return

    # Get row bounds from CSR format
    row_start = ${rowPtrName}[row]
    row_end = ${rowPtrName}[row + 1]

    # Compute dot product for this row
    sum_val = ct.float32(0.0)

    for j in range(row_start, row_end):
        col = ${colIdxName}[j]
        val = ${valuesName}[j]
        sum_val = sum_val + val * ${xName}[col]

    # Store result
    ${yName}[row] = sum_val


def launch_${ir.name}(${rowPtrName}, ${colIdxName}, ${valuesName}, ${xName}, ${yName}):
    """Launch the ${ir.name} SpMV CSR kernel"""
    num_rows = ${rowPtrName}.shape[0] - 1
    grid = (ct.cdiv(num_rows, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${rowPtrName}, ${colIdxName}, ${valuesName}, ${xName}, ${yName}, num_rows, TILE_SIZE))`;
}

/**
 * Generate SpMV CSR kernel (warp-per-row variant)
 * Better for matrices with longer rows
 */
export function generateSpMVCSRWarp(ir: SparseIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const warpsPerRow = ir.sparseConfig?.warpsPerRow || 1;

  const rowPtrName = ir.loads[0]?.source || 'row_ptr';
  const colIdxName = ir.loads[1]?.source || 'col_idx';
  const valuesName = ir.loads[2]?.source || 'values';
  const xName = ir.loads[3]?.source || 'x';
  const yName = ir.stores[0]?.target || 'y';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
WARP_SIZE = 32
WARPS_PER_ROW = ${warpsPerRow}

@ct.kernel
def ${ir.name}(
    ${rowPtrName}, ${colIdxName}, ${valuesName}, ${xName}, ${yName},
    num_rows: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    SpMV CSR kernel (warp-per-row) - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: spmv_csr_warp

    Each warp processes one row using shuffle-based reduction.
    Better for matrices with long rows.
    """
    tid = ct.tid(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE
    row = ct.bid(0) * (tile_size // WARP_SIZE) + warp_id

    if row >= num_rows:
        return

    # Get row bounds
    row_start = ${rowPtrName}[row]
    row_end = ${rowPtrName}[row + 1]
    row_len = row_end - row_start

    # Each lane processes strided elements within the row
    sum_val = ct.float32(0.0)

    for j in range(lane, row_len, WARP_SIZE):
        idx = row_start + j
        col = ${colIdxName}[idx]
        val = ${valuesName}[idx]
        sum_val = sum_val + val * ${xName}[col]

    # Warp-level reduction using shuffle
    for offset in [16, 8, 4, 2, 1]:
        sum_val = sum_val + ct.shfl_down(sum_val, offset)

    # Lane 0 writes the result
    if lane == 0:
        ${yName}[row] = sum_val


def launch_${ir.name}(${rowPtrName}, ${colIdxName}, ${valuesName}, ${xName}, ${yName}):
    """Launch the ${ir.name} SpMV CSR warp kernel"""
    num_rows = ${rowPtrName}.shape[0] - 1
    warps_per_block = TILE_SIZE // WARP_SIZE
    grid = (ct.cdiv(num_rows, warps_per_block), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${rowPtrName}, ${colIdxName}, ${valuesName}, ${xName}, ${yName}, num_rows, TILE_SIZE))`;
}

/**
 * Generate SpMV COO kernel
 * Sparse Matrix-Vector multiplication using Coordinate format
 */
export function generateSpMVCOO(ir: SparseIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;

  const rowIdxName = ir.loads[0]?.source || 'row_idx';
  const colIdxName = ir.loads[1]?.source || 'col_idx';
  const valuesName = ir.loads[2]?.source || 'values';
  const xName = ir.loads[3]?.source || 'x';
  const yName = ir.stores[0]?.target || 'y';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(
    ${rowIdxName}, ${colIdxName}, ${valuesName}, ${xName}, ${yName},
    nnz: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    SpMV COO kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: spmv_coo

    Each thread processes one non-zero element.
    Uses atomic operations to accumulate results.
    """
    idx = ct.bid(0) * tile_size + ct.tid(0)

    if idx >= nnz:
        return

    row = ${rowIdxName}[idx]
    col = ${colIdxName}[idx]
    val = ${valuesName}[idx]

    # Atomic accumulation (COO may have multiple entries per row)
    ct.atomic_add(${yName}, row, val * ${xName}[col])


def launch_${ir.name}(${rowIdxName}, ${colIdxName}, ${valuesName}, ${xName}, ${yName}, nnz):
    """Launch the ${ir.name} SpMV COO kernel"""
    grid = (ct.cdiv(nnz, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${rowIdxName}, ${colIdxName}, ${valuesName}, ${xName}, ${yName}, nnz, TILE_SIZE))`;
}

/**
 * Generate SpMV ELL kernel
 * Sparse Matrix-Vector multiplication using ELLPACK format
 */
export function generateSpMVELL(ir: SparseIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;

  const colIdxName = ir.loads[0]?.source || 'col_idx';
  const valuesName = ir.loads[1]?.source || 'values';
  const xName = ir.loads[2]?.source || 'x';
  const yName = ir.stores[0]?.target || 'y';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(
    ${colIdxName}, ${valuesName}, ${xName}, ${yName},
    num_rows: ct.Constant[int],
    max_nnz_per_row: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    SpMV ELL kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: spmv_ell

    ELLPACK format: fixed number of elements per row (padded with zeros).
    Column-major storage for coalesced memory access.
    """
    row = ct.bid(0) * tile_size + ct.tid(0)

    if row >= num_rows:
        return

    sum_val = ct.float32(0.0)

    # ELL uses column-major layout: data[row + col * num_rows]
    for j in range(max_nnz_per_row):
        idx = row + j * num_rows
        col = ${colIdxName}[idx]

        # Skip padding (invalid column index)
        if col >= 0:
            val = ${valuesName}[idx]
            sum_val = sum_val + val * ${xName}[col]

    ${yName}[row] = sum_val


def launch_${ir.name}(${colIdxName}, ${valuesName}, ${xName}, ${yName}, max_nnz_per_row):
    """Launch the ${ir.name} SpMV ELL kernel"""
    num_rows = ${yName}.shape[0]
    grid = (ct.cdiv(num_rows, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${colIdxName}, ${valuesName}, ${xName}, ${yName}, num_rows, max_nnz_per_row, TILE_SIZE))`;
}

/**
 * Generate SpMM CSR kernel
 * Sparse Matrix-Matrix multiplication
 */
export function generateSpMMCSR(ir: SparseIR): string {
  const blockM = ir.tileConfig.blockM || 32;
  const blockN = ir.tileConfig.blockN || 64;

  const rowPtrName = ir.loads[0]?.source || 'row_ptr';
  const colIdxName = ir.loads[1]?.source || 'col_idx';
  const valuesName = ir.loads[2]?.source || 'values';
  const bName = ir.loads[3]?.source || 'B';
  const cName = ir.stores[0]?.target || 'C';

  return `import cuda_tile as ct
import cupy

BLOCK_M = ${blockM}
BLOCK_N = ${blockN}

@ct.kernel
def ${ir.name}(
    ${rowPtrName}, ${colIdxName}, ${valuesName}, ${bName}, ${cName},
    num_rows: ct.Constant[int],
    num_cols_b: ct.Constant[int],
    block_m: ct.Constant[int],
    block_n: ct.Constant[int]
):
    """
    SpMM CSR kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: spmm_csr

    Sparse A (CSR) @ Dense B = Dense C
    Each thread block handles a tile of output.
    """
    row_block = ct.bid(0)
    col_block = ct.bid(1)
    tid = ct.tid(0)

    row = row_block * block_m + (tid // block_n)
    col = col_block * block_n + (tid % block_n)

    if row >= num_rows or col >= num_cols_b:
        return

    # Get row bounds
    row_start = ${rowPtrName}[row]
    row_end = ${rowPtrName}[row + 1]

    # Accumulate dot product
    sum_val = ct.float32(0.0)

    for j in range(row_start, row_end):
        k = ${colIdxName}[j]
        val = ${valuesName}[j]
        sum_val = sum_val + val * ${bName}[k * num_cols_b + col]

    ${cName}[row * num_cols_b + col] = sum_val


def launch_${ir.name}(${rowPtrName}, ${colIdxName}, ${valuesName}, ${bName}, ${cName}):
    """Launch the ${ir.name} SpMM CSR kernel"""
    num_rows = ${rowPtrName}.shape[0] - 1
    num_cols_b = ${bName}.shape[1]
    grid = (ct.cdiv(num_rows, BLOCK_M), ct.cdiv(num_cols_b, BLOCK_N), 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${rowPtrName}, ${colIdxName}, ${valuesName}, ${bName}, ${cName}, num_rows, num_cols_b, BLOCK_M, BLOCK_N))`;
}

/**
 * Generate sparse SDDMM kernel
 * Sampled Dense-Dense Matrix Multiplication
 */
export function generateSDDMM(ir: SparseIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;

  const rowIdxName = ir.loads[0]?.source || 'row_idx';
  const colIdxName = ir.loads[1]?.source || 'col_idx';
  const aName = ir.loads[2]?.source || 'A';
  const bName = ir.loads[3]?.source || 'B';
  const cName = ir.stores[0]?.target || 'C';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(
    ${rowIdxName}, ${colIdxName}, ${aName}, ${bName}, ${cName},
    nnz: ct.Constant[int],
    k_dim: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    SDDMM kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: sddmm

    Sampled Dense-Dense Matrix Multiplication:
    C[i,j] = A[i,:] @ B[:,j] for non-zero positions only.
    Common in GNN backward pass.
    """
    idx = ct.bid(0) * tile_size + ct.tid(0)

    if idx >= nnz:
        return

    row = ${rowIdxName}[idx]
    col = ${colIdxName}[idx]

    # Compute dot product A[row, :] @ B[:, col]
    sum_val = ct.float32(0.0)

    for k in range(k_dim):
        sum_val = sum_val + ${aName}[row * k_dim + k] * ${bName}[k * ct.Constant[int] + col]

    ${cName}[idx] = sum_val


def launch_${ir.name}(${rowIdxName}, ${colIdxName}, ${aName}, ${bName}, ${cName}):
    """Launch the ${ir.name} SDDMM kernel"""
    nnz = ${rowIdxName}.shape[0]
    k_dim = ${aName}.shape[1]
    grid = (ct.cdiv(nnz, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${rowIdxName}, ${colIdxName}, ${aName}, ${bName}, ${cName}, nnz, k_dim, TILE_SIZE))`;
}

/**
 * Get the appropriate sparse template generator based on variant
 */
export function getSparseGenerator(variant?: string): (ir: SparseIR) => string {
  switch (variant) {
    case 'spmv_csr':
      return generateSpMVCSR;
    case 'spmv_csr_warp':
      return generateSpMVCSRWarp;
    case 'spmv_coo':
      return generateSpMVCOO;
    case 'spmv_ell':
      return generateSpMVELL;
    case 'spmm_csr':
      return generateSpMMCSR;
    case 'sddmm':
      return generateSDDMM;
    default:
      return generateSpMVCSR;
  }
}
