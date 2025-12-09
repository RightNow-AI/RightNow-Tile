/**
 * Reduction Pattern Templates
 * Specialized code generation for reduction variants
 */

import { EnhancedKernelIR } from '../../ir/types';

export function generateTreeReduction(ir: EnhancedKernelIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const inputName = ir.semanticInfo.inputArrays[0] || 'input_arr';
  const outputName = ir.semanticInfo.outputArrays[0] || 'output_arr';
  const reductionOp = ir.semanticInfo.reductionOp || 'sum';

  const opFunc = reductionOp === 'max' ? 'maximum' : reductionOp === 'min' ? 'minimum' : 'add';
  const atomicOp = reductionOp === 'max' ? 'max' : reductionOp === 'min' ? 'min' : 'add';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(${inputName}, ${outputName}, n: ct.Constant[int], tile_size: ct.Constant[int]):
    """
    Tree Reduction kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: tree_reduction
    """
    pid = ct.bid(0)
    tid = ct.tid(0)

    # Calculate global index
    idx = pid * tile_size * 2 + tid

    # Load two elements per thread for first level reduction
    tile = ct.zeros((tile_size,), dtype=ct.float32)

    if idx < n:
        tile[tid] = ${inputName}[idx]
    if idx + tile_size < n:
        tile[tid] += ${inputName}[idx + tile_size]

    ct.sync_threads()

    # Tree reduction in shared memory
    stride = tile_size // 2
    while stride > 0:
        if tid < stride:
            tile[tid] = ct.${opFunc}(tile[tid], tile[tid + stride])
        ct.sync_threads()
        stride //= 2

    # Write block result
    if tid == 0:
        ct.atomic_${atomicOp}(${outputName}, 0, tile[0])


def launch_${ir.name}(${inputName}, ${outputName}):
    """Launch the ${ir.name} tree reduction kernel"""
    n = ${inputName}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE * 2), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, n, TILE_SIZE))`;
}

export function generateWarpShuffleReduction(ir: EnhancedKernelIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const elementsPerThread = ir.tileConfig.elementsPerThread || 4;
  const inputName = ir.semanticInfo.inputArrays[0] || 'input_arr';
  const outputName = ir.semanticInfo.outputArrays[0] || 'output_arr';
  const reductionOp = ir.semanticInfo.reductionOp || 'sum';

  const opFunc = reductionOp === 'max' ? 'maximum' : reductionOp === 'min' ? 'minimum' : 'add';
  const atomicOp = reductionOp === 'max' ? 'max' : reductionOp === 'min' ? 'min' : 'add';
  const numWarps = tileSize / 32;

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
ELEMS_PER_THREAD = ${elementsPerThread}
WARP_SIZE = 32

@ct.kernel
def ${ir.name}(${inputName}, ${outputName}, n: ct.Constant[int], tile_size: ct.Constant[int]):
    """
    Warp Shuffle Reduction kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: warp_shuffle
    """
    pid = ct.bid(0)
    tid = ct.tid(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE

    # Each thread processes multiple elements
    val = ct.float32(0.0)
    base_idx = pid * tile_size * ELEMS_PER_THREAD + tid

    for i in range(ELEMS_PER_THREAD):
        idx = base_idx + i * tile_size
        if idx < n:
            val = ct.${opFunc}(val, ${inputName}[idx])

    # Warp-level reduction using shuffle
    for offset in [16, 8, 4, 2, 1]:
        val = ct.${opFunc}(val, ct.shfl_down(val, offset))

    # First thread in each warp writes to shared memory
    warp_results = ct.shared_zeros((${numWarps},), dtype=ct.float32)
    if lane == 0:
        warp_results[warp_id] = val

    ct.sync_threads()

    # Final reduction across warps
    if warp_id == 0:
        val = warp_results[lane] if lane < ${numWarps} else ct.float32(0.0)
        for offset in [16, 8, 4, 2, 1]:
            val = ct.${opFunc}(val, ct.shfl_down(val, offset))

        if lane == 0:
            ct.atomic_${atomicOp}(${outputName}, 0, val)


def launch_${ir.name}(${inputName}, ${outputName}):
    """Launch the ${ir.name} warp shuffle reduction kernel"""
    n = ${inputName}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE * ELEMS_PER_THREAD), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, n, TILE_SIZE))`;
}

export function generateMultiBlockReduction(ir: EnhancedKernelIR): string {
  const tileSize = ir.tileConfig.tileSize || 512;
  const inputName = ir.semanticInfo.inputArrays[0] || 'input_arr';
  const outputName = ir.semanticInfo.outputArrays[0] || 'output_arr';
  const reductionOp = ir.semanticInfo.reductionOp || 'sum';

  const atomicOp = reductionOp === 'max' ? 'max' : reductionOp === 'min' ? 'min' : 'add';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(${inputName}, ${outputName}, n: ct.Constant[int], tile_size: ct.Constant[int]):
    """
    Multi-Block Reduction kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: multi_block
    """
    pid = ct.bid(0)

    # Load tile
    tile = ct.load(${inputName}, index=(pid,), shape=(tile_size,))

    # Reduce entire tile
    block_result = ct.reduce(tile, op=ct.${reductionOp})

    # Atomic accumulation to global output
    ct.atomic_${atomicOp}(${outputName}, 0, block_result)


def launch_${ir.name}(${inputName}, ${outputName}):
    """Launch the ${ir.name} multi-block reduction kernel"""
    n = ${inputName}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, n, TILE_SIZE))`;
}

export function generateSegmentedReduction(ir: EnhancedKernelIR): string {
  const tileSize = ir.tileConfig.tileSize || 128;
  const inputName = ir.semanticInfo.inputArrays[0] || 'input_arr';
  const outputName = ir.semanticInfo.outputArrays[0] || 'output_arr';
  const reductionOp = ir.semanticInfo.reductionOp || 'sum';

  const opFunc = reductionOp === 'max' ? 'maximum' : reductionOp === 'min' ? 'minimum' : 'add';
  const initVal = reductionOp === 'max' ? '-inf' : reductionOp === 'min' ? 'inf' : '0.0';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName},
    segment_offsets,
    num_segments: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Segmented Reduction kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: segmented
    """
    pid = ct.bid(0)
    tid = ct.tid(0)

    if pid >= num_segments:
        return

    # Get segment bounds
    seg_start = segment_offsets[pid]
    seg_end = segment_offsets[pid + 1]
    seg_len = seg_end - seg_start

    # Initialize accumulator
    acc = ct.float32(${initVal})

    # Each thread processes strided elements within segment
    for i in range(tid, seg_len, tile_size):
        if seg_start + i < seg_end:
            val = ${inputName}[seg_start + i]
            acc = ct.${opFunc}(acc, val)

    # Warp-level reduction
    for offset in [16, 8, 4, 2, 1]:
        acc = ct.${opFunc}(acc, ct.shfl_down(acc, offset))

    # Write segment result
    if tid == 0:
        ${outputName}[pid] = acc


def launch_${ir.name}(${inputName}, ${outputName}, segment_offsets):
    """Launch the ${ir.name} segmented reduction kernel"""
    num_segments = segment_offsets.shape[0] - 1
    grid = (num_segments, 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, segment_offsets, num_segments, TILE_SIZE))`;
}
