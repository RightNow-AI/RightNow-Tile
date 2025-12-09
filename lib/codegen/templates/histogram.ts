/**
 * Histogram Pattern Templates
 * Specialized code generation for histogram computation variants
 */

import { EnhancedKernelIR } from '../../ir/types';

/**
 * Histogram IR extension
 */
export interface HistogramIR extends EnhancedKernelIR {
  histogramConfig?: {
    numBins: number;
    binWidth?: number;
    minVal?: number;
    maxVal?: number;
    usePrivatization: boolean;
    binsPerThread?: number;
  };
}

/**
 * Generate atomic histogram kernel
 * Simple but may have contention for skewed distributions
 */
export function generateHistogramAtomic(ir: HistogramIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const numBins = ir.histogramConfig?.numBins || 256;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const histName = ir.stores[0]?.target || 'histogram';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
NUM_BINS = ${numBins}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${histName},
    n: ct.Constant[int],
    num_bins: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Atomic Histogram kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: histogram_atomic

    Each thread atomically increments the appropriate bin.
    Simple but may have contention for skewed data.
    """
    idx = ct.bid(0) * tile_size + ct.tid(0)

    if idx >= n:
        return

    # Load data value
    val = ${inputName}[idx]

    # Compute bin index (assuming data is already in [0, num_bins) range)
    # For floating point data: bin = int(val * num_bins) clamped to [0, num_bins-1]
    bin_idx = ct.int32(val)
    bin_idx = ct.maximum(0, ct.minimum(bin_idx, num_bins - 1))

    # Atomic increment
    ct.atomic_add(${histName}, bin_idx, 1)


def launch_${ir.name}(${inputName}, ${histName}):
    """Launch the ${ir.name} atomic histogram kernel"""
    n = ${inputName}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${histName}, n, NUM_BINS, TILE_SIZE))`;
}

/**
 * Generate privatized histogram kernel
 * Uses per-block shared memory histogram, then reduces to global
 */
export function generateHistogramPrivatized(ir: HistogramIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const numBins = ir.histogramConfig?.numBins || 256;
  const elemsPerThread = 4;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const histName = ir.stores[0]?.target || 'histogram';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
NUM_BINS = ${numBins}
ELEMS_PER_THREAD = ${elemsPerThread}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${histName},
    n: ct.Constant[int],
    num_bins: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Privatized Histogram kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: histogram_privatized

    Each block builds a local histogram in shared memory,
    then atomically adds to global histogram.
    Reduces contention significantly.
    """
    pid = ct.bid(0)
    tid = ct.tid(0)

    # Shared memory histogram (per-block)
    shared_hist = ct.shared_zeros((num_bins,), dtype=ct.int32)

    # Initialize shared histogram
    for i in range(tid, num_bins, tile_size):
        shared_hist[i] = 0
    ct.sync_threads()

    # Each thread processes multiple elements
    base_idx = pid * tile_size * ELEMS_PER_THREAD + tid

    for e in range(ELEMS_PER_THREAD):
        idx = base_idx + e * tile_size
        if idx < n:
            val = ${inputName}[idx]
            bin_idx = ct.int32(val)
            bin_idx = ct.maximum(0, ct.minimum(bin_idx, num_bins - 1))
            ct.atomic_add(shared_hist, bin_idx, 1)

    ct.sync_threads()

    # Merge shared histogram to global
    for i in range(tid, num_bins, tile_size):
        if shared_hist[i] > 0:
            ct.atomic_add(${histName}, i, shared_hist[i])


def launch_${ir.name}(${inputName}, ${histName}):
    """Launch the ${ir.name} privatized histogram kernel"""
    n = ${inputName}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE * ELEMS_PER_THREAD), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${histName}, n, NUM_BINS, TILE_SIZE))`;
}

/**
 * Generate multi-pass histogram for large bin counts
 * Processes subsets of bins to fit in shared memory
 */
export function generateHistogramMultiPass(ir: HistogramIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const numBins = ir.histogramConfig?.numBins || 4096;
  const binsPerPass = 256;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const histName = ir.stores[0]?.target || 'histogram';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
NUM_BINS = ${numBins}
BINS_PER_PASS = ${binsPerPass}
NUM_PASSES = (NUM_BINS + BINS_PER_PASS - 1) // BINS_PER_PASS

@ct.kernel
def ${ir.name}(
    ${inputName}, ${histName},
    n: ct.Constant[int],
    num_bins: ct.Constant[int],
    pass_idx: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Multi-Pass Histogram kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: histogram_multipass

    For large bin counts, processes bins in multiple passes.
    Each pass handles BINS_PER_PASS bins using shared memory.
    """
    pid = ct.bid(0)
    tid = ct.tid(0)

    bin_start = pass_idx * BINS_PER_PASS
    bin_end = ct.minimum(bin_start + BINS_PER_PASS, num_bins)

    # Shared histogram for this pass's bins
    shared_hist = ct.shared_zeros((BINS_PER_PASS,), dtype=ct.int32)

    # Initialize
    for i in range(tid, BINS_PER_PASS, tile_size):
        shared_hist[i] = 0
    ct.sync_threads()

    # Process elements
    for idx in range(pid * tile_size + tid, n, ct.gridDim(0) * tile_size):
        val = ${inputName}[idx]
        bin_idx = ct.int32(val)

        # Only count if in this pass's bin range
        if bin_idx >= bin_start and bin_idx < bin_end:
            local_bin = bin_idx - bin_start
            ct.atomic_add(shared_hist, local_bin, 1)

    ct.sync_threads()

    # Write to global histogram
    for i in range(tid, bin_end - bin_start, tile_size):
        if shared_hist[i] > 0:
            ct.atomic_add(${histName}, bin_start + i, shared_hist[i])


def launch_${ir.name}(${inputName}, ${histName}):
    """Launch the ${ir.name} multi-pass histogram kernel"""
    n = ${inputName}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()

    # Launch multiple passes
    for pass_idx in range(NUM_PASSES):
        ct.launch(stream, grid, ${ir.name}, (${inputName}, ${histName}, n, NUM_BINS, pass_idx, TILE_SIZE))`;
}

/**
 * Generate weighted histogram kernel
 * Accumulates weights instead of counts
 */
export function generateHistogramWeighted(ir: HistogramIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const numBins = ir.histogramConfig?.numBins || 256;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const weightsName = ir.loads[1]?.source || 'weights';
  const histName = ir.stores[0]?.target || 'histogram';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
NUM_BINS = ${numBins}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${weightsName}, ${histName},
    n: ct.Constant[int],
    num_bins: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Weighted Histogram kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: histogram_weighted

    Accumulates weights instead of counts.
    Useful for probability distributions.
    """
    pid = ct.bid(0)
    tid = ct.tid(0)

    # Shared histogram (floating point for weights)
    shared_hist = ct.shared_zeros((num_bins,), dtype=ct.float32)

    # Initialize
    for i in range(tid, num_bins, tile_size):
        shared_hist[i] = 0.0
    ct.sync_threads()

    # Process elements
    idx = pid * tile_size + tid
    if idx < n:
        val = ${inputName}[idx]
        weight = ${weightsName}[idx]
        bin_idx = ct.int32(val)
        bin_idx = ct.maximum(0, ct.minimum(bin_idx, num_bins - 1))
        ct.atomic_add(shared_hist, bin_idx, weight)

    ct.sync_threads()

    # Merge to global
    for i in range(tid, num_bins, tile_size):
        if shared_hist[i] > 0.0:
            ct.atomic_add(${histName}, i, shared_hist[i])


def launch_${ir.name}(${inputName}, ${weightsName}, ${histName}):
    """Launch the ${ir.name} weighted histogram kernel"""
    n = ${inputName}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${weightsName}, ${histName}, n, NUM_BINS, TILE_SIZE))`;
}

/**
 * Generate 2D histogram kernel
 * Joint histogram for two variables
 */
export function generateHistogram2D(ir: HistogramIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const numBinsX = ir.histogramConfig?.numBins || 64;
  const numBinsY = 64;

  const inputXName = ir.loads[0]?.source || 'input_x';
  const inputYName = ir.loads[1]?.source || 'input_y';
  const histName = ir.stores[0]?.target || 'histogram_2d';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
NUM_BINS_X = ${numBinsX}
NUM_BINS_Y = ${numBinsY}

@ct.kernel
def ${ir.name}(
    ${inputXName}, ${inputYName}, ${histName},
    n: ct.Constant[int],
    num_bins_x: ct.Constant[int],
    num_bins_y: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    2D Histogram kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: histogram_2d

    Joint histogram for two variables.
    Output shape: [num_bins_x, num_bins_y]
    """
    idx = ct.bid(0) * tile_size + ct.tid(0)

    if idx >= n:
        return

    val_x = ${inputXName}[idx]
    val_y = ${inputYName}[idx]

    bin_x = ct.int32(val_x)
    bin_y = ct.int32(val_y)

    bin_x = ct.maximum(0, ct.minimum(bin_x, num_bins_x - 1))
    bin_y = ct.maximum(0, ct.minimum(bin_y, num_bins_y - 1))

    # 2D index in row-major format
    bin_idx = bin_x * num_bins_y + bin_y

    ct.atomic_add(${histName}, bin_idx, 1)


def launch_${ir.name}(${inputXName}, ${inputYName}, ${histName}):
    """Launch the ${ir.name} 2D histogram kernel"""
    n = ${inputXName}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputXName}, ${inputYName}, ${histName}, n, NUM_BINS_X, NUM_BINS_Y, TILE_SIZE))`;
}

/**
 * Get the appropriate histogram template generator based on variant
 */
export function getHistogramGenerator(variant?: string): (ir: HistogramIR) => string {
  switch (variant) {
    case 'histogram_atomic':
      return generateHistogramAtomic;
    case 'histogram_privatized':
      return generateHistogramPrivatized;
    case 'histogram_multipass':
      return generateHistogramMultiPass;
    case 'histogram_weighted':
      return generateHistogramWeighted;
    case 'histogram_2d':
      return generateHistogram2D;
    default:
      return generateHistogramPrivatized; // Default to privatized for better performance
  }
}
