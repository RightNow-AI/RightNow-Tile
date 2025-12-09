/**
 * Sorting Pattern Templates
 * Specialized code generation for bitonic sort, radix sort, and merge sort
 */

import { EnhancedKernelIR } from '../../ir/types';

/**
 * Sorting IR extension
 */
export interface SortingIR extends EnhancedKernelIR {
  sortConfig?: {
    sortType: 'bitonic' | 'radix' | 'merge';
    direction: 'ascending' | 'descending';
    keyType: 'int' | 'float' | 'custom';
    hasValues: boolean;  // key-value sort
    maxElements?: number;
  };
}

/**
 * Generate bitonic sort kernel
 * Parallel sorting network with O(log^2 n) depth
 */
export function generateBitonicSort(ir: SortingIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const direction = ir.sortConfig?.direction || 'ascending';

  const inputName = ir.loads[0]?.source || 'data';
  const outputName = ir.stores[0]?.target || 'data';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(
    ${inputName},
    n: ct.Constant[int],
    stage: ct.Constant[int],
    step: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Bitonic Sort kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: bitonic_sort

    One step of bitonic sort network.
    Direction: ${direction}
    """
    idx = ct.bid(0) * tile_size + ct.tid(0)

    if idx >= n:
        return

    # Partner index using XOR
    partner = idx ^ step

    if partner > idx and partner < n:
        # Determine sort direction for this pair
        # Direction alternates based on position within bitonic sequence
        dir_mask = idx & stage
        ascending = (dir_mask == 0) ${direction === 'ascending' ? '' : '^ True'}

        val_idx = ${inputName}[idx]
        val_partner = ${inputName}[partner]

        # Compare and swap
        if ascending:
            if val_idx > val_partner:
                ${inputName}[idx] = val_partner
                ${inputName}[partner] = val_idx
        else:
            if val_idx < val_partner:
                ${inputName}[idx] = val_partner
                ${inputName}[partner] = val_idx


def launch_${ir.name}(${inputName}):
    """Launch full bitonic sort"""
    import math
    n = ${inputName}.shape[0]
    # Pad to power of 2 if needed
    n_padded = 1 << int(math.ceil(math.log2(n)))

    stream = cupy.cuda.get_current_stream()

    # Bitonic sort: O(log^2 n) stages
    stage = 2
    while stage <= n_padded:
        step = stage // 2
        while step > 0:
            grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
            ct.launch(stream, grid, ${ir.name}, (${inputName}, n, stage, step, TILE_SIZE))
            step //= 2
        stage *= 2`;
}

/**
 * Generate shared memory bitonic sort (for small arrays)
 */
export function generateBitonicSortShared(ir: SortingIR): string {
  const tileSize = ir.tileConfig.tileSize || 512;

  const inputName = ir.loads[0]?.source || 'data';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(
    ${inputName},
    n: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Bitonic Sort (Shared Memory) - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: bitonic_sort_shared

    Sorts small arrays entirely in shared memory.
    """
    tid = ct.tid(0)
    bid = ct.bid(0)
    base_idx = bid * tile_size

    # Load into shared memory
    shared = ct.shared_zeros((tile_size,), dtype=ct.float32)
    if base_idx + tid < n:
        shared[tid] = ${inputName}[base_idx + tid]
    else:
        shared[tid] = ct.float32('inf')  # Pad with infinity

    ct.sync_threads()

    # Bitonic sort in shared memory
    stage = 2
    while stage <= tile_size:
        step = stage // 2
        while step > 0:
            partner = tid ^ step
            if partner > tid:
                dir_mask = tid & stage
                ascending = (dir_mask == 0)

                if ascending:
                    if shared[tid] > shared[partner]:
                        temp = shared[tid]
                        shared[tid] = shared[partner]
                        shared[partner] = temp
                else:
                    if shared[tid] < shared[partner]:
                        temp = shared[tid]
                        shared[tid] = shared[partner]
                        shared[partner] = temp

            ct.sync_threads()
            step //= 2
        stage *= 2

    # Write back
    if base_idx + tid < n:
        ${inputName}[base_idx + tid] = shared[tid]


def launch_${ir.name}(${inputName}):
    """Launch shared memory bitonic sort"""
    n = ${inputName}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, n, TILE_SIZE))`;
}

/**
 * Generate radix sort kernel (single pass for specific bits)
 */
export function generateRadixSort(ir: SortingIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const radixBits = 4;  // Process 4 bits at a time (16 buckets)

  const keysName = ir.loads[0]?.source || 'keys';
  const keysOutName = ir.stores[0]?.target || 'keys_out';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
RADIX_BITS = ${radixBits}
NUM_BUCKETS = 1 << RADIX_BITS  # 16

@ct.kernel
def ${ir.name}_histogram(
    ${keysName}, histograms,
    n: ct.Constant[int],
    shift: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Radix Sort - Histogram phase
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%

    Counts elements per bucket for this digit.
    """
    tid = ct.tid(0)
    bid = ct.bid(0)

    # Shared memory histogram
    shared_hist = ct.shared_zeros((NUM_BUCKETS,), dtype=ct.int32)
    if tid < NUM_BUCKETS:
        shared_hist[tid] = 0
    ct.sync_threads()

    # Count elements in each bucket
    idx = bid * tile_size + tid
    if idx < n:
        key = ${keysName}[idx]
        bucket = (key >> shift) & (NUM_BUCKETS - 1)
        ct.atomic_add(shared_hist, bucket, 1)

    ct.sync_threads()

    # Write to global histogram
    if tid < NUM_BUCKETS:
        histograms[bid * NUM_BUCKETS + tid] = shared_hist[tid]


@ct.kernel
def ${ir.name}_scatter(
    ${keysName}, ${keysOutName}, offsets,
    n: ct.Constant[int],
    shift: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Radix Sort - Scatter phase
    Scatters elements to sorted positions.
    """
    tid = ct.tid(0)
    bid = ct.bid(0)

    # Shared prefix sums for this block
    shared_offsets = ct.shared_zeros((NUM_BUCKETS,), dtype=ct.int32)
    if tid < NUM_BUCKETS:
        shared_offsets[tid] = offsets[bid * NUM_BUCKETS + tid]
    ct.sync_threads()

    idx = bid * tile_size + tid
    if idx < n:
        key = ${keysName}[idx]
        bucket = (key >> shift) & (NUM_BUCKETS - 1)

        # Get output position and increment
        out_pos = ct.atomic_add(shared_offsets, bucket, 1)
        ${keysOutName}[out_pos] = key


def launch_${ir.name}(${keysName}):
    """Launch full radix sort (32-bit keys)"""
    n = ${keysName}.shape[0]
    num_blocks = ct.cdiv(n, TILE_SIZE)

    # Allocate temporary buffers
    ${keysOutName} = cupy.empty_like(${keysName})
    histograms = cupy.zeros((num_blocks, NUM_BUCKETS), dtype=cupy.int32)
    offsets = cupy.zeros((num_blocks, NUM_BUCKETS), dtype=cupy.int32)

    stream = cupy.cuda.get_current_stream()
    grid = (num_blocks, 1, 1)

    # Process 4 bits at a time, 8 passes for 32-bit keys
    for pass_idx in range(8):
        shift = pass_idx * RADIX_BITS

        # Histogram
        ct.launch(stream, grid, ${ir.name}_histogram, (${keysName}, histograms, n, shift, TILE_SIZE))

        # Prefix sum on histograms (exclusive scan)
        # This computes output positions for each bucket
        offsets = cupy.cumsum(histograms.flatten()).reshape(num_blocks, NUM_BUCKETS)
        offsets = cupy.roll(offsets, 1)
        offsets.flat[0] = 0

        # Scatter
        ct.launch(stream, grid, ${ir.name}_scatter, (${keysName}, ${keysOutName}, offsets, n, shift, TILE_SIZE))

        # Swap buffers
        ${keysName}, ${keysOutName} = ${keysOutName}, ${keysName}

    return ${keysName}`;
}

/**
 * Generate merge sort kernel
 */
export function generateMergeSort(ir: SortingIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;

  const inputName = ir.loads[0]?.source || 'data';
  const outputName = ir.stores[0]?.target || 'data_out';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName},
    n: ct.Constant[int],
    width: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Merge Sort kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: merge_sort

    Merges sorted sub-arrays of size 'width'.
    """
    tid = ct.tid(0)
    bid = ct.bid(0)

    # Each block merges one pair of sorted subarrays
    merge_idx = bid
    left_start = merge_idx * 2 * width
    mid = ct.minimum(left_start + width, n)
    right_end = ct.minimum(left_start + 2 * width, n)

    if left_start >= n:
        return

    # Load left and right subarrays into shared memory
    shared_left = ct.shared_zeros((width,), dtype=ct.float32)
    shared_right = ct.shared_zeros((width,), dtype=ct.float32)

    # Load left subarray
    for i in range(tid, mid - left_start, tile_size):
        shared_left[i] = ${inputName}[left_start + i]

    # Load right subarray
    for i in range(tid, right_end - mid, tile_size):
        shared_right[i] = ${inputName}[mid + i]

    ct.sync_threads()

    # Sequential merge (for simplicity; can be parallelized)
    if tid == 0:
        i = 0
        j = 0
        k = left_start
        left_len = mid - left_start
        right_len = right_end - mid

        while i < left_len and j < right_len:
            if shared_left[i] <= shared_right[j]:
                ${outputName}[k] = shared_left[i]
                i += 1
            else:
                ${outputName}[k] = shared_right[j]
                j += 1
            k += 1

        # Copy remaining elements
        while i < left_len:
            ${outputName}[k] = shared_left[i]
            i += 1
            k += 1

        while j < right_len:
            ${outputName}[k] = shared_right[j]
            j += 1
            k += 1


def launch_${ir.name}(${inputName}):
    """Launch full merge sort"""
    n = ${inputName}.shape[0]
    ${outputName} = cupy.empty_like(${inputName})

    stream = cupy.cuda.get_current_stream()

    # Bottom-up merge sort
    width = 1
    while width < n:
        num_merges = ct.cdiv(n, 2 * width)
        grid = (num_merges, 1, 1)
        ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, n, width, TILE_SIZE))

        # Swap buffers
        ${inputName}, ${outputName} = ${outputName}, ${inputName}
        width *= 2

    return ${inputName}`;
}

/**
 * Get the appropriate sorting template generator based on variant
 */
export function getSortingGenerator(variant?: string): (ir: SortingIR) => string {
  switch (variant) {
    case 'bitonic_sort':
      return generateBitonicSort;
    case 'bitonic_sort_shared':
      return generateBitonicSortShared;
    case 'radix_sort':
      return generateRadixSort;
    case 'merge_sort':
      return generateMergeSort;
    default:
      return generateBitonicSort;
  }
}
