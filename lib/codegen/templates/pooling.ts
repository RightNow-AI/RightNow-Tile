/**
 * Pooling Pattern Templates
 * Specialized code generation for max pooling, average pooling, and global pooling
 */

import { EnhancedKernelIR } from '../../ir/types';

/**
 * Pooling IR extension
 */
export interface PoolingIR extends EnhancedKernelIR {
  poolConfig?: {
    poolType: 'max' | 'avg' | 'global_max' | 'global_avg';
    kernelSize: number[];
    stride: number[];
    padding: number[];
    ceilMode: boolean;
    countIncludePad: boolean;
  };
}

/**
 * Generate 2D max pooling kernel
 */
export function generateMaxPool2D(ir: PoolingIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const poolH = ir.poolConfig?.kernelSize[0] || 2;
  const poolW = ir.poolConfig?.kernelSize[1] || 2;
  const strideH = ir.poolConfig?.stride[0] || 2;
  const strideW = ir.poolConfig?.stride[1] || 2;
  const padH = ir.poolConfig?.padding[0] || 0;
  const padW = ir.poolConfig?.padding[1] || 0;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
POOL_H = ${poolH}
POOL_W = ${poolW}
STRIDE_H = ${strideH}
STRIDE_W = ${strideW}
PAD_H = ${padH}
PAD_W = ${padW}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName},
    batch_size: ct.Constant[int],
    channels: ct.Constant[int],
    in_h: ct.Constant[int],
    in_w: ct.Constant[int],
    out_h: ct.Constant[int],
    out_w: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    2D Max Pooling kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: max_pool_2d

    Input: [batch, channels, H, W]
    Output: [batch, channels, outH, outW]
    """
    total_out = batch_size * channels * out_h * out_w
    flat_idx = ct.bid(0) * tile_size + ct.tid(0)

    if flat_idx >= total_out:
        return

    # Decompose flat index
    b = flat_idx // (channels * out_h * out_w)
    rem = flat_idx % (channels * out_h * out_w)
    c = rem // (out_h * out_w)
    rem2 = rem % (out_h * out_w)
    oh = rem2 // out_w
    ow = rem2 % out_w

    # Compute max over pooling window
    max_val = ct.float32('-inf')
    ih_start = oh * STRIDE_H - PAD_H
    iw_start = ow * STRIDE_W - PAD_W

    for ph in range(POOL_H):
        for pw in range(POOL_W):
            ih = ih_start + ph
            iw = iw_start + pw

            if ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
                in_idx = b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw
                val = ${inputName}[in_idx]
                max_val = ct.maximum(max_val, val)

    ${outputName}[flat_idx] = max_val


def launch_${ir.name}(${inputName}, ${outputName}):
    """Launch max pool 2D kernel"""
    batch_size, channels, in_h, in_w = ${inputName}.shape
    out_h = (in_h + 2 * PAD_H - POOL_H) // STRIDE_H + 1
    out_w = (in_w + 2 * PAD_W - POOL_W) // STRIDE_W + 1
    total_out = batch_size * channels * out_h * out_w
    grid = (ct.cdiv(total_out, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, batch_size, channels, in_h, in_w, out_h, out_w, TILE_SIZE))`;
}

/**
 * Generate 2D average pooling kernel
 */
export function generateAvgPool2D(ir: PoolingIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const poolH = ir.poolConfig?.kernelSize[0] || 2;
  const poolW = ir.poolConfig?.kernelSize[1] || 2;
  const strideH = ir.poolConfig?.stride[0] || 2;
  const strideW = ir.poolConfig?.stride[1] || 2;
  const padH = ir.poolConfig?.padding[0] || 0;
  const padW = ir.poolConfig?.padding[1] || 0;
  const countIncludePad = ir.poolConfig?.countIncludePad ?? false;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
POOL_H = ${poolH}
POOL_W = ${poolW}
STRIDE_H = ${strideH}
STRIDE_W = ${strideW}
PAD_H = ${padH}
PAD_W = ${padW}
COUNT_INCLUDE_PAD = ${countIncludePad}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName},
    batch_size: ct.Constant[int],
    channels: ct.Constant[int],
    in_h: ct.Constant[int],
    in_w: ct.Constant[int],
    out_h: ct.Constant[int],
    out_w: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    2D Average Pooling kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: avg_pool_2d

    Input: [batch, channels, H, W]
    Output: [batch, channels, outH, outW]
    """
    total_out = batch_size * channels * out_h * out_w
    flat_idx = ct.bid(0) * tile_size + ct.tid(0)

    if flat_idx >= total_out:
        return

    # Decompose flat index
    b = flat_idx // (channels * out_h * out_w)
    rem = flat_idx % (channels * out_h * out_w)
    c = rem // (out_h * out_w)
    rem2 = rem % (out_h * out_w)
    oh = rem2 // out_w
    ow = rem2 % out_w

    # Compute average over pooling window
    sum_val = ct.float32(0.0)
    count = ct.int32(0)
    ih_start = oh * STRIDE_H - PAD_H
    iw_start = ow * STRIDE_W - PAD_W

    for ph in range(POOL_H):
        for pw in range(POOL_W):
            ih = ih_start + ph
            iw = iw_start + pw

            if ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
                in_idx = b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw
                sum_val = sum_val + ${inputName}[in_idx]
                count = count + 1
            elif COUNT_INCLUDE_PAD:
                count = count + 1

    # Avoid division by zero
    if count > 0:
        ${outputName}[flat_idx] = sum_val / ct.float32(count)
    else:
        ${outputName}[flat_idx] = 0.0


def launch_${ir.name}(${inputName}, ${outputName}):
    """Launch avg pool 2D kernel"""
    batch_size, channels, in_h, in_w = ${inputName}.shape
    out_h = (in_h + 2 * PAD_H - POOL_H) // STRIDE_H + 1
    out_w = (in_w + 2 * PAD_W - POOL_W) // STRIDE_W + 1
    total_out = batch_size * channels * out_h * out_w
    grid = (ct.cdiv(total_out, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, batch_size, channels, in_h, in_w, out_h, out_w, TILE_SIZE))`;
}

/**
 * Generate global average pooling kernel
 */
export function generateGlobalAvgPool(ir: PoolingIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
WARP_SIZE = 32

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName},
    batch_size: ct.Constant[int],
    channels: ct.Constant[int],
    spatial_size: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Global Average Pooling kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: global_avg_pool

    Reduces entire spatial dimension to single value.
    Input: [batch, channels, H, W]
    Output: [batch, channels, 1, 1] or [batch, channels]
    """
    tid = ct.tid(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE

    b = ct.bid(0)
    c = ct.bid(1)

    if b >= batch_size or c >= channels:
        return

    # Each thread accumulates strided elements
    sum_val = ct.float32(0.0)
    base_idx = b * channels * spatial_size + c * spatial_size

    for i in range(tid, spatial_size, tile_size):
        sum_val = sum_val + ${inputName}[base_idx + i]

    # Warp reduction
    for offset in [16, 8, 4, 2, 1]:
        sum_val = sum_val + ct.shfl_down(sum_val, offset)

    # Cross-warp reduction via shared memory
    num_warps = tile_size // WARP_SIZE
    warp_sums = ct.shared_zeros((num_warps,), dtype=ct.float32)

    if lane == 0:
        warp_sums[warp_id] = sum_val

    ct.sync_threads()

    # Final reduction by warp 0
    if warp_id == 0:
        sum_val = warp_sums[lane] if lane < num_warps else ct.float32(0.0)
        for offset in [16, 8, 4, 2, 1]:
            sum_val = sum_val + ct.shfl_down(sum_val, offset)

        if lane == 0:
            ${outputName}[b * channels + c] = sum_val / ct.float32(spatial_size)


def launch_${ir.name}(${inputName}, ${outputName}):
    """Launch global avg pool kernel"""
    batch_size, channels, h, w = ${inputName}.shape
    spatial_size = h * w
    grid = (batch_size, channels, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, batch_size, channels, spatial_size, TILE_SIZE))`;
}

/**
 * Generate global max pooling kernel
 */
export function generateGlobalMaxPool(ir: PoolingIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
WARP_SIZE = 32

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName},
    batch_size: ct.Constant[int],
    channels: ct.Constant[int],
    spatial_size: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Global Max Pooling kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: global_max_pool

    Reduces entire spatial dimension to max value.
    Input: [batch, channels, H, W]
    Output: [batch, channels]
    """
    tid = ct.tid(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE

    b = ct.bid(0)
    c = ct.bid(1)

    if b >= batch_size or c >= channels:
        return

    # Each thread finds max of strided elements
    max_val = ct.float32('-inf')
    base_idx = b * channels * spatial_size + c * spatial_size

    for i in range(tid, spatial_size, tile_size):
        max_val = ct.maximum(max_val, ${inputName}[base_idx + i])

    # Warp reduction for max
    for offset in [16, 8, 4, 2, 1]:
        max_val = ct.maximum(max_val, ct.shfl_down(max_val, offset))

    # Cross-warp reduction
    num_warps = tile_size // WARP_SIZE
    warp_maxes = ct.shared_zeros((num_warps,), dtype=ct.float32)

    if lane == 0:
        warp_maxes[warp_id] = max_val

    ct.sync_threads()

    if warp_id == 0:
        max_val = warp_maxes[lane] if lane < num_warps else ct.float32('-inf')
        for offset in [16, 8, 4, 2, 1]:
            max_val = ct.maximum(max_val, ct.shfl_down(max_val, offset))

        if lane == 0:
            ${outputName}[b * channels + c] = max_val


def launch_${ir.name}(${inputName}, ${outputName}):
    """Launch global max pool kernel"""
    batch_size, channels, h, w = ${inputName}.shape
    spatial_size = h * w
    grid = (batch_size, channels, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, batch_size, channels, spatial_size, TILE_SIZE))`;
}

/**
 * Generate adaptive average pooling kernel
 */
export function generateAdaptiveAvgPool(ir: PoolingIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName},
    batch_size: ct.Constant[int],
    channels: ct.Constant[int],
    in_h: ct.Constant[int],
    in_w: ct.Constant[int],
    out_h: ct.Constant[int],
    out_w: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Adaptive Average Pooling kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: adaptive_avg_pool

    Automatically computes kernel size to match output size.
    Input: [batch, channels, H, W]
    Output: [batch, channels, out_h, out_w]
    """
    total_out = batch_size * channels * out_h * out_w
    flat_idx = ct.bid(0) * tile_size + ct.tid(0)

    if flat_idx >= total_out:
        return

    # Decompose flat index
    b = flat_idx // (channels * out_h * out_w)
    rem = flat_idx % (channels * out_h * out_w)
    c = rem // (out_h * out_w)
    rem2 = rem % (out_h * out_w)
    oh = rem2 // out_w
    ow = rem2 % out_w

    # Adaptive kernel: compute input region for this output pixel
    ih_start = oh * in_h // out_h
    ih_end = (oh + 1) * in_h // out_h
    iw_start = ow * in_w // out_w
    iw_end = (ow + 1) * in_w // out_w

    # Compute average over adaptive window
    sum_val = ct.float32(0.0)
    count = ct.int32(0)

    for ih in range(ih_start, ih_end):
        for iw in range(iw_start, iw_end):
            in_idx = b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw
            sum_val = sum_val + ${inputName}[in_idx]
            count = count + 1

    ${outputName}[flat_idx] = sum_val / ct.float32(count)


def launch_${ir.name}(${inputName}, ${outputName}, out_h, out_w):
    """Launch adaptive avg pool kernel"""
    batch_size, channels, in_h, in_w = ${inputName}.shape
    total_out = batch_size * channels * out_h * out_w
    grid = (ct.cdiv(total_out, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, batch_size, channels, in_h, in_w, out_h, out_w, TILE_SIZE))`;
}

/**
 * Get the appropriate pooling template generator based on variant
 */
export function getPoolingGenerator(variant?: string): (ir: PoolingIR) => string {
  switch (variant) {
    case 'max_pool_2d':
      return generateMaxPool2D;
    case 'avg_pool_2d':
      return generateAvgPool2D;
    case 'global_avg_pool':
      return generateGlobalAvgPool;
    case 'global_max_pool':
      return generateGlobalMaxPool;
    case 'adaptive_avg_pool':
    case 'adaptive_max_pool':
      return generateAdaptiveAvgPool;
    default:
      return generateMaxPool2D;
  }
}
