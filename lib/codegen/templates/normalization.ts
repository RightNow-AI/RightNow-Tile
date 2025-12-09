/**
 * Normalization Pattern Templates
 * Specialized code generation for LayerNorm, BatchNorm, GroupNorm, RMSNorm, etc.
 */

import { EnhancedKernelIR } from '../../ir/types';

/**
 * Normalization IR extension
 */
export interface NormalizationIR extends EnhancedKernelIR {
  normConfig?: {
    normType: 'layernorm' | 'batchnorm' | 'groupnorm' | 'instancenorm' | 'rmsnorm';
    epsilon: number;
    affine: boolean;
    normalizedShape?: number[];
    numGroups?: number;
  };
}

/**
 * Generate LayerNorm kernel
 * Normalizes over last dimension(s)
 */
export function generateLayerNorm(ir: NormalizationIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const epsilon = ir.normConfig?.epsilon || 1e-5;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const gammaName = ir.loads[1]?.source || 'gamma';
  const betaName = ir.loads[2]?.source || 'beta';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
EPSILON = ${epsilon}
WARP_SIZE = 32

@ct.kernel
def ${ir.name}(
    ${inputName}, ${gammaName}, ${betaName}, ${outputName},
    batch_size: ct.Constant[int],
    hidden_size: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Layer Normalization kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: layernorm

    Normalizes over hidden dimension:
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    tid = ct.tid(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE
    batch_idx = ct.bid(0)

    if batch_idx >= batch_size:
        return

    # First pass: compute mean
    sum_val = ct.float32(0.0)
    base_idx = batch_idx * hidden_size

    for i in range(tid, hidden_size, tile_size):
        sum_val = sum_val + ${inputName}[base_idx + i]

    # Warp reduction for sum
    for offset in [16, 8, 4, 2, 1]:
        sum_val = sum_val + ct.shfl_down(sum_val, offset)

    # Cross-warp reduction
    num_warps = tile_size // WARP_SIZE
    shared_sum = ct.shared_zeros((num_warps,), dtype=ct.float32)
    if lane == 0:
        shared_sum[warp_id] = sum_val
    ct.sync_threads()

    if warp_id == 0 and lane < num_warps:
        sum_val = shared_sum[lane]
        for offset in [16, 8, 4, 2, 1]:
            sum_val = sum_val + ct.shfl_down(sum_val, offset)
        if lane == 0:
            shared_sum[0] = sum_val / ct.float32(hidden_size)
    ct.sync_threads()
    mean = shared_sum[0]

    # Second pass: compute variance
    var_sum = ct.float32(0.0)
    for i in range(tid, hidden_size, tile_size):
        diff = ${inputName}[base_idx + i] - mean
        var_sum = var_sum + diff * diff

    for offset in [16, 8, 4, 2, 1]:
        var_sum = var_sum + ct.shfl_down(var_sum, offset)

    if lane == 0:
        shared_sum[warp_id] = var_sum
    ct.sync_threads()

    if warp_id == 0 and lane < num_warps:
        var_sum = shared_sum[lane]
        for offset in [16, 8, 4, 2, 1]:
            var_sum = var_sum + ct.shfl_down(var_sum, offset)
        if lane == 0:
            shared_sum[0] = ct.rsqrt(var_sum / ct.float32(hidden_size) + EPSILON)
    ct.sync_threads()
    rstd = shared_sum[0]

    # Third pass: normalize and apply affine transform
    for i in range(tid, hidden_size, tile_size):
        val = ${inputName}[base_idx + i]
        normalized = (val - mean) * rstd
        ${outputName}[base_idx + i] = ${gammaName}[i] * normalized + ${betaName}[i]


def launch_${ir.name}(${inputName}, ${gammaName}, ${betaName}, ${outputName}):
    """Launch LayerNorm kernel"""
    batch_size = ${inputName}.shape[0]
    hidden_size = ${inputName}.shape[-1]
    grid = (batch_size, 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${gammaName}, ${betaName}, ${outputName}, batch_size, hidden_size, TILE_SIZE))`;
}

/**
 * Generate RMSNorm kernel
 * Root Mean Square Layer Normalization (used in LLaMA)
 */
export function generateRMSNorm(ir: NormalizationIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const epsilon = ir.normConfig?.epsilon || 1e-6;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const weightName = ir.loads[1]?.source || 'weight';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
EPSILON = ${epsilon}
WARP_SIZE = 32

@ct.kernel
def ${ir.name}(
    ${inputName}, ${weightName}, ${outputName},
    batch_size: ct.Constant[int],
    hidden_size: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    RMS Normalization kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: rmsnorm

    y = x / sqrt(mean(x^2) + eps) * weight
    No mean subtraction, more efficient than LayerNorm.
    """
    tid = ct.tid(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE
    batch_idx = ct.bid(0)

    if batch_idx >= batch_size:
        return

    base_idx = batch_idx * hidden_size

    # Compute sum of squares
    sq_sum = ct.float32(0.0)
    for i in range(tid, hidden_size, tile_size):
        val = ${inputName}[base_idx + i]
        sq_sum = sq_sum + val * val

    # Warp reduction
    for offset in [16, 8, 4, 2, 1]:
        sq_sum = sq_sum + ct.shfl_down(sq_sum, offset)

    # Cross-warp reduction
    num_warps = tile_size // WARP_SIZE
    shared_sum = ct.shared_zeros((num_warps,), dtype=ct.float32)
    if lane == 0:
        shared_sum[warp_id] = sq_sum
    ct.sync_threads()

    if warp_id == 0 and lane < num_warps:
        sq_sum = shared_sum[lane]
        for offset in [16, 8, 4, 2, 1]:
            sq_sum = sq_sum + ct.shfl_down(sq_sum, offset)
        if lane == 0:
            shared_sum[0] = ct.rsqrt(sq_sum / ct.float32(hidden_size) + EPSILON)
    ct.sync_threads()

    rms_scale = shared_sum[0]

    # Apply normalization with weight
    for i in range(tid, hidden_size, tile_size):
        ${outputName}[base_idx + i] = ${inputName}[base_idx + i] * rms_scale * ${weightName}[i]


def launch_${ir.name}(${inputName}, ${weightName}, ${outputName}):
    """Launch RMSNorm kernel"""
    batch_size = ${inputName}.shape[0]
    hidden_size = ${inputName}.shape[-1]
    grid = (batch_size, 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${weightName}, ${outputName}, batch_size, hidden_size, TILE_SIZE))`;
}

/**
 * Generate GroupNorm kernel
 */
export function generateGroupNorm(ir: NormalizationIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const epsilon = ir.normConfig?.epsilon || 1e-5;
  const numGroups = ir.normConfig?.numGroups || 32;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const gammaName = ir.loads[1]?.source || 'gamma';
  const betaName = ir.loads[2]?.source || 'beta';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
EPSILON = ${epsilon}
NUM_GROUPS = ${numGroups}
WARP_SIZE = 32

@ct.kernel
def ${ir.name}(
    ${inputName}, ${gammaName}, ${betaName}, ${outputName},
    batch_size: ct.Constant[int],
    channels: ct.Constant[int],
    spatial_size: ct.Constant[int],
    num_groups: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Group Normalization kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: groupnorm

    Divides channels into groups, normalizes within each group.
    """
    tid = ct.tid(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE
    batch_idx = ct.bid(0)
    group_idx = ct.bid(1)

    if batch_idx >= batch_size or group_idx >= num_groups:
        return

    channels_per_group = channels // num_groups
    group_size = channels_per_group * spatial_size
    base_idx = batch_idx * channels * spatial_size + group_idx * channels_per_group * spatial_size

    # Compute mean over group
    sum_val = ct.float32(0.0)
    for i in range(tid, group_size, tile_size):
        sum_val = sum_val + ${inputName}[base_idx + i]

    for offset in [16, 8, 4, 2, 1]:
        sum_val = sum_val + ct.shfl_down(sum_val, offset)

    num_warps = tile_size // WARP_SIZE
    shared_sum = ct.shared_zeros((num_warps,), dtype=ct.float32)
    if lane == 0:
        shared_sum[warp_id] = sum_val
    ct.sync_threads()

    if warp_id == 0 and lane < num_warps:
        sum_val = shared_sum[lane]
        for offset in [16, 8, 4, 2, 1]:
            sum_val = sum_val + ct.shfl_down(sum_val, offset)
        if lane == 0:
            shared_sum[0] = sum_val / ct.float32(group_size)
    ct.sync_threads()
    mean = shared_sum[0]

    # Compute variance
    var_sum = ct.float32(0.0)
    for i in range(tid, group_size, tile_size):
        diff = ${inputName}[base_idx + i] - mean
        var_sum = var_sum + diff * diff

    for offset in [16, 8, 4, 2, 1]:
        var_sum = var_sum + ct.shfl_down(var_sum, offset)

    if lane == 0:
        shared_sum[warp_id] = var_sum
    ct.sync_threads()

    if warp_id == 0 and lane < num_warps:
        var_sum = shared_sum[lane]
        for offset in [16, 8, 4, 2, 1]:
            var_sum = var_sum + ct.shfl_down(var_sum, offset)
        if lane == 0:
            shared_sum[0] = ct.rsqrt(var_sum / ct.float32(group_size) + EPSILON)
    ct.sync_threads()
    rstd = shared_sum[0]

    # Normalize and apply affine
    for i in range(tid, group_size, tile_size):
        c_in_group = i // spatial_size
        c_global = group_idx * channels_per_group + c_in_group
        val = ${inputName}[base_idx + i]
        normalized = (val - mean) * rstd
        ${outputName}[base_idx + i] = ${gammaName}[c_global] * normalized + ${betaName}[c_global]


def launch_${ir.name}(${inputName}, ${gammaName}, ${betaName}, ${outputName}):
    """Launch GroupNorm kernel"""
    batch_size, channels, h, w = ${inputName}.shape
    spatial_size = h * w
    grid = (batch_size, NUM_GROUPS, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${gammaName}, ${betaName}, ${outputName}, batch_size, channels, spatial_size, NUM_GROUPS, TILE_SIZE))`;
}

/**
 * Generate BatchNorm kernel (inference mode)
 */
export function generateBatchNorm(ir: NormalizationIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const epsilon = ir.normConfig?.epsilon || 1e-5;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const meanName = ir.loads[1]?.source || 'running_mean';
  const varName = ir.loads[2]?.source || 'running_var';
  const gammaName = ir.loads[3]?.source || 'gamma';
  const betaName = ir.loads[4]?.source || 'beta';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
EPSILON = ${epsilon}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${meanName}, ${varName}, ${gammaName}, ${betaName}, ${outputName},
    batch_size: ct.Constant[int],
    channels: ct.Constant[int],
    spatial_size: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Batch Normalization (Inference) kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: batchnorm

    Uses pre-computed running mean/variance for inference.
    y = gamma * (x - mean) / sqrt(var + eps) + beta
    """
    total_out = batch_size * channels * spatial_size
    flat_idx = ct.bid(0) * tile_size + ct.tid(0)

    if flat_idx >= total_out:
        return

    # Decompose index
    b = flat_idx // (channels * spatial_size)
    rem = flat_idx % (channels * spatial_size)
    c = rem // spatial_size

    # Load statistics for this channel
    mean = ${meanName}[c]
    inv_std = ct.rsqrt(${varName}[c] + EPSILON)
    gamma = ${gammaName}[c]
    beta = ${betaName}[c]

    # Normalize
    val = ${inputName}[flat_idx]
    normalized = (val - mean) * inv_std
    ${outputName}[flat_idx] = gamma * normalized + beta


def launch_${ir.name}(${inputName}, ${meanName}, ${varName}, ${gammaName}, ${betaName}, ${outputName}):
    """Launch BatchNorm inference kernel"""
    batch_size, channels, h, w = ${inputName}.shape
    spatial_size = h * w
    total_out = batch_size * channels * spatial_size
    grid = (ct.cdiv(total_out, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${meanName}, ${varName}, ${gammaName}, ${betaName}, ${outputName}, batch_size, channels, spatial_size, TILE_SIZE))`;
}

/**
 * Generate InstanceNorm kernel
 */
export function generateInstanceNorm(ir: NormalizationIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const epsilon = ir.normConfig?.epsilon || 1e-5;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const gammaName = ir.loads[1]?.source || 'gamma';
  const betaName = ir.loads[2]?.source || 'beta';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
EPSILON = ${epsilon}
WARP_SIZE = 32

@ct.kernel
def ${ir.name}(
    ${inputName}, ${gammaName}, ${betaName}, ${outputName},
    batch_size: ct.Constant[int],
    channels: ct.Constant[int],
    spatial_size: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Instance Normalization kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: instancenorm

    Normalizes each (batch, channel) instance independently.
    """
    tid = ct.tid(0)
    lane = tid % WARP_SIZE
    warp_id = tid // WARP_SIZE
    batch_idx = ct.bid(0)
    channel_idx = ct.bid(1)

    if batch_idx >= batch_size or channel_idx >= channels:
        return

    base_idx = batch_idx * channels * spatial_size + channel_idx * spatial_size

    # Compute mean
    sum_val = ct.float32(0.0)
    for i in range(tid, spatial_size, tile_size):
        sum_val = sum_val + ${inputName}[base_idx + i]

    for offset in [16, 8, 4, 2, 1]:
        sum_val = sum_val + ct.shfl_down(sum_val, offset)

    num_warps = tile_size // WARP_SIZE
    shared_sum = ct.shared_zeros((num_warps,), dtype=ct.float32)
    if lane == 0:
        shared_sum[warp_id] = sum_val
    ct.sync_threads()

    if warp_id == 0 and lane < num_warps:
        sum_val = shared_sum[lane]
        for offset in [16, 8, 4, 2, 1]:
            sum_val = sum_val + ct.shfl_down(sum_val, offset)
        if lane == 0:
            shared_sum[0] = sum_val / ct.float32(spatial_size)
    ct.sync_threads()
    mean = shared_sum[0]

    # Compute variance
    var_sum = ct.float32(0.0)
    for i in range(tid, spatial_size, tile_size):
        diff = ${inputName}[base_idx + i] - mean
        var_sum = var_sum + diff * diff

    for offset in [16, 8, 4, 2, 1]:
        var_sum = var_sum + ct.shfl_down(var_sum, offset)

    if lane == 0:
        shared_sum[warp_id] = var_sum
    ct.sync_threads()

    if warp_id == 0 and lane < num_warps:
        var_sum = shared_sum[lane]
        for offset in [16, 8, 4, 2, 1]:
            var_sum = var_sum + ct.shfl_down(var_sum, offset)
        if lane == 0:
            shared_sum[0] = ct.rsqrt(var_sum / ct.float32(spatial_size) + EPSILON)
    ct.sync_threads()
    rstd = shared_sum[0]

    gamma = ${gammaName}[channel_idx]
    beta = ${betaName}[channel_idx]

    for i in range(tid, spatial_size, tile_size):
        val = ${inputName}[base_idx + i]
        normalized = (val - mean) * rstd
        ${outputName}[base_idx + i] = gamma * normalized + beta


def launch_${ir.name}(${inputName}, ${gammaName}, ${betaName}, ${outputName}):
    """Launch InstanceNorm kernel"""
    batch_size, channels, h, w = ${inputName}.shape
    spatial_size = h * w
    grid = (batch_size, channels, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${gammaName}, ${betaName}, ${outputName}, batch_size, channels, spatial_size, TILE_SIZE))`;
}

/**
 * Get the appropriate normalization template generator based on variant
 */
export function getNormalizationGenerator(variant?: string): (ir: NormalizationIR) => string {
  switch (variant) {
    case 'layernorm':
      return generateLayerNorm;
    case 'rmsnorm':
      return generateRMSNorm;
    case 'groupnorm':
      return generateGroupNorm;
    case 'batchnorm':
      return generateBatchNorm;
    case 'instancenorm':
      return generateInstanceNorm;
    default:
      return generateLayerNorm;
  }
}
