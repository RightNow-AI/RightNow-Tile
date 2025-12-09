/**
 * Quantization Pattern Templates
 * Specialized code generation for INT8, INT4, and FP8 quantization/dequantization
 */

import { EnhancedKernelIR } from '../../ir/types';

/**
 * Quantization IR extension
 */
export interface QuantizationTemplateIR extends EnhancedKernelIR {
  quantConfig?: {
    bits: number;
    isSymmetric: boolean;
    perChannel: boolean;
    groupSize?: number;
  };
}

/**
 * Generate INT8 quantization kernel
 */
export function generateQuantizeInt8(ir: QuantizationTemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const isSymmetric = ir.quantConfig?.isSymmetric ?? true;
  const perChannel = ir.quantConfig?.perChannel ?? false;

  const inputName = ir.loads[0]?.source || 'input';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
QMIN = -128
QMAX = 127

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName}, scale, ${isSymmetric ? '' : 'zero_point,'}
    n: ct.Constant[int],
    ${perChannel ? 'num_channels: ct.Constant[int],' : ''}
    tile_size: ct.Constant[int]
):
    """
    INT8 Quantization kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: quant_int8

    Quantizes float32 to int8.
    Mode: ${isSymmetric ? 'symmetric' : 'asymmetric'}, ${perChannel ? 'per-channel' : 'per-tensor'}
    """
    pid = ct.bid(0)
    tid = ct.tid(0)
    idx = pid * tile_size + tid

    if idx >= n:
        return

    ${perChannel ? `
    # Per-channel quantization
    channel_idx = idx % num_channels
    s = scale[channel_idx]
    ${isSymmetric ? '' : 'zp = zero_point[channel_idx]'}
    ` : `
    # Per-tensor quantization
    s = scale[0]
    ${isSymmetric ? '' : 'zp = zero_point[0]'}
    `}

    val = ${inputName}[idx]

    ${isSymmetric ? `
    # Symmetric quantization: q = round(x / scale)
    q = ct.round(val / s)
    q = ct.clamp(q, QMIN, QMAX)
    ` : `
    # Asymmetric quantization: q = round(x / scale) + zero_point
    q = ct.round(val / s) + zp
    q = ct.clamp(q, QMIN, QMAX)
    `}

    ${outputName}[idx] = ct.int8(q)


@ct.kernel
def ${ir.name}_compute_scale(
    ${inputName}, scale, ${isSymmetric ? '' : 'zero_point,'}
    n: ct.Constant[int],
    ${perChannel ? 'num_channels: ct.Constant[int],' : ''}
    tile_size: ct.Constant[int]
):
    """
    Compute quantization scale (and zero point for asymmetric)
    """
    ${perChannel ? `
    channel_idx = ct.bid(0)
    if channel_idx >= num_channels:
        return

    # Find min/max for this channel
    min_val = ct.float32('inf')
    max_val = ct.float32('-inf')

    for i in range(channel_idx, n, num_channels):
        val = ${inputName}[i]
        min_val = ct.minimum(min_val, val)
        max_val = ct.maximum(max_val, val)
    ` : `
    tid = ct.tid(0)
    pid = ct.bid(0)

    # Shared memory for reduction
    shared_min = ct.shared_zeros((tile_size,), dtype=ct.float32)
    shared_max = ct.shared_zeros((tile_size,), dtype=ct.float32)

    # Initialize
    local_min = ct.float32('inf')
    local_max = ct.float32('-inf')

    # Grid-stride loop
    for i in range(pid * tile_size + tid, n, ct.gridDim(0) * tile_size):
        val = ${inputName}[i]
        local_min = ct.minimum(local_min, val)
        local_max = ct.maximum(local_max, val)

    shared_min[tid] = local_min
    shared_max[tid] = local_max
    ct.sync_threads()

    # Reduction
    stride = tile_size // 2
    while stride > 0:
        if tid < stride:
            shared_min[tid] = ct.minimum(shared_min[tid], shared_min[tid + stride])
            shared_max[tid] = ct.maximum(shared_max[tid], shared_max[tid + stride])
        ct.sync_threads()
        stride //= 2

    if tid == 0:
        min_val = shared_min[0]
        max_val = shared_max[0]
    `}

    ${isSymmetric ? `
    # Symmetric: scale = max(abs(min), abs(max)) / 127
    abs_max = ct.maximum(ct.abs(min_val), ct.abs(max_val))
    s = abs_max / 127.0
    if s == 0:
        s = 1.0
    ${perChannel ? 'scale[channel_idx] = s' : 'if tid == 0: scale[0] = s'}
    ` : `
    # Asymmetric: scale = (max - min) / 255, zero_point = round(-min / scale)
    s = (max_val - min_val) / 255.0
    if s == 0:
        s = 1.0
    zp = ct.round(-min_val / s)
    zp = ct.clamp(zp, QMIN, QMAX)
    ${perChannel ? `
    scale[channel_idx] = s
    zero_point[channel_idx] = ct.int8(zp)
    ` : `
    if tid == 0:
        scale[0] = s
        zero_point[0] = ct.int8(zp)
    `}
    `}


def launch_${ir.name}(${inputName}, ${outputName}=None, scale=None${isSymmetric ? '' : ', zero_point=None'}):
    """Launch INT8 quantization"""
    n = ${inputName}.size
    ${perChannel ? 'num_channels = ' + inputName + '.shape[-1]' : ''}

    if ${outputName} is None:
        ${outputName} = cupy.empty(${inputName}.shape, dtype=cupy.int8)

    if scale is None:
        ${perChannel ? 'scale = cupy.empty((num_channels,), dtype=cupy.float32)' : 'scale = cupy.empty((1,), dtype=cupy.float32)'}
        ${isSymmetric ? '' : (perChannel ? 'zero_point = cupy.empty((num_channels,), dtype=cupy.int8)' : 'zero_point = cupy.empty((1,), dtype=cupy.int8)')}

        # Compute scale
        ${perChannel ? `
        grid = (num_channels, 1, 1)
        ` : `
        grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
        `}
        stream = cupy.cuda.get_current_stream()
        ct.launch(stream, grid, ${ir.name}_compute_scale, (${inputName}.ravel(), scale${isSymmetric ? '' : ', zero_point'}, n, ${perChannel ? 'num_channels, ' : ''}TILE_SIZE))

    # Quantize
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}.ravel(), ${outputName}.ravel(), scale${isSymmetric ? '' : ', zero_point'}, n, ${perChannel ? 'num_channels, ' : ''}TILE_SIZE))

    return ${outputName}, scale${isSymmetric ? '' : ', zero_point'}`;
}

/**
 * Generate INT4 quantization kernel
 */
export function generateQuantizeInt4(ir: QuantizationTemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const groupSize = ir.quantConfig?.groupSize || 128;

  const inputName = ir.loads[0]?.source || 'input';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
GROUP_SIZE = ${groupSize}
QMIN = -8
QMAX = 7

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName}, scale, zero_point,
    n: ct.Constant[int],
    group_size: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    INT4 Quantization kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: quant_int4

    Quantizes float32 to int4 (packed as uint8, 2 values per byte).
    Uses group-wise quantization.
    """
    pid = ct.bid(0)
    tid = ct.tid(0)
    idx = pid * tile_size + tid

    # Each output byte holds 2 int4 values
    out_idx = idx // 2
    is_high = idx % 2

    if idx >= n:
        return

    # Get group index for scale/zero_point
    group_idx = idx // group_size

    s = scale[group_idx]
    zp = zero_point[group_idx]

    val = ${inputName}[idx]

    # Quantize to int4
    q = ct.round(val / s) + zp
    q = ct.clamp(q, QMIN, QMAX)

    # Pack into nibble (4 bits)
    q_uint = ct.uint8(q + 8)  # Shift to 0-15 range

    if is_high:
        # High nibble
        ct.atomic_or(${outputName}, out_idx, q_uint << 4)
    else:
        # Low nibble
        ct.atomic_or(${outputName}, out_idx, q_uint)


@ct.kernel
def ${ir.name}_compute_scale(
    ${inputName}, scale, zero_point,
    n: ct.Constant[int],
    group_size: ct.Constant[int],
    num_groups: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Compute group-wise quantization parameters
    """
    group_idx = ct.bid(0)
    tid = ct.tid(0)

    if group_idx >= num_groups:
        return

    start = group_idx * group_size
    end = ct.minimum(start + group_size, n)

    # Find min/max in group
    shared_min = ct.shared_zeros((tile_size,), dtype=ct.float32)
    shared_max = ct.shared_zeros((tile_size,), dtype=ct.float32)

    local_min = ct.float32('inf')
    local_max = ct.float32('-inf')

    for i in range(start + tid, end, tile_size):
        val = ${inputName}[i]
        local_min = ct.minimum(local_min, val)
        local_max = ct.maximum(local_max, val)

    shared_min[tid] = local_min
    shared_max[tid] = local_max
    ct.sync_threads()

    # Reduction
    stride = tile_size // 2
    while stride > 0:
        if tid < stride:
            shared_min[tid] = ct.minimum(shared_min[tid], shared_min[tid + stride])
            shared_max[tid] = ct.maximum(shared_max[tid], shared_max[tid + stride])
        ct.sync_threads()
        stride //= 2

    if tid == 0:
        min_val = shared_min[0]
        max_val = shared_max[0]

        # Asymmetric quantization for int4
        s = (max_val - min_val) / 15.0
        if s == 0:
            s = 1.0
        zp = ct.round(-min_val / s)
        zp = ct.clamp(zp, QMIN, QMAX)

        scale[group_idx] = s
        zero_point[group_idx] = ct.int8(zp)


def launch_${ir.name}(${inputName}, ${outputName}=None, scale=None, zero_point=None, group_size=${groupSize}):
    """Launch INT4 quantization"""
    n = ${inputName}.size
    num_groups = (n + group_size - 1) // group_size

    if ${outputName} is None:
        # Packed output: 2 int4 values per byte
        ${outputName} = cupy.zeros((n + 1) // 2, dtype=cupy.uint8)

    if scale is None:
        scale = cupy.empty((num_groups,), dtype=cupy.float32)
        zero_point = cupy.empty((num_groups,), dtype=cupy.int8)

        # Compute scales
        grid = (num_groups, 1, 1)
        stream = cupy.cuda.get_current_stream()
        ct.launch(stream, grid, ${ir.name}_compute_scale, (${inputName}.ravel(), scale, zero_point, n, group_size, num_groups, TILE_SIZE))

    # Quantize
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}.ravel(), ${outputName}, scale, zero_point, n, group_size, TILE_SIZE))

    return ${outputName}, scale, zero_point`;
}

/**
 * Generate FP8 quantization kernel
 */
export function generateQuantizeFP8(ir: QuantizationTemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;

  const inputName = ir.loads[0]?.source || 'input';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

# FP8 E4M3 format constants
FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN = -448.0

# FP8 E5M2 format constants
FP8_E5M2_MAX = 57344.0
FP8_E5M2_MIN = -57344.0

@ct.kernel
def ${ir.name}_e4m3(
    ${inputName}, ${outputName}, scale,
    n: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    FP8 (E4M3) Quantization kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: quant_fp8

    Quantizes float32 to FP8 E4M3 format.
    E4M3: 4-bit exponent, 3-bit mantissa (better for weights)
    """
    pid = ct.bid(0)
    tid = ct.tid(0)
    idx = pid * tile_size + tid

    if idx >= n:
        return

    s = scale[0]
    val = ${inputName}[idx] / s

    # Clamp to FP8 E4M3 range
    val = ct.clamp(val, FP8_E4M3_MIN, FP8_E4M3_MAX)

    # Convert to FP8 (using hardware if available, else software emulation)
    ${outputName}[idx] = ct.fp8_e4m3(val)


@ct.kernel
def ${ir.name}_e5m2(
    ${inputName}, ${outputName}, scale,
    n: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    FP8 (E5M2) Quantization kernel

    E5M2: 5-bit exponent, 2-bit mantissa (better for gradients)
    """
    pid = ct.bid(0)
    tid = ct.tid(0)
    idx = pid * tile_size + tid

    if idx >= n:
        return

    s = scale[0]
    val = ${inputName}[idx] / s

    # Clamp to FP8 E5M2 range
    val = ct.clamp(val, FP8_E5M2_MIN, FP8_E5M2_MAX)

    ${outputName}[idx] = ct.fp8_e5m2(val)


@ct.kernel
def ${ir.name}_compute_scale(
    ${inputName}, scale,
    n: ct.Constant[int],
    max_val: ct.Constant[float],
    tile_size: ct.Constant[int]
):
    """
    Compute optimal scale for FP8 quantization
    """
    tid = ct.tid(0)
    pid = ct.bid(0)

    shared_max = ct.shared_zeros((tile_size,), dtype=ct.float32)

    local_max = ct.float32(0.0)

    for i in range(pid * tile_size + tid, n, ct.gridDim(0) * tile_size):
        local_max = ct.maximum(local_max, ct.abs(${inputName}[i]))

    shared_max[tid] = local_max
    ct.sync_threads()

    stride = tile_size // 2
    while stride > 0:
        if tid < stride:
            shared_max[tid] = ct.maximum(shared_max[tid], shared_max[tid + stride])
        ct.sync_threads()
        stride //= 2

    if tid == 0 and pid == 0:
        amax = shared_max[0]
        # Scale to use full FP8 range
        s = amax / max_val if amax > 0 else 1.0
        scale[0] = s


def launch_${ir.name}(${inputName}, ${outputName}=None, scale=None, format='e4m3'):
    """Launch FP8 quantization"""
    n = ${inputName}.size

    if ${outputName} is None:
        ${outputName} = cupy.empty(${inputName}.shape, dtype=cupy.uint8)  # FP8 stored as uint8

    max_val = FP8_E4M3_MAX if format == 'e4m3' else FP8_E5M2_MAX

    if scale is None:
        scale = cupy.empty((1,), dtype=cupy.float32)
        grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
        stream = cupy.cuda.get_current_stream()
        ct.launch(stream, grid, ${ir.name}_compute_scale, (${inputName}.ravel(), scale, n, max_val, TILE_SIZE))

    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()

    kernel = ${ir.name}_e4m3 if format == 'e4m3' else ${ir.name}_e5m2
    ct.launch(stream, grid, kernel, (${inputName}.ravel(), ${outputName}.ravel(), scale, n, TILE_SIZE))

    return ${outputName}, scale`;
}

/**
 * Generate dequantization kernel
 */
export function generateDequantize(ir: QuantizationTemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const bits = ir.quantConfig?.bits || 8;

  const inputName = ir.loads[0]?.source || 'input';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}_int8(
    ${inputName}, ${outputName}, scale, zero_point,
    n: ct.Constant[int],
    has_zero_point: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    INT8 Dequantization kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: dequantize

    Dequantizes int8 back to float32.
    """
    pid = ct.bid(0)
    tid = ct.tid(0)
    idx = pid * tile_size + tid

    if idx >= n:
        return

    s = scale[0]
    q = ct.float32(${inputName}[idx])

    if has_zero_point:
        zp = ct.float32(zero_point[0])
        ${outputName}[idx] = (q - zp) * s
    else:
        ${outputName}[idx] = q * s


@ct.kernel
def ${ir.name}_int4(
    ${inputName}, ${outputName}, scale, zero_point,
    n: ct.Constant[int],
    group_size: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    INT4 Dequantization kernel
    Unpacks int4 from uint8 and dequantizes.
    """
    pid = ct.bid(0)
    tid = ct.tid(0)
    idx = pid * tile_size + tid

    if idx >= n:
        return

    # Get packed byte
    byte_idx = idx // 2
    is_high = idx % 2

    packed = ${inputName}[byte_idx]

    if is_high:
        q = ct.int8((packed >> 4) & 0xF) - 8
    else:
        q = ct.int8(packed & 0xF) - 8

    group_idx = idx // group_size
    s = scale[group_idx]
    zp = ct.float32(zero_point[group_idx])

    ${outputName}[idx] = (ct.float32(q) - zp) * s


@ct.kernel
def ${ir.name}_fp8(
    ${inputName}, ${outputName}, scale,
    n: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    FP8 Dequantization kernel
    """
    pid = ct.bid(0)
    tid = ct.tid(0)
    idx = pid * tile_size + tid

    if idx >= n:
        return

    s = scale[0]
    val = ct.float32_from_fp8(${inputName}[idx])
    ${outputName}[idx] = val * s


def launch_${ir.name}(${inputName}, scale, zero_point=None, ${outputName}=None, bits=${bits}, group_size=128):
    """Launch dequantization"""
    if bits == 4:
        n = ${inputName}.size * 2  # Packed int4
    else:
        n = ${inputName}.size

    if ${outputName} is None:
        if bits == 4:
            ${outputName} = cupy.empty((n,), dtype=cupy.float32)
        else:
            ${outputName} = cupy.empty(${inputName}.shape, dtype=cupy.float32)

    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()

    if bits == 8:
        has_zp = 1 if zero_point is not None else 0
        if zero_point is None:
            zero_point = cupy.zeros((1,), dtype=cupy.int8)
        ct.launch(stream, grid, ${ir.name}_int8, (${inputName}.ravel(), ${outputName}.ravel(), scale, zero_point, n, has_zp, TILE_SIZE))
    elif bits == 4:
        ct.launch(stream, grid, ${ir.name}_int4, (${inputName}, ${outputName}.ravel(), scale, zero_point, n, group_size, TILE_SIZE))
    else:  # FP8
        ct.launch(stream, grid, ${ir.name}_fp8, (${inputName}.ravel(), ${outputName}.ravel(), scale, n, TILE_SIZE))

    return ${outputName}`;
}

/**
 * Get the appropriate quantization template generator based on variant
 */
export function getQuantizationGenerator(variant?: string): (ir: QuantizationTemplateIR) => string {
  switch (variant) {
    case 'quant_int8':
    case 'quantize':
      return generateQuantizeInt8;
    case 'quant_int4':
      return generateQuantizeInt4;
    case 'quant_fp8':
      return generateQuantizeFP8;
    case 'dequantize':
      return generateDequantize;
    default:
      return generateQuantizeInt8;
  }
}
