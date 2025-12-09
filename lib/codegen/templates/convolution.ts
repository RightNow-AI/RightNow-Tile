/**
 * Convolution Pattern Templates
 * Specialized code generation for various convolution types
 * Supports 1D, 2D, 3D, depthwise, grouped, im2col, and Winograd variants
 */

import { EnhancedKernelIR } from '../../ir/types';

/**
 * Convolution IR extension
 */
export interface ConvolutionIR extends EnhancedKernelIR {
  convConfig?: {
    convType: 'conv1d' | 'conv2d' | 'conv3d' | 'depthwise' | 'grouped' | 'transposed';
    kernelSize: number[];
    stride: number[];
    padding: number[];
    dilation: number[];
    groups: number;
    inChannels: number;
    outChannels: number;
    layout: 'NCHW' | 'NHWC';
  };
}

/**
 * Generate 1D convolution kernel
 */
export function generateConv1D(ir: ConvolutionIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const kernelSize = ir.convConfig?.kernelSize[0] || 3;
  const stride = ir.convConfig?.stride[0] || 1;
  const padding = ir.convConfig?.padding[0] || 0;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const weightName = ir.loads[1]?.source || 'weight';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
KERNEL_SIZE = ${kernelSize}
STRIDE = ${stride}
PADDING = ${padding}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${weightName}, ${outputName},
    batch_size: ct.Constant[int],
    in_channels: ct.Constant[int],
    out_channels: ct.Constant[int],
    seq_len: ct.Constant[int],
    out_len: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    1D Convolution kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: conv_1d

    Input: [batch, in_channels, seq_len]
    Weight: [out_channels, in_channels, kernel_size]
    Output: [batch, out_channels, out_len]
    """
    pid = ct.bid(0)
    tid = ct.tid(0)

    # Calculate output position
    total_out = batch_size * out_channels * out_len
    flat_idx = pid * tile_size + tid

    if flat_idx >= total_out:
        return

    # Decompose flat index
    b = flat_idx // (out_channels * out_len)
    rem = flat_idx % (out_channels * out_len)
    oc = rem // out_len
    ox = rem % out_len

    # Compute convolution
    sum_val = ct.float32(0.0)
    ix_start = ox * STRIDE - PADDING

    for ic in range(in_channels):
        for kx in range(KERNEL_SIZE):
            ix = ix_start + kx
            if ix >= 0 and ix < seq_len:
                in_val = ${inputName}[b * in_channels * seq_len + ic * seq_len + ix]
                w_val = ${weightName}[oc * in_channels * KERNEL_SIZE + ic * KERNEL_SIZE + kx]
                sum_val = sum_val + in_val * w_val

    ${outputName}[flat_idx] = sum_val


def launch_${ir.name}(${inputName}, ${weightName}, ${outputName}):
    """Launch the ${ir.name} 1D convolution kernel"""
    batch_size, in_channels, seq_len = ${inputName}.shape
    out_channels = ${weightName}.shape[0]
    out_len = (seq_len + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
    total_out = batch_size * out_channels * out_len
    grid = (ct.cdiv(total_out, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${weightName}, ${outputName}, batch_size, in_channels, out_channels, seq_len, out_len, TILE_SIZE))`;
}

/**
 * Generate 2D convolution kernel (direct method)
 */
export function generateConv2D(ir: ConvolutionIR): string {
  const blockM = ir.tileConfig.blockM || 16;
  const blockN = ir.tileConfig.blockN || 16;
  const kernelH = ir.convConfig?.kernelSize[0] || 3;
  const kernelW = ir.convConfig?.kernelSize[1] || 3;
  const strideH = ir.convConfig?.stride[0] || 1;
  const strideW = ir.convConfig?.stride[1] || 1;
  const padH = ir.convConfig?.padding[0] || 0;
  const padW = ir.convConfig?.padding[1] || 0;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const weightName = ir.loads[1]?.source || 'weight';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

BLOCK_M = ${blockM}
BLOCK_N = ${blockN}
KERNEL_H = ${kernelH}
KERNEL_W = ${kernelW}
STRIDE_H = ${strideH}
STRIDE_W = ${strideW}
PAD_H = ${padH}
PAD_W = ${padW}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${weightName}, ${outputName},
    batch_size: ct.Constant[int],
    in_channels: ct.Constant[int],
    out_channels: ct.Constant[int],
    in_h: ct.Constant[int],
    in_w: ct.Constant[int],
    out_h: ct.Constant[int],
    out_w: ct.Constant[int]
):
    """
    2D Convolution kernel (direct) - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: conv_2d

    Input: [batch, in_channels, H, W] (NCHW)
    Weight: [out_channels, in_channels, kH, kW]
    Output: [batch, out_channels, outH, outW]
    """
    # Block and thread indices
    by = ct.bid(0)
    bx = ct.bid(1)
    ty = ct.tid(0) // BLOCK_N
    tx = ct.tid(0) % BLOCK_N

    # Output coordinates
    b = ct.bid(2)
    oc = by * BLOCK_M + ty
    oh = bx // ct.cdiv(out_w, BLOCK_N) * BLOCK_N + tx
    ow = bx % ct.cdiv(out_w, BLOCK_N) * BLOCK_N + tx

    if oc >= out_channels or oh >= out_h or ow >= out_w:
        return

    # Compute convolution
    sum_val = ct.float32(0.0)
    ih_start = oh * STRIDE_H - PAD_H
    iw_start = ow * STRIDE_W - PAD_W

    for ic in range(in_channels):
        for kh in range(KERNEL_H):
            for kw in range(KERNEL_W):
                ih = ih_start + kh
                iw = iw_start + kw

                if ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
                    in_idx = b * in_channels * in_h * in_w + ic * in_h * in_w + ih * in_w + iw
                    w_idx = oc * in_channels * KERNEL_H * KERNEL_W + ic * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw
                    sum_val = sum_val + ${inputName}[in_idx] * ${weightName}[w_idx]

    out_idx = b * out_channels * out_h * out_w + oc * out_h * out_w + oh * out_w + ow
    ${outputName}[out_idx] = sum_val


def launch_${ir.name}(${inputName}, ${weightName}, ${outputName}):
    """Launch the ${ir.name} 2D convolution kernel"""
    batch_size, in_channels, in_h, in_w = ${inputName}.shape
    out_channels = ${weightName}.shape[0]
    out_h = (in_h + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    out_w = (in_w + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1
    grid = (ct.cdiv(out_channels, BLOCK_M), ct.cdiv(out_h * out_w, BLOCK_N), batch_size)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${weightName}, ${outputName}, batch_size, in_channels, out_channels, in_h, in_w, out_h, out_w))`;
}

/**
 * Generate depthwise convolution kernel
 * Each input channel is convolved with its own filter
 */
export function generateConvDepthwise(ir: ConvolutionIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const kernelH = ir.convConfig?.kernelSize[0] || 3;
  const kernelW = ir.convConfig?.kernelSize[1] || 3;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const weightName = ir.loads[1]?.source || 'weight';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
KERNEL_H = ${kernelH}
KERNEL_W = ${kernelW}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${weightName}, ${outputName},
    batch_size: ct.Constant[int],
    channels: ct.Constant[int],
    in_h: ct.Constant[int],
    in_w: ct.Constant[int],
    out_h: ct.Constant[int],
    out_w: ct.Constant[int],
    stride_h: ct.Constant[int],
    stride_w: ct.Constant[int],
    pad_h: ct.Constant[int],
    pad_w: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Depthwise Convolution kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: conv_depthwise

    Each channel convolved independently.
    Input: [batch, channels, H, W]
    Weight: [channels, 1, kH, kW]
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

    # Compute depthwise convolution
    sum_val = ct.float32(0.0)
    ih_start = oh * stride_h - pad_h
    iw_start = ow * stride_w - pad_w

    for kh in range(KERNEL_H):
        for kw in range(KERNEL_W):
            ih = ih_start + kh
            iw = iw_start + kw

            if ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
                in_idx = b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw
                w_idx = c * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw
                sum_val = sum_val + ${inputName}[in_idx] * ${weightName}[w_idx]

    ${outputName}[flat_idx] = sum_val


def launch_${ir.name}(${inputName}, ${weightName}, ${outputName}, stride_h=1, stride_w=1, pad_h=0, pad_w=0):
    """Launch the ${ir.name} depthwise convolution kernel"""
    batch_size, channels, in_h, in_w = ${inputName}.shape
    out_h = (in_h + 2 * pad_h - KERNEL_H) // stride_h + 1
    out_w = (in_w + 2 * pad_w - KERNEL_W) // stride_w + 1
    total_out = batch_size * channels * out_h * out_w
    grid = (ct.cdiv(total_out, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${weightName}, ${outputName}, batch_size, channels, in_h, in_w, out_h, out_w, stride_h, stride_w, pad_h, pad_w, TILE_SIZE))`;
}

/**
 * Generate im2col transformation kernel
 * Transforms input patches to columns for GEMM-based convolution
 */
export function generateIm2Col(ir: ConvolutionIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const kernelH = ir.convConfig?.kernelSize[0] || 3;
  const kernelW = ir.convConfig?.kernelSize[1] || 3;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const colName = ir.stores[0]?.target || 'col';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
KERNEL_H = ${kernelH}
KERNEL_W = ${kernelW}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${colName},
    batch_size: ct.Constant[int],
    in_channels: ct.Constant[int],
    in_h: ct.Constant[int],
    in_w: ct.Constant[int],
    out_h: ct.Constant[int],
    out_w: ct.Constant[int],
    stride_h: ct.Constant[int],
    stride_w: ct.Constant[int],
    pad_h: ct.Constant[int],
    pad_w: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Im2Col transformation kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: conv_im2col

    Extracts image patches to column format for GEMM.
    Input: [batch, channels, H, W]
    Output: [batch * out_h * out_w, channels * kH * kW]
    """
    total_cols = batch_size * out_h * out_w
    col_idx = ct.bid(0) * tile_size + ct.tid(0)

    if col_idx >= total_cols:
        return

    # Decompose column index
    b = col_idx // (out_h * out_w)
    rem = col_idx % (out_h * out_w)
    oh = rem // out_w
    ow = rem % out_w

    ih_start = oh * stride_h - pad_h
    iw_start = ow * stride_w - pad_w

    col_width = in_channels * KERNEL_H * KERNEL_W

    # Fill column for this output position
    for ic in range(in_channels):
        for kh in range(KERNEL_H):
            for kw in range(KERNEL_W):
                ih = ih_start + kh
                iw = iw_start + kw

                col_offset = ic * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw

                if ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
                    in_idx = b * in_channels * in_h * in_w + ic * in_h * in_w + ih * in_w + iw
                    ${colName}[col_idx * col_width + col_offset] = ${inputName}[in_idx]
                else:
                    ${colName}[col_idx * col_width + col_offset] = 0.0


def launch_${ir.name}(${inputName}, ${colName}, stride_h=1, stride_w=1, pad_h=0, pad_w=0):
    """Launch the ${ir.name} im2col kernel"""
    batch_size, in_channels, in_h, in_w = ${inputName}.shape
    out_h = (in_h + 2 * pad_h - KERNEL_H) // stride_h + 1
    out_w = (in_w + 2 * pad_w - KERNEL_W) // stride_w + 1
    total_cols = batch_size * out_h * out_w
    grid = (ct.cdiv(total_cols, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${colName}, batch_size, in_channels, in_h, in_w, out_h, out_w, stride_h, stride_w, pad_h, pad_w, TILE_SIZE))`;
}

/**
 * Generate Winograd convolution kernel (F(2,3) for 3x3 filters)
 * Reduces multiplications at the cost of additions
 */
export function generateConvWinograd(ir: ConvolutionIR): string {
  const tileSize = ir.tileConfig.tileSize || 64;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const weightName = ir.loads[1]?.source || 'weight';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
# Winograd F(2,3): 4x4 input tile -> 2x2 output tile with 3x3 filter

@ct.kernel
def ${ir.name}(
    ${inputName}, ${weightName}, ${outputName},
    batch_size: ct.Constant[int],
    in_channels: ct.Constant[int],
    out_channels: ct.Constant[int],
    in_h: ct.Constant[int],
    in_w: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Winograd F(2,3) Convolution kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: conv_winograd

    Uses Winograd algorithm for 3x3 convolution.
    Processes 4x4 input tiles to produce 2x2 output tiles.
    """
    # Tile indices
    tile_h = ct.bid(0)
    tile_w = ct.bid(1)
    tid = ct.tid(0)

    b = ct.bid(2) // out_channels
    oc = ct.bid(2) % out_channels

    # Output tile position
    oh_base = tile_h * 2
    ow_base = tile_w * 2

    out_h = in_h - 2  # For valid padding with 3x3
    out_w = in_w - 2

    if oh_base >= out_h or ow_base >= out_w:
        return

    # Initialize accumulator for Winograd domain
    U = ct.shared_zeros((4, 4), dtype=ct.float32)

    # For each input channel, transform and accumulate
    for ic in range(in_channels):
        # Load 4x4 input tile
        d = ct.zeros((4, 4), dtype=ct.float32)
        for i in range(4):
            for j in range(4):
                ih = oh_base + i
                iw = ow_base + j
                if ih < in_h and iw < in_w:
                    d[i, j] = ${inputName}[b * in_channels * in_h * in_w + ic * in_h * in_w + ih * in_w + iw]

        # Winograd input transform: BT d B
        # BT = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
        temp = ct.zeros((4, 4), dtype=ct.float32)
        temp[0, :] = d[0, :] - d[2, :]
        temp[1, :] = d[1, :] + d[2, :]
        temp[2, :] = -d[1, :] + d[2, :]
        temp[3, :] = d[1, :] - d[3, :]

        V = ct.zeros((4, 4), dtype=ct.float32)
        V[:, 0] = temp[:, 0] - temp[:, 2]
        V[:, 1] = temp[:, 1] + temp[:, 2]
        V[:, 2] = -temp[:, 1] + temp[:, 2]
        V[:, 3] = temp[:, 1] - temp[:, 3]

        # Load transformed filter (precomputed: G g GT)
        # For simplicity, compute on-the-fly here
        g = ct.zeros((3, 3), dtype=ct.float32)
        for i in range(3):
            for j in range(3):
                g[i, j] = ${weightName}[oc * in_channels * 9 + ic * 9 + i * 3 + j]

        # Filter transform
        Gg = ct.zeros((4, 3), dtype=ct.float32)
        Gg[0, :] = g[0, :]
        Gg[1, :] = 0.5 * (g[0, :] + g[1, :] + g[2, :])
        Gg[2, :] = 0.5 * (g[0, :] - g[1, :] + g[2, :])
        Gg[3, :] = g[2, :]

        GgGT = ct.zeros((4, 4), dtype=ct.float32)
        GgGT[:, 0] = Gg[:, 0]
        GgGT[:, 1] = 0.5 * (Gg[:, 0] + Gg[:, 1] + Gg[:, 2])
        GgGT[:, 2] = 0.5 * (Gg[:, 0] - Gg[:, 1] + Gg[:, 2])
        GgGT[:, 3] = Gg[:, 2]

        # Element-wise multiply and accumulate
        U = U + V * GgGT

    ct.sync_threads()

    # Output inverse transform: AT M A
    # AT = [[1, 1, 1, 0], [0, 1, -1, -1]]
    temp2 = ct.zeros((2, 4), dtype=ct.float32)
    temp2[0, :] = U[0, :] + U[1, :] + U[2, :]
    temp2[1, :] = U[1, :] - U[2, :] - U[3, :]

    result = ct.zeros((2, 2), dtype=ct.float32)
    result[:, 0] = temp2[:, 0] + temp2[:, 1] + temp2[:, 2]
    result[:, 1] = temp2[:, 1] - temp2[:, 2] - temp2[:, 3]

    # Store 2x2 output tile
    for i in range(2):
        for j in range(2):
            if oh_base + i < out_h and ow_base + j < out_w:
                out_idx = b * out_channels * out_h * out_w + oc * out_h * out_w + (oh_base + i) * out_w + (ow_base + j)
                ${outputName}[out_idx] = result[i, j]


def launch_${ir.name}(${inputName}, ${weightName}, ${outputName}):
    """Launch the ${ir.name} Winograd convolution kernel"""
    batch_size, in_channels, in_h, in_w = ${inputName}.shape
    out_channels = ${weightName}.shape[0]
    out_h = in_h - 2
    out_w = in_w - 2
    tile_h = ct.cdiv(out_h, 2)
    tile_w = ct.cdiv(out_w, 2)
    grid = (tile_h, tile_w, batch_size * out_channels)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${weightName}, ${outputName}, batch_size, in_channels, out_channels, in_h, in_w, TILE_SIZE))`;
}

/**
 * Generate 3D convolution kernel
 */
export function generateConv3D(ir: ConvolutionIR): string {
  const tileSize = ir.tileConfig.tileSize || 128;
  const kernelD = ir.convConfig?.kernelSize[0] || 3;
  const kernelH = ir.convConfig?.kernelSize[1] || 3;
  const kernelW = ir.convConfig?.kernelSize[2] || 3;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const weightName = ir.loads[1]?.source || 'weight';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
KERNEL_D = ${kernelD}
KERNEL_H = ${kernelH}
KERNEL_W = ${kernelW}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${weightName}, ${outputName},
    batch_size: ct.Constant[int],
    in_channels: ct.Constant[int],
    out_channels: ct.Constant[int],
    in_d: ct.Constant[int],
    in_h: ct.Constant[int],
    in_w: ct.Constant[int],
    out_d: ct.Constant[int],
    out_h: ct.Constant[int],
    out_w: ct.Constant[int],
    stride_d: ct.Constant[int],
    stride_h: ct.Constant[int],
    stride_w: ct.Constant[int],
    pad_d: ct.Constant[int],
    pad_h: ct.Constant[int],
    pad_w: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    3D Convolution kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: conv_3d

    Input: [batch, in_channels, D, H, W]
    Weight: [out_channels, in_channels, kD, kH, kW]
    Output: [batch, out_channels, outD, outH, outW]
    """
    total_out = batch_size * out_channels * out_d * out_h * out_w
    flat_idx = ct.bid(0) * tile_size + ct.tid(0)

    if flat_idx >= total_out:
        return

    # Decompose flat index
    b = flat_idx // (out_channels * out_d * out_h * out_w)
    rem = flat_idx % (out_channels * out_d * out_h * out_w)
    oc = rem // (out_d * out_h * out_w)
    rem2 = rem % (out_d * out_h * out_w)
    od = rem2 // (out_h * out_w)
    rem3 = rem2 % (out_h * out_w)
    oh = rem3 // out_w
    ow = rem3 % out_w

    # Compute 3D convolution
    sum_val = ct.float32(0.0)
    id_start = od * stride_d - pad_d
    ih_start = oh * stride_h - pad_h
    iw_start = ow * stride_w - pad_w

    for ic in range(in_channels):
        for kd in range(KERNEL_D):
            for kh in range(KERNEL_H):
                for kw in range(KERNEL_W):
                    id_pos = id_start + kd
                    ih = ih_start + kh
                    iw = iw_start + kw

                    if id_pos >= 0 and id_pos < in_d and ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
                        in_idx = b * in_channels * in_d * in_h * in_w + ic * in_d * in_h * in_w + id_pos * in_h * in_w + ih * in_w + iw
                        w_idx = oc * in_channels * KERNEL_D * KERNEL_H * KERNEL_W + ic * KERNEL_D * KERNEL_H * KERNEL_W + kd * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw
                        sum_val = sum_val + ${inputName}[in_idx] * ${weightName}[w_idx]

    ${outputName}[flat_idx] = sum_val


def launch_${ir.name}(${inputName}, ${weightName}, ${outputName}, stride=(1,1,1), padding=(0,0,0)):
    """Launch the ${ir.name} 3D convolution kernel"""
    batch_size, in_channels, in_d, in_h, in_w = ${inputName}.shape
    out_channels = ${weightName}.shape[0]
    out_d = (in_d + 2 * padding[0] - KERNEL_D) // stride[0] + 1
    out_h = (in_h + 2 * padding[1] - KERNEL_H) // stride[1] + 1
    out_w = (in_w + 2 * padding[2] - KERNEL_W) // stride[2] + 1
    total_out = batch_size * out_channels * out_d * out_h * out_w
    grid = (ct.cdiv(total_out, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${weightName}, ${outputName}, batch_size, in_channels, out_channels, in_d, in_h, in_w, out_d, out_h, out_w, stride[0], stride[1], stride[2], padding[0], padding[1], padding[2], TILE_SIZE))`;
}

/**
 * Generate grouped convolution kernel
 */
export function generateConvGrouped(ir: ConvolutionIR): string {
  const tileSize = ir.tileConfig.tileSize || 256;
  const kernelH = ir.convConfig?.kernelSize[0] || 3;
  const kernelW = ir.convConfig?.kernelSize[1] || 3;
  const groups = ir.convConfig?.groups || 2;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const weightName = ir.loads[1]?.source || 'weight';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
KERNEL_H = ${kernelH}
KERNEL_W = ${kernelW}
GROUPS = ${groups}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${weightName}, ${outputName},
    batch_size: ct.Constant[int],
    in_channels: ct.Constant[int],
    out_channels: ct.Constant[int],
    in_h: ct.Constant[int],
    in_w: ct.Constant[int],
    out_h: ct.Constant[int],
    out_w: ct.Constant[int],
    stride_h: ct.Constant[int],
    stride_w: ct.Constant[int],
    pad_h: ct.Constant[int],
    pad_w: ct.Constant[int],
    groups: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Grouped Convolution kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: conv_grouped

    Splits channels into groups, convolves each independently.
    Used in ResNeXt, ShuffleNet, etc.
    """
    total_out = batch_size * out_channels * out_h * out_w
    flat_idx = ct.bid(0) * tile_size + ct.tid(0)

    if flat_idx >= total_out:
        return

    # Decompose index
    b = flat_idx // (out_channels * out_h * out_w)
    rem = flat_idx % (out_channels * out_h * out_w)
    oc = rem // (out_h * out_w)
    rem2 = rem % (out_h * out_w)
    oh = rem2 // out_w
    ow = rem2 % out_w

    # Determine group
    out_channels_per_group = out_channels // groups
    in_channels_per_group = in_channels // groups
    group = oc // out_channels_per_group
    oc_in_group = oc % out_channels_per_group

    # Input channel range for this group
    ic_start = group * in_channels_per_group
    ic_end = ic_start + in_channels_per_group

    # Compute convolution within group
    sum_val = ct.float32(0.0)
    ih_start = oh * stride_h - pad_h
    iw_start = ow * stride_w - pad_w

    for ic_offset in range(in_channels_per_group):
        ic = ic_start + ic_offset
        for kh in range(KERNEL_H):
            for kw in range(KERNEL_W):
                ih = ih_start + kh
                iw = iw_start + kw

                if ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
                    in_idx = b * in_channels * in_h * in_w + ic * in_h * in_w + ih * in_w + iw
                    # Weight layout: [out_channels, in_channels/groups, kH, kW]
                    w_idx = oc * in_channels_per_group * KERNEL_H * KERNEL_W + ic_offset * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw
                    sum_val = sum_val + ${inputName}[in_idx] * ${weightName}[w_idx]

    ${outputName}[flat_idx] = sum_val


def launch_${ir.name}(${inputName}, ${weightName}, ${outputName}, stride=(1,1), padding=(0,0)):
    """Launch the ${ir.name} grouped convolution kernel"""
    batch_size, in_channels, in_h, in_w = ${inputName}.shape
    out_channels = ${weightName}.shape[0]
    out_h = (in_h + 2 * padding[0] - KERNEL_H) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - KERNEL_W) // stride[1] + 1
    total_out = batch_size * out_channels * out_h * out_w
    grid = (ct.cdiv(total_out, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${weightName}, ${outputName}, batch_size, in_channels, out_channels, in_h, in_w, out_h, out_w, stride[0], stride[1], padding[0], padding[1], GROUPS, TILE_SIZE))`;
}

/**
 * Get the appropriate convolution template generator based on variant
 */
export function getConvolutionGenerator(variant?: string): (ir: ConvolutionIR) => string {
  switch (variant) {
    case 'conv_1d':
      return generateConv1D;
    case 'conv_2d':
      return generateConv2D;
    case 'conv_3d':
      return generateConv3D;
    case 'conv_depthwise':
      return generateConvDepthwise;
    case 'conv_grouped':
      return generateConvGrouped;
    case 'conv_winograd':
      return generateConvWinograd;
    case 'conv_im2col':
      return generateIm2Col;
    default:
      return generateConv2D;
  }
}
