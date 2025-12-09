/**
 * RoPE (Rotary Position Embedding) Pattern Templates
 * Specialized code generation for rotary position embeddings used in LLMs
 */

import { EnhancedKernelIR } from '../../ir/types';

/**
 * RoPE IR extension
 */
export interface RoPETemplateIR extends EnhancedKernelIR {
  ropeConfig?: {
    headDim: number;
    rotaryDim: number;
    base: number;
    maxSeqLen: number;
    variant: 'standard' | 'neox' | 'cached';
  };
}

/**
 * Generate standard RoPE kernel
 */
export function generateRoPEStandard(ir: RoPETemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 64;
  const headDim = ir.ropeConfig?.headDim || 64;
  const rotaryDim = ir.ropeConfig?.rotaryDim || headDim;
  const base = ir.ropeConfig?.base || 10000;

  const inputName = ir.loads[0]?.source || 'x';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy
import math

TILE_SIZE = ${tileSize}
HEAD_DIM = ${headDim}
ROTARY_DIM = ${rotaryDim}
ROPE_BASE = ${base}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName}, positions,
    batch_size: ct.Constant[int],
    num_heads: ct.Constant[int],
    seq_len: ct.Constant[int],
    head_dim: ct.Constant[int],
    rotary_dim: ct.Constant[int],
    rope_base: ct.Constant[float],
    tile_size: ct.Constant[int]
):
    """
    RoPE (Rotary Position Embedding) kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: rope_standard

    Applies rotary position embeddings to Q/K vectors.
    """
    # Grid: (batch * num_heads, seq_len, 1)
    batch_head_idx = ct.bid(0)
    seq_idx = ct.bid(1)
    tid = ct.tid(0)

    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Get position for this token
    pos = positions[batch_idx * seq_len + seq_idx]

    # Apply rotary embedding to pairs of dimensions
    for d in range(tid, rotary_dim // 2, tile_size):
        # Compute frequency
        freq = 1.0 / ct.pow(rope_base, ct.float32(2 * d) / ct.float32(rotary_dim))
        angle = ct.float32(pos) * freq

        cos_val = ct.cos(angle)
        sin_val = ct.sin(angle)

        # Index into the input tensor
        base_idx = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * head_dim

        # Get the pair of values
        x0 = ${inputName}[base_idx + d]
        x1 = ${inputName}[base_idx + d + rotary_dim // 2]

        # Apply rotation
        ${outputName}[base_idx + d] = x0 * cos_val - x1 * sin_val
        ${outputName}[base_idx + d + rotary_dim // 2] = x0 * sin_val + x1 * cos_val

    # Copy non-rotary dimensions unchanged
    for d in range(rotary_dim + tid, head_dim, tile_size):
        base_idx = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * head_dim
        ${outputName}[base_idx + d] = ${inputName}[base_idx + d]


def launch_${ir.name}(${inputName}, positions, ${outputName}=None):
    """Launch the RoPE kernel"""
    batch_size, num_heads, seq_len, head_dim = ${inputName}.shape

    if ${outputName} is None:
        ${outputName} = cupy.empty_like(${inputName})

    grid = (batch_size * num_heads, seq_len, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (
        ${inputName}.ravel(), ${outputName}.ravel(), positions.ravel(),
        batch_size, num_heads, seq_len, head_dim, ROTARY_DIM, ROPE_BASE, TILE_SIZE
    ))
    return ${outputName}`;
}

/**
 * Generate NeoX-style RoPE kernel (interleaved rotation)
 */
export function generateRoPENeox(ir: RoPETemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 64;
  const headDim = ir.ropeConfig?.headDim || 64;
  const base = ir.ropeConfig?.base || 10000;

  const inputName = ir.loads[0]?.source || 'x';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy
import math

TILE_SIZE = ${tileSize}
HEAD_DIM = ${headDim}
ROPE_BASE = ${base}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName}, positions,
    batch_size: ct.Constant[int],
    num_heads: ct.Constant[int],
    seq_len: ct.Constant[int],
    head_dim: ct.Constant[int],
    rope_base: ct.Constant[float],
    tile_size: ct.Constant[int]
):
    """
    RoPE (NeoX-style) kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: rope_neox

    NeoX-style interleaved rotary position embeddings.
    Pairs are (d, d+1) instead of (d, d+half).
    """
    batch_head_idx = ct.bid(0)
    seq_idx = ct.bid(1)
    tid = ct.tid(0)

    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    pos = positions[batch_idx * seq_len + seq_idx]

    # Process pairs of consecutive dimensions
    for d in range(tid * 2, head_dim, tile_size * 2):
        if d + 1 >= head_dim:
            continue

        # Compute frequency for this pair
        freq = 1.0 / ct.pow(rope_base, ct.float32(d) / ct.float32(head_dim))
        angle = ct.float32(pos) * freq

        cos_val = ct.cos(angle)
        sin_val = ct.sin(angle)

        base_idx = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * head_dim

        # Get consecutive pair
        x0 = ${inputName}[base_idx + d]
        x1 = ${inputName}[base_idx + d + 1]

        # Apply rotation to consecutive elements
        ${outputName}[base_idx + d] = x0 * cos_val - x1 * sin_val
        ${outputName}[base_idx + d + 1] = x0 * sin_val + x1 * cos_val


def launch_${ir.name}(${inputName}, positions, ${outputName}=None):
    """Launch the NeoX-style RoPE kernel"""
    batch_size, num_heads, seq_len, head_dim = ${inputName}.shape

    if ${outputName} is None:
        ${outputName} = cupy.empty_like(${inputName})

    grid = (batch_size * num_heads, seq_len, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (
        ${inputName}.ravel(), ${outputName}.ravel(), positions.ravel(),
        batch_size, num_heads, seq_len, head_dim, ROPE_BASE, TILE_SIZE
    ))
    return ${outputName}`;
}

/**
 * Generate cached RoPE kernel (precomputed sin/cos)
 */
export function generateRoPECached(ir: RoPETemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 64;
  const headDim = ir.ropeConfig?.headDim || 64;
  const maxSeqLen = ir.ropeConfig?.maxSeqLen || 4096;
  const base = ir.ropeConfig?.base || 10000;

  const inputName = ir.loads[0]?.source || 'x';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy
import math

TILE_SIZE = ${tileSize}
HEAD_DIM = ${headDim}
MAX_SEQ_LEN = ${maxSeqLen}
ROPE_BASE = ${base}

# Precompute sin/cos cache
def precompute_rope_cache(max_seq_len, head_dim, base=${base}):
    """Precompute sin/cos values for RoPE"""
    positions = cupy.arange(max_seq_len, dtype=cupy.float32)
    dim_indices = cupy.arange(0, head_dim, 2, dtype=cupy.float32)
    freqs = 1.0 / cupy.power(base, dim_indices / head_dim)

    # Outer product: [max_seq_len, head_dim/2]
    angles = cupy.outer(positions, freqs)

    # [max_seq_len, head_dim/2, 2] -> reshape to [max_seq_len, head_dim]
    cos_cache = cupy.cos(angles)
    sin_cache = cupy.sin(angles)

    return cos_cache, sin_cache


@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName}, cos_cache, sin_cache, positions,
    batch_size: ct.Constant[int],
    num_heads: ct.Constant[int],
    seq_len: ct.Constant[int],
    head_dim: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    RoPE (Cached) kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: rope_cached

    Uses precomputed sin/cos for efficiency.
    """
    batch_head_idx = ct.bid(0)
    seq_idx = ct.bid(1)
    tid = ct.tid(0)

    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    pos = positions[batch_idx * seq_len + seq_idx]
    half_dim = head_dim // 2

    for d in range(tid, half_dim, tile_size):
        # Load precomputed sin/cos
        cos_val = cos_cache[pos * half_dim + d]
        sin_val = sin_cache[pos * half_dim + d]

        base_idx = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * head_dim

        # Get the pair of values
        x0 = ${inputName}[base_idx + d]
        x1 = ${inputName}[base_idx + d + half_dim]

        # Apply cached rotation
        ${outputName}[base_idx + d] = x0 * cos_val - x1 * sin_val
        ${outputName}[base_idx + d + half_dim] = x0 * sin_val + x1 * cos_val


class RoPECache:
    """Cache manager for RoPE sin/cos values"""

    def __init__(self, max_seq_len=${maxSeqLen}, head_dim=${headDim}, base=${base}):
        self.cos_cache, self.sin_cache = precompute_rope_cache(max_seq_len, head_dim, base)

    def apply(self, ${inputName}, positions, ${outputName}=None):
        batch_size, num_heads, seq_len, head_dim = ${inputName}.shape

        if ${outputName} is None:
            ${outputName} = cupy.empty_like(${inputName})

        grid = (batch_size * num_heads, seq_len, 1)
        stream = cupy.cuda.get_current_stream()
        ct.launch(stream, grid, ${ir.name}, (
            ${inputName}.ravel(), ${outputName}.ravel(),
            self.cos_cache.ravel(), self.sin_cache.ravel(),
            positions.ravel(),
            batch_size, num_heads, seq_len, head_dim, TILE_SIZE
        ))
        return ${outputName}


def launch_${ir.name}(${inputName}, positions, cos_cache, sin_cache, ${outputName}=None):
    """Launch the cached RoPE kernel"""
    batch_size, num_heads, seq_len, head_dim = ${inputName}.shape

    if ${outputName} is None:
        ${outputName} = cupy.empty_like(${inputName})

    grid = (batch_size * num_heads, seq_len, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (
        ${inputName}.ravel(), ${outputName}.ravel(),
        cos_cache.ravel(), sin_cache.ravel(),
        positions.ravel(),
        batch_size, num_heads, seq_len, head_dim, TILE_SIZE
    ))
    return ${outputName}`;
}

/**
 * Get the appropriate RoPE template generator based on variant
 */
export function getRoPEGenerator(variant?: string): (ir: RoPETemplateIR) => string {
  switch (variant) {
    case 'rope_standard':
      return generateRoPEStandard;
    case 'rope_neox':
      return generateRoPENeox;
    case 'rope_cached':
      return generateRoPECached;
    default:
      return generateRoPEStandard;
  }
}
