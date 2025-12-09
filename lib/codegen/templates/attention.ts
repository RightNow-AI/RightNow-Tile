/**
 * Attention Pattern Templates
 * Specialized code generation for attention variants including Flash Attention
 */

import { EnhancedKernelIR } from '../../ir/types';
import { AttentionIR } from '../../ir/builder';

/**
 * Generate Flash Attention kernel using online softmax
 * Memory-efficient attention with O(N) memory instead of O(N^2)
 */
export function generateFlashAttention(ir: EnhancedKernelIR & Partial<AttentionIR>): string {
  const blockQ = ir.tileConfig.blockM || 64;
  const blockKV = ir.tileConfig.blockN || 64;
  const headDim = ir.attentionConfig?.headDim || 64;
  const scale = ir.attentionConfig?.softmaxScale || (1.0 / Math.sqrt(headDim));
  const useCausal = ir.attentionConfig?.useCausalMask || false;

  // Extract array names from loads or use defaults
  const qName = ir.loads[0]?.source || 'Q';
  const kName = ir.loads[1]?.source || 'K';
  const vName = ir.loads[2]?.source || 'V';
  const oName = ir.stores[0]?.target || 'O';

  return `import cuda_tile as ct
import cupy
import math

# Constants
BLOCK_Q = ${blockQ}
BLOCK_KV = ${blockKV}
HEAD_DIM = ${headDim}
SCALE = ${scale.toFixed(6)}

@ct.kernel
def ${ir.name}(
    ${qName}, ${kName}, ${vName}, ${oName},
    seq_len_q: ct.Constant[int],
    seq_len_kv: ct.Constant[int],
    num_heads: ct.Constant[int],
    head_dim: ct.Constant[int]
):
    """
    Flash Attention kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: flash_attention

    Uses online softmax for memory efficiency:
    - O(N) memory instead of O(N^2)
    - Numerically stable softmax computation
    - Block-wise processing of K/V
    """
    # Block and head indices
    block_q = ct.bid(0)
    head_idx = ct.bid(1)
    tid = ct.tid(0)

    # Initialize output accumulator and softmax statistics
    acc = ct.zeros((BLOCK_Q, HEAD_DIM), dtype=ct.float32)
    m_i = ct.full((BLOCK_Q,), float('-inf'), dtype=ct.float32)  # Running max
    l_i = ct.zeros((BLOCK_Q,), dtype=ct.float32)  # Running sum of exp

    # Calculate Q start position
    q_start = block_q * BLOCK_Q

    # Load Q tile (stays in registers throughout)
    q_tile = ct.load(
        ${qName},
        index=(head_idx, q_start),
        shape=(BLOCK_Q, HEAD_DIM),
        mask=q_start + ct.arange(BLOCK_Q)[:, None] < seq_len_q
    )

    # Determine K/V block range (for causal: only up to current Q position)
    ${useCausal ? `num_kv_blocks = ct.cdiv(ct.minimum(seq_len_kv, q_start + BLOCK_Q), BLOCK_KV)` : `num_kv_blocks = ct.cdiv(seq_len_kv, BLOCK_KV)`}

    # Iterate over K/V blocks
    for block_kv in range(num_kv_blocks):
        kv_start = block_kv * BLOCK_KV

        # Load K and V tiles
        k_tile = ct.load(
            ${kName},
            index=(head_idx, kv_start),
            shape=(BLOCK_KV, HEAD_DIM),
            mask=kv_start + ct.arange(BLOCK_KV)[:, None] < seq_len_kv
        )
        v_tile = ct.load(
            ${vName},
            index=(head_idx, kv_start),
            shape=(BLOCK_KV, HEAD_DIM),
            mask=kv_start + ct.arange(BLOCK_KV)[:, None] < seq_len_kv
        )

        # Compute QK^T: (BLOCK_Q, HEAD_DIM) @ (HEAD_DIM, BLOCK_KV) -> (BLOCK_Q, BLOCK_KV)
        qk = ct.tile_matmul(q_tile, ct.transpose(k_tile)) * SCALE

        ${useCausal ? `
        # Apply causal mask: mask out positions where q_pos < k_pos
        causal_mask = (q_start + ct.arange(BLOCK_Q)[:, None]) >= (kv_start + ct.arange(BLOCK_KV)[None, :])
        qk = ct.where(causal_mask, qk, float('-inf'))
        ` : ''}

        # Online softmax: compute new max
        m_ij = ct.reduce(qk, op=ct.max, axis=1)
        m_new = ct.maximum(m_i, m_ij)

        # Rescale previous accumulator
        alpha = ct.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # Compute attention weights with numerical stability
        p = ct.exp(qk - m_new[:, None])
        l_ij = ct.reduce(p, op=ct.sum, axis=1)
        l_i = l_i + l_ij

        # Accumulate weighted values: P @ V
        acc = acc + ct.tile_matmul(p, v_tile)

        # Update running max
        m_i = m_new

    # Normalize output by sum of exponentials
    out = acc / l_i[:, None]

    # Store result
    ct.store(
        ${oName},
        index=(head_idx, q_start),
        tile=out,
        mask=q_start + ct.arange(BLOCK_Q)[:, None] < seq_len_q
    )


def launch_${ir.name}(${qName}, ${kName}, ${vName}, ${oName}):
    """Launch the ${ir.name} Flash Attention kernel"""
    batch_heads, seq_len_q, head_dim = ${qName}.shape
    _, seq_len_kv, _ = ${kName}.shape
    num_heads = batch_heads  # Assuming batch*heads flattened

    # Grid: (num_q_blocks, num_heads)
    num_q_blocks = (seq_len_q + BLOCK_Q - 1) // BLOCK_Q
    grid = (num_q_blocks, num_heads, 1)

    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${qName}, ${kName}, ${vName}, ${oName}, seq_len_q, seq_len_kv, num_heads, head_dim))`;
}

/**
 * Generate standard attention (non-Flash variant)
 * Simpler but uses O(N^2) memory for attention scores
 */
export function generateStandardAttention(ir: EnhancedKernelIR & Partial<AttentionIR>): string {
  const blockSize = ir.tileConfig.tileSize || 128;
  const headDim = ir.attentionConfig?.headDim || 64;
  const scale = ir.attentionConfig?.softmaxScale || (1.0 / Math.sqrt(headDim));

  const qName = ir.loads[0]?.source || 'Q';
  const kName = ir.loads[1]?.source || 'K';
  const vName = ir.loads[2]?.source || 'V';
  const oName = ir.stores[0]?.target || 'O';

  return `import cuda_tile as ct
import cupy
import math

BLOCK_SIZE = ${blockSize}
HEAD_DIM = ${headDim}
SCALE = ${scale.toFixed(6)}

@ct.kernel
def ${ir.name}(
    ${qName}, ${kName}, ${vName}, ${oName},
    seq_len: ct.Constant[int],
    num_heads: ct.Constant[int],
    head_dim: ct.Constant[int]
):
    """
    Standard Attention kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: standard_attention

    Standard attention computation:
    1. Compute QK^T
    2. Apply softmax
    3. Multiply by V
    """
    # Thread indices
    pid_q = ct.bid(0)
    head_idx = ct.bid(1)
    tid = ct.tid(0)

    # Load Q for this block
    q_tile = ct.load(${qName}, index=(head_idx, pid_q), shape=(1, HEAD_DIM))

    # Compute attention scores for all K positions
    scores = ct.zeros((seq_len,), dtype=ct.float32)

    for k_block in range(ct.cdiv(seq_len, BLOCK_SIZE)):
        k_start = k_block * BLOCK_SIZE
        k_tile = ct.load(${kName}, index=(head_idx, k_start), shape=(BLOCK_SIZE, HEAD_DIM))

        # QK^T for this block
        block_scores = ct.tile_matmul(q_tile, ct.transpose(k_tile)) * SCALE

        # Store scores
        for i in range(BLOCK_SIZE):
            if k_start + i < seq_len:
                scores[k_start + i] = block_scores[0, i]

    # Softmax over all scores
    max_score = ct.reduce(scores, op=ct.max)
    exp_scores = ct.exp(scores - max_score)
    sum_exp = ct.reduce(exp_scores, op=ct.sum)
    attn_weights = exp_scores / sum_exp

    # Compute weighted sum of V
    output = ct.zeros((HEAD_DIM,), dtype=ct.float32)

    for v_block in range(ct.cdiv(seq_len, BLOCK_SIZE)):
        v_start = v_block * BLOCK_SIZE
        v_tile = ct.load(${vName}, index=(head_idx, v_start), shape=(BLOCK_SIZE, HEAD_DIM))

        for i in range(BLOCK_SIZE):
            if v_start + i < seq_len:
                output = output + attn_weights[v_start + i] * v_tile[i]

    # Store output
    ct.store(${oName}, index=(head_idx, pid_q), tile=output)


def launch_${ir.name}(${qName}, ${kName}, ${vName}, ${oName}):
    """Launch the ${ir.name} standard attention kernel"""
    num_heads, seq_len, head_dim = ${qName}.shape
    grid = (seq_len, num_heads, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${qName}, ${kName}, ${vName}, ${oName}, seq_len, num_heads, head_dim))`;
}

/**
 * Generate Multi-Head Attention kernel
 * Handles multiple attention heads in parallel
 */
export function generateMultiHeadAttention(ir: EnhancedKernelIR & Partial<AttentionIR>): string {
  const blockQ = ir.tileConfig.blockM || 64;
  const blockKV = ir.tileConfig.blockN || 64;
  const headDim = ir.attentionConfig?.headDim || 64;
  const scale = ir.attentionConfig?.softmaxScale || (1.0 / Math.sqrt(headDim));

  const qName = ir.loads[0]?.source || 'Q';
  const kName = ir.loads[1]?.source || 'K';
  const vName = ir.loads[2]?.source || 'V';
  const oName = ir.stores[0]?.target || 'O';

  return `import cuda_tile as ct
import cupy
import math

BLOCK_Q = ${blockQ}
BLOCK_KV = ${blockKV}
HEAD_DIM = ${headDim}
SCALE = ${scale.toFixed(6)}

@ct.kernel
def ${ir.name}(
    ${qName}, ${kName}, ${vName}, ${oName},
    batch_size: ct.Constant[int],
    seq_len: ct.Constant[int],
    num_heads: ct.Constant[int],
    head_dim: ct.Constant[int]
):
    """
    Multi-Head Attention kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: multi_head_attention

    Processes multiple attention heads in parallel.
    Uses blocked algorithm for memory efficiency.
    """
    # Block indices: (batch, head, q_block)
    batch_idx = ct.bid(0)
    head_idx = ct.bid(1)
    block_q = ct.bid(2)

    # Output accumulator with online softmax stats
    acc = ct.zeros((BLOCK_Q, HEAD_DIM), dtype=ct.float32)
    m_i = ct.full((BLOCK_Q,), float('-inf'), dtype=ct.float32)
    l_i = ct.zeros((BLOCK_Q,), dtype=ct.float32)

    q_start = block_q * BLOCK_Q

    # Load Q tile for this block
    q_tile = ct.load(
        ${qName},
        index=(batch_idx, head_idx, q_start),
        shape=(BLOCK_Q, HEAD_DIM),
        mask=q_start + ct.arange(BLOCK_Q)[:, None] < seq_len
    )

    # Process K/V blocks
    num_kv_blocks = ct.cdiv(seq_len, BLOCK_KV)

    for block_kv in range(num_kv_blocks):
        kv_start = block_kv * BLOCK_KV

        k_tile = ct.load(
            ${kName},
            index=(batch_idx, head_idx, kv_start),
            shape=(BLOCK_KV, HEAD_DIM),
            mask=kv_start + ct.arange(BLOCK_KV)[:, None] < seq_len
        )
        v_tile = ct.load(
            ${vName},
            index=(batch_idx, head_idx, kv_start),
            shape=(BLOCK_KV, HEAD_DIM),
            mask=kv_start + ct.arange(BLOCK_KV)[:, None] < seq_len
        )

        # QK^T with scaling
        qk = ct.tile_matmul(q_tile, ct.transpose(k_tile)) * SCALE

        # Online softmax update
        m_ij = ct.reduce(qk, op=ct.max, axis=1)
        m_new = ct.maximum(m_i, m_ij)

        alpha = ct.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        p = ct.exp(qk - m_new[:, None])
        l_ij = ct.reduce(p, op=ct.sum, axis=1)
        l_i = l_i + l_ij

        acc = acc + ct.tile_matmul(p, v_tile)
        m_i = m_new

    # Normalize and store
    out = acc / l_i[:, None]
    ct.store(
        ${oName},
        index=(batch_idx, head_idx, q_start),
        tile=out,
        mask=q_start + ct.arange(BLOCK_Q)[:, None] < seq_len
    )


def launch_${ir.name}(${qName}, ${kName}, ${vName}, ${oName}):
    """Launch the ${ir.name} multi-head attention kernel"""
    batch_size, num_heads, seq_len, head_dim = ${qName}.shape
    num_q_blocks = (seq_len + BLOCK_Q - 1) // BLOCK_Q
    grid = (batch_size, num_heads, num_q_blocks)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${qName}, ${kName}, ${vName}, ${oName}, batch_size, seq_len, num_heads, head_dim))`;
}

/**
 * Generate Causal Attention kernel
 * Attention with causal mask for autoregressive models
 */
export function generateCausalAttention(ir: EnhancedKernelIR & Partial<AttentionIR>): string {
  // Causal attention is Flash Attention with causal mask enabled
  const modifiedIR = {
    ...ir,
    attentionConfig: {
      ...ir.attentionConfig,
      useCausalMask: true,
    },
  };

  return generateFlashAttention(modifiedIR as EnhancedKernelIR & Partial<AttentionIR>);
}

/**
 * Get the appropriate attention template generator based on variant
 */
export function getAttentionGenerator(variant?: string): (ir: EnhancedKernelIR & Partial<AttentionIR>) => string {
  switch (variant) {
    case 'flash_attention':
    case 'flash_attention_v2':
      return generateFlashAttention;
    case 'multi_head_attention':
      return generateMultiHeadAttention;
    case 'causal_attention':
      return generateCausalAttention;
    case 'cross_attention':
      return generateStandardAttention; // Cross-attention uses standard structure
    default:
      // Default to Flash Attention for better memory efficiency
      return generateFlashAttention;
  }
}
