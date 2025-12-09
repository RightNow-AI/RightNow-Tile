/**
 * Embedding Pattern Templates
 * Specialized code generation for embedding lookup, embedding bag, and positional embeddings
 */

import { EnhancedKernelIR } from '../../ir/types';

/**
 * Embedding IR extension
 */
export interface EmbeddingTemplateIR extends EnhancedKernelIR {
  embeddingConfig?: {
    vocabSize: number;
    embeddingDim: number;
    paddingIdx?: number;
    mode?: 'sum' | 'mean' | 'max';  // for embedding bag
  };
}

/**
 * Generate standard embedding lookup kernel
 */
export function generateEmbeddingLookup(ir: EmbeddingTemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 128;
  const embeddingDim = ir.embeddingConfig?.embeddingDim || 512;

  const weightsName = ir.loads[0]?.source || 'weight';
  const indicesName = ir.loads[1]?.source || 'indices';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
EMBEDDING_DIM = ${embeddingDim}

@ct.kernel
def ${ir.name}(
    ${weightsName}, ${indicesName}, ${outputName},
    num_indices: ct.Constant[int],
    vocab_size: ct.Constant[int],
    embedding_dim: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Embedding Lookup kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: embedding_lookup

    Looks up embeddings by index.
    """
    pid = ct.bid(0)
    tid = ct.tid(0)
    idx = pid * tile_size + tid

    if idx >= num_indices:
        return

    # Get the embedding index
    embed_idx = ${indicesName}[idx]

    # Bounds check
    if embed_idx < 0 or embed_idx >= vocab_size:
        # Fill with zeros for out-of-bounds
        for d in range(embedding_dim):
            ${outputName}[idx * embedding_dim + d] = 0.0
        return

    # Copy the embedding vector
    for d in range(embedding_dim):
        ${outputName}[idx * embedding_dim + d] = ${weightsName}[embed_idx * embedding_dim + d]


def launch_${ir.name}(${weightsName}, ${indicesName}, ${outputName}=None):
    """Launch the embedding lookup kernel"""
    num_indices = ${indicesName}.shape[0]
    vocab_size, embedding_dim = ${weightsName}.shape

    if ${outputName} is None:
        ${outputName} = cupy.empty((num_indices, embedding_dim), dtype=${weightsName}.dtype)

    grid = (ct.cdiv(num_indices, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${weightsName}, ${indicesName}, ${outputName}, num_indices, vocab_size, embedding_dim, TILE_SIZE))
    return ${outputName}`;
}

/**
 * Generate embedding bag kernel (sum/mean of multiple embeddings)
 */
export function generateEmbeddingBag(ir: EmbeddingTemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 128;
  const mode = ir.embeddingConfig?.mode || 'sum';

  const weightsName = ir.loads[0]?.source || 'weight';
  const indicesName = ir.loads[1]?.source || 'indices';
  const offsetsName = ir.loads[2]?.source || 'offsets';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(
    ${weightsName}, ${indicesName}, ${offsetsName}, ${outputName},
    num_bags: ct.Constant[int],
    num_indices: ct.Constant[int],
    vocab_size: ct.Constant[int],
    embedding_dim: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Embedding Bag kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: embedding_bag

    Aggregates multiple embeddings per bag using ${mode}.
    """
    bid = ct.bid(0)
    tid = ct.tid(0)
    bag_idx = bid

    if bag_idx >= num_bags:
        return

    # Get the range of indices for this bag
    start = ${offsetsName}[bag_idx]
    end = ${offsetsName}[bag_idx + 1] if bag_idx + 1 < num_bags else num_indices

    # Process embedding dimensions in chunks
    for d_base in range(0, embedding_dim, tile_size):
        d = d_base + tid
        if d >= embedding_dim:
            continue

        # Accumulate embeddings
        acc = ct.float32(0.0)
        count = 0

        for i in range(start, end):
            embed_idx = ${indicesName}[i]
            if embed_idx >= 0 and embed_idx < vocab_size:
                acc += ${weightsName}[embed_idx * embedding_dim + d]
                count += 1

        # Apply aggregation mode
        ${mode === 'mean' ? `
        if count > 0:
            acc = acc / ct.float32(count)
        ` : mode === 'max' ? `
        # For max mode, we need different logic
        acc = ct.float32('-inf')
        for i in range(start, end):
            embed_idx = ${indicesName}[i]
            if embed_idx >= 0 and embed_idx < vocab_size:
                val = ${weightsName}[embed_idx * embedding_dim + d]
                acc = ct.maximum(acc, val)
        if acc == ct.float32('-inf'):
            acc = ct.float32(0.0)
        ` : '# Sum mode - no additional processing'}

        ${outputName}[bag_idx * embedding_dim + d] = acc


def launch_${ir.name}(${weightsName}, ${indicesName}, ${offsetsName}, ${outputName}=None):
    """Launch the embedding bag kernel"""
    num_bags = ${offsetsName}.shape[0]
    num_indices = ${indicesName}.shape[0]
    vocab_size, embedding_dim = ${weightsName}.shape

    if ${outputName} is None:
        ${outputName} = cupy.zeros((num_bags, embedding_dim), dtype=${weightsName}.dtype)

    grid = (num_bags, 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${weightsName}, ${indicesName}, ${offsetsName}, ${outputName}, num_bags, num_indices, vocab_size, embedding_dim, TILE_SIZE))
    return ${outputName}`;
}

/**
 * Generate positional embedding kernel
 */
export function generatePositionalEmbedding(ir: EmbeddingTemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 128;
  const embeddingDim = ir.embeddingConfig?.embeddingDim || 512;

  const inputName = ir.loads[0]?.source || 'input';
  const posEmbedName = ir.loads[1]?.source || 'pos_embed';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy
import math

TILE_SIZE = ${tileSize}
EMBEDDING_DIM = ${embeddingDim}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${posEmbedName}, ${outputName},
    batch_size: ct.Constant[int],
    seq_len: ct.Constant[int],
    embedding_dim: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Positional Embedding kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: positional_embedding

    Adds positional embeddings to input.
    """
    bid = ct.bid(0)
    tid = ct.tid(0)

    # Calculate position
    total_tokens = batch_size * seq_len
    token_idx = bid * tile_size + tid

    if token_idx >= total_tokens:
        return

    batch_idx = token_idx // seq_len
    pos_idx = token_idx % seq_len

    # Add positional embedding to each dimension
    for d in range(embedding_dim):
        input_val = ${inputName}[token_idx * embedding_dim + d]
        pos_val = ${posEmbedName}[pos_idx * embedding_dim + d]
        ${outputName}[token_idx * embedding_dim + d] = input_val + pos_val


@ct.kernel
def ${ir.name}_sincos(
    ${outputName},
    seq_len: ct.Constant[int],
    embedding_dim: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Generate sinusoidal positional embeddings
    """
    tid = ct.tid(0)
    pos = ct.bid(0)

    if pos >= seq_len:
        return

    for d in range(tid, embedding_dim, tile_size):
        div_term = ct.exp(ct.float32(d // 2 * 2) * -(math.log(10000.0) / ct.float32(embedding_dim)))
        angle = ct.float32(pos) * div_term

        if d % 2 == 0:
            ${outputName}[pos * embedding_dim + d] = ct.sin(angle)
        else:
            ${outputName}[pos * embedding_dim + d] = ct.cos(angle)


def launch_${ir.name}(${inputName}, ${posEmbedName}, ${outputName}=None):
    """Launch the positional embedding kernel"""
    batch_size, seq_len, embedding_dim = ${inputName}.shape

    if ${outputName} is None:
        ${outputName} = cupy.empty_like(${inputName})

    total_tokens = batch_size * seq_len
    grid = (ct.cdiv(total_tokens, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}.ravel(), ${posEmbedName}.ravel(), ${outputName}.ravel(), batch_size, seq_len, embedding_dim, TILE_SIZE))
    return ${outputName}.reshape(batch_size, seq_len, embedding_dim)


def generate_sincos_embeddings(seq_len, embedding_dim):
    """Generate sinusoidal positional embeddings"""
    pos_embed = cupy.zeros((seq_len, embedding_dim), dtype=cupy.float32)
    grid = (seq_len, 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}_sincos, (pos_embed.ravel(), seq_len, embedding_dim, TILE_SIZE))
    return pos_embed`;
}

/**
 * Get the appropriate embedding template generator based on variant
 */
export function getEmbeddingGenerator(variant?: string): (ir: EmbeddingTemplateIR) => string {
  switch (variant) {
    case 'embedding_lookup':
      return generateEmbeddingLookup;
    case 'embedding_bag':
      return generateEmbeddingBag;
    case 'positional_embedding':
      return generatePositionalEmbedding;
    default:
      return generateEmbeddingLookup;
  }
}
