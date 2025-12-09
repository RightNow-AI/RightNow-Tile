/**
 * KV Cache Pattern Templates
 * Specialized code generation for key-value caching in LLM inference
 */

import { EnhancedKernelIR } from '../../ir/types';

/**
 * KV Cache IR extension
 */
export interface KVCacheTemplateIR extends EnhancedKernelIR {
  kvConfig?: {
    isPaged: boolean;
    blockSize: number;
    numKVHeads: number;
    headDim: number;
    maxSeqLen: number;
    numLayers: number;
  };
}

/**
 * Generate standard KV cache append kernel
 */
export function generateKVCacheAppend(ir: KVCacheTemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 64;
  const headDim = ir.kvConfig?.headDim || 128;

  const keyName = ir.loads[0]?.source || 'key';
  const valueName = ir.loads[1]?.source || 'value';
  const keyCacheName = ir.stores[0]?.target || 'key_cache';
  const valueCacheName = ir.stores[1]?.target || 'value_cache';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
HEAD_DIM = ${headDim}

@ct.kernel
def ${ir.name}(
    ${keyName}, ${valueName},
    ${keyCacheName}, ${valueCacheName},
    positions,
    batch_size: ct.Constant[int],
    num_kv_heads: ct.Constant[int],
    seq_len: ct.Constant[int],
    head_dim: ct.Constant[int],
    max_seq_len: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    KV Cache Append kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: kvcache_append

    Appends new K/V tokens to the cache.
    """
    batch_head_idx = ct.bid(0)
    seq_idx = ct.bid(1)
    tid = ct.tid(0)

    batch_idx = batch_head_idx // num_kv_heads
    head_idx = batch_head_idx % num_kv_heads

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Get cache position for this token
    cache_pos = positions[batch_idx * seq_len + seq_idx]

    # Source index in the input
    src_idx = ((batch_idx * num_kv_heads + head_idx) * seq_len + seq_idx) * head_dim

    # Destination index in the cache
    dst_idx = ((batch_idx * num_kv_heads + head_idx) * max_seq_len + cache_pos) * head_dim

    # Copy key and value vectors
    for d in range(tid, head_dim, tile_size):
        ${keyCacheName}[dst_idx + d] = ${keyName}[src_idx + d]
        ${valueCacheName}[dst_idx + d] = ${valueName}[src_idx + d]


def launch_${ir.name}(${keyName}, ${valueName}, ${keyCacheName}, ${valueCacheName}, positions):
    """Launch the KV cache append kernel"""
    batch_size, num_kv_heads, seq_len, head_dim = ${keyName}.shape

    max_seq_len = ${keyCacheName}.shape[2]

    grid = (batch_size * num_kv_heads, seq_len, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (
        ${keyName}.ravel(), ${valueName}.ravel(),
        ${keyCacheName}.ravel(), ${valueCacheName}.ravel(),
        positions.ravel(),
        batch_size, num_kv_heads, seq_len, head_dim, max_seq_len, TILE_SIZE
    ))`;
}

/**
 * Generate paged attention KV cache kernel
 */
export function generateKVCachePaged(ir: KVCacheTemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 64;
  const blockSize = ir.kvConfig?.blockSize || 16;
  const headDim = ir.kvConfig?.headDim || 128;

  const keyName = ir.loads[0]?.source || 'key';
  const valueName = ir.loads[1]?.source || 'value';
  const keyCacheName = ir.stores[0]?.target || 'key_cache';
  const valueCacheName = ir.stores[1]?.target || 'value_cache';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
BLOCK_SIZE = ${blockSize}
HEAD_DIM = ${headDim}

@ct.kernel
def ${ir.name}(
    ${keyName}, ${valueName},
    ${keyCacheName}, ${valueCacheName},
    block_tables, slot_mapping,
    batch_size: ct.Constant[int],
    num_kv_heads: ct.Constant[int],
    seq_len: ct.Constant[int],
    head_dim: ct.Constant[int],
    block_size: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Paged KV Cache kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: kvcache_paged

    Paged attention-style KV caching with block tables.
    """
    batch_head_idx = ct.bid(0)
    seq_idx = ct.bid(1)
    tid = ct.tid(0)

    batch_idx = batch_head_idx // num_kv_heads
    head_idx = batch_head_idx % num_kv_heads

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Get the slot (physical location) for this token
    slot = slot_mapping[batch_idx * seq_len + seq_idx]

    # Decode block and offset from slot
    block_idx = slot // block_size
    block_offset = slot % block_size

    # Source index
    src_idx = ((batch_idx * num_kv_heads + head_idx) * seq_len + seq_idx) * head_dim

    # Destination: cache[head_idx, block_idx, block_offset, :]
    dst_base = (head_idx * (slot // block_size + 1) + block_idx) * block_size * head_dim
    dst_idx = dst_base + block_offset * head_dim

    # Copy to cache block
    for d in range(tid, head_dim, tile_size):
        ${keyCacheName}[dst_idx + d] = ${keyName}[src_idx + d]
        ${valueCacheName}[dst_idx + d] = ${valueName}[src_idx + d]


@ct.kernel
def ${ir.name}_fetch(
    ${keyCacheName}, ${valueCacheName},
    key_out, value_out,
    block_tables, context_lens,
    batch_size: ct.Constant[int],
    num_kv_heads: ct.Constant[int],
    max_context_len: ct.Constant[int],
    head_dim: ct.Constant[int],
    block_size: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Fetch from paged KV cache for attention
    """
    batch_head_idx = ct.bid(0)
    pos_idx = ct.bid(1)
    tid = ct.tid(0)

    batch_idx = batch_head_idx // num_kv_heads
    head_idx = batch_head_idx % num_kv_heads

    context_len = context_lens[batch_idx]
    if batch_idx >= batch_size or pos_idx >= context_len:
        return

    # Get physical block for this position
    block_idx = pos_idx // block_size
    block_offset = pos_idx % block_size

    physical_block = block_tables[batch_idx * max_context_len // block_size + block_idx]

    # Source in cache
    src_base = (head_idx * physical_block + physical_block) * block_size * head_dim
    src_idx = src_base + block_offset * head_dim

    # Destination
    dst_idx = ((batch_idx * num_kv_heads + head_idx) * max_context_len + pos_idx) * head_dim

    for d in range(tid, head_dim, tile_size):
        key_out[dst_idx + d] = ${keyCacheName}[src_idx + d]
        value_out[dst_idx + d] = ${valueCacheName}[src_idx + d]


class PagedKVCache:
    """Manager for paged KV cache"""

    def __init__(self, num_blocks, block_size=${blockSize}, num_kv_heads=32, head_dim=${headDim}):
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Pre-allocate cache blocks
        self.key_cache = cupy.zeros((num_kv_heads, num_blocks, block_size, head_dim), dtype=cupy.float16)
        self.value_cache = cupy.zeros((num_kv_heads, num_blocks, block_size, head_dim), dtype=cupy.float16)

        # Block allocation tracking
        self.free_blocks = list(range(num_blocks))
        self.block_tables = {}  # seq_id -> list of block indices

    def allocate_blocks(self, seq_id, num_tokens):
        """Allocate blocks for a sequence"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("Out of cache blocks")

        blocks = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        self.block_tables[seq_id] = blocks
        return blocks

    def free_blocks_for_seq(self, seq_id):
        """Free blocks for a completed sequence"""
        if seq_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[seq_id])
            del self.block_tables[seq_id]


def launch_${ir.name}(${keyName}, ${valueName}, ${keyCacheName}, ${valueCacheName}, block_tables, slot_mapping):
    """Launch the paged KV cache kernel"""
    batch_size, num_kv_heads, seq_len, head_dim = ${keyName}.shape

    grid = (batch_size * num_kv_heads, seq_len, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (
        ${keyName}.ravel(), ${valueName}.ravel(),
        ${keyCacheName}.ravel(), ${valueCacheName}.ravel(),
        block_tables.ravel(), slot_mapping.ravel(),
        batch_size, num_kv_heads, seq_len, head_dim, BLOCK_SIZE, TILE_SIZE
    ))`;
}

/**
 * Generate prefix caching KV cache kernel
 */
export function generateKVCachePrefix(ir: KVCacheTemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 64;
  const headDim = ir.kvConfig?.headDim || 128;

  const keyCacheName = ir.loads[0]?.source || 'key_cache';
  const valueCacheName = ir.loads[1]?.source || 'value_cache';
  const keyOutName = ir.stores[0]?.target || 'key_out';
  const valueOutName = ir.stores[1]?.target || 'value_out';

  return `import cuda_tile as ct
import cupy
import hashlib

TILE_SIZE = ${tileSize}
HEAD_DIM = ${headDim}

@ct.kernel
def ${ir.name}(
    ${keyCacheName}, ${valueCacheName},
    ${keyOutName}, ${valueOutName},
    prefix_mapping, prefix_lens,
    batch_size: ct.Constant[int],
    num_kv_heads: ct.Constant[int],
    max_prefix_len: ct.Constant[int],
    head_dim: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Prefix KV Cache kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: kvcache_prefix

    Reuses cached KV for common prefixes (system prompts, etc.)
    """
    batch_head_idx = ct.bid(0)
    pos_idx = ct.bid(1)
    tid = ct.tid(0)

    batch_idx = batch_head_idx // num_kv_heads
    head_idx = batch_head_idx % num_kv_heads

    prefix_len = prefix_lens[batch_idx]
    if batch_idx >= batch_size or pos_idx >= prefix_len:
        return

    # Get the prefix cache index for this batch
    prefix_idx = prefix_mapping[batch_idx]

    # Source: prefix cache
    src_idx = ((prefix_idx * num_kv_heads + head_idx) * max_prefix_len + pos_idx) * head_dim

    # Destination: output
    dst_idx = ((batch_idx * num_kv_heads + head_idx) * max_prefix_len + pos_idx) * head_dim

    # Copy from prefix cache
    for d in range(tid, head_dim, tile_size):
        ${keyOutName}[dst_idx + d] = ${keyCacheName}[src_idx + d]
        ${valueOutName}[dst_idx + d] = ${valueCacheName}[src_idx + d]


class PrefixCache:
    """Manager for prefix KV caching"""

    def __init__(self, max_prefixes=100, max_prefix_len=1024, num_kv_heads=32, head_dim=${headDim}):
        self.max_prefix_len = max_prefix_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Cache storage
        self.key_cache = cupy.zeros((max_prefixes, num_kv_heads, max_prefix_len, head_dim), dtype=cupy.float16)
        self.value_cache = cupy.zeros((max_prefixes, num_kv_heads, max_prefix_len, head_dim), dtype=cupy.float16)

        # Prefix hash -> cache index
        self.prefix_to_idx = {}
        self.next_idx = 0

    def get_prefix_hash(self, tokens):
        """Compute hash for a token sequence"""
        return hashlib.md5(tokens.tobytes()).hexdigest()

    def lookup(self, tokens):
        """Look up cached prefix"""
        h = self.get_prefix_hash(tokens)
        return self.prefix_to_idx.get(h)

    def store(self, tokens, key, value):
        """Store a prefix in cache"""
        h = self.get_prefix_hash(tokens)
        if h in self.prefix_to_idx:
            return self.prefix_to_idx[h]

        idx = self.next_idx
        self.next_idx += 1

        seq_len = key.shape[2]
        self.key_cache[idx, :, :seq_len, :] = key
        self.value_cache[idx, :, :seq_len, :] = value
        self.prefix_to_idx[h] = idx

        return idx


def launch_${ir.name}(${keyCacheName}, ${valueCacheName}, ${keyOutName}, ${valueOutName}, prefix_mapping, prefix_lens):
    """Launch the prefix KV cache kernel"""
    batch_size = prefix_mapping.shape[0]
    num_kv_heads = ${keyCacheName}.shape[1]
    max_prefix_len = ${keyCacheName}.shape[2]
    head_dim = ${keyCacheName}.shape[3]

    grid = (batch_size * num_kv_heads, max_prefix_len, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (
        ${keyCacheName}.ravel(), ${valueCacheName}.ravel(),
        ${keyOutName}.ravel(), ${valueOutName}.ravel(),
        prefix_mapping.ravel(), prefix_lens.ravel(),
        batch_size, num_kv_heads, max_prefix_len, head_dim, TILE_SIZE
    ))`;
}

/**
 * Generate GQA (Grouped Query Attention) KV cache kernel
 */
export function generateKVCacheGQA(ir: KVCacheTemplateIR): string {
  const tileSize = ir.tileConfig.tileSize || 64;
  const headDim = ir.kvConfig?.headDim || 128;

  const keyName = ir.loads[0]?.source || 'key';
  const valueName = ir.loads[1]?.source || 'value';
  const keyCacheName = ir.stores[0]?.target || 'key_cache';
  const valueCacheName = ir.stores[1]?.target || 'value_cache';

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}
HEAD_DIM = ${headDim}

@ct.kernel
def ${ir.name}(
    ${keyName}, ${valueName},
    ${keyCacheName}, ${valueCacheName},
    positions,
    batch_size: ct.Constant[int],
    num_query_heads: ct.Constant[int],
    num_kv_heads: ct.Constant[int],
    seq_len: ct.Constant[int],
    head_dim: ct.Constant[int],
    max_seq_len: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    GQA KV Cache kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: kvcache_gqa

    Grouped Query Attention: num_query_heads > num_kv_heads.
    Multiple query heads share the same KV heads.
    """
    batch_kv_idx = ct.bid(0)
    seq_idx = ct.bid(1)
    tid = ct.tid(0)

    batch_idx = batch_kv_idx // num_kv_heads
    kv_head_idx = batch_kv_idx % num_kv_heads

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Get cache position
    cache_pos = positions[batch_idx * seq_len + seq_idx]

    # Source index (KV heads)
    src_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + seq_idx) * head_dim

    # Destination in cache
    dst_idx = ((batch_idx * num_kv_heads + kv_head_idx) * max_seq_len + cache_pos) * head_dim

    # Copy key and value
    for d in range(tid, head_dim, tile_size):
        ${keyCacheName}[dst_idx + d] = ${keyName}[src_idx + d]
        ${valueCacheName}[dst_idx + d] = ${valueName}[src_idx + d]


@ct.kernel
def ${ir.name}_expand(
    ${keyCacheName}, ${valueCacheName},
    key_expanded, value_expanded,
    positions, context_lens,
    batch_size: ct.Constant[int],
    num_query_heads: ct.Constant[int],
    num_kv_heads: ct.Constant[int],
    max_context_len: ct.Constant[int],
    head_dim: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Expand KV cache for GQA during attention
    Each KV head is broadcast to multiple query heads
    """
    batch_query_idx = ct.bid(0)
    pos_idx = ct.bid(1)
    tid = ct.tid(0)

    batch_idx = batch_query_idx // num_query_heads
    query_head_idx = batch_query_idx % num_query_heads

    context_len = context_lens[batch_idx]
    if batch_idx >= batch_size or pos_idx >= context_len:
        return

    # Map query head to KV head
    heads_per_group = num_query_heads // num_kv_heads
    kv_head_idx = query_head_idx // heads_per_group

    cache_pos = positions[batch_idx * max_context_len + pos_idx]

    # Source from KV cache
    src_idx = ((batch_idx * num_kv_heads + kv_head_idx) * max_context_len + cache_pos) * head_dim

    # Destination (expanded for query heads)
    dst_idx = ((batch_idx * num_query_heads + query_head_idx) * max_context_len + pos_idx) * head_dim

    for d in range(tid, head_dim, tile_size):
        key_expanded[dst_idx + d] = ${keyCacheName}[src_idx + d]
        value_expanded[dst_idx + d] = ${valueCacheName}[src_idx + d]


def launch_${ir.name}(${keyName}, ${valueName}, ${keyCacheName}, ${valueCacheName}, positions, num_query_heads):
    """Launch the GQA KV cache kernel"""
    batch_size, num_kv_heads, seq_len, head_dim = ${keyName}.shape
    max_seq_len = ${keyCacheName}.shape[2]

    grid = (batch_size * num_kv_heads, seq_len, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (
        ${keyName}.ravel(), ${valueName}.ravel(),
        ${keyCacheName}.ravel(), ${valueCacheName}.ravel(),
        positions.ravel(),
        batch_size, num_query_heads, num_kv_heads, seq_len, head_dim, max_seq_len, TILE_SIZE
    ))`;
}

/**
 * Get the appropriate KV cache template generator based on variant
 */
export function getKVCacheGenerator(variant?: string): (ir: KVCacheTemplateIR) => string {
  switch (variant) {
    case 'kvcache_append':
      return generateKVCacheAppend;
    case 'kvcache_paged':
      return generateKVCachePaged;
    case 'kvcache_prefix':
      return generateKVCachePrefix;
    case 'kvcache_gqa':
      return generateKVCacheGQA;
    default:
      return generateKVCacheAppend;
  }
}
