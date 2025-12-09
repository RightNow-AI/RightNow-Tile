/**
 * KV Cache Pattern Matcher
 * Detects key-value cache operations used in autoregressive transformer inference
 */

import { CudaKernelInfo, PatternMatch, Evidence, PatternVariant } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class KVCacheMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;

    // === Primary Indicators (high weight) ===

    // 1. Cache append operation (write new KV to position)
    const appendInfo = this.detectCacheAppend(source);
    if (appendInfo.found) {
      addEvidence(evidence, 'cache_append', 0.35,
        `Cache append: ${appendInfo.type}`);
    }

    // 2. Paged attention pattern
    const pagedInfo = this.detectPagedAttention(source);
    if (pagedInfo.found) {
      addEvidence(evidence, 'paged_attention', 0.35,
        'Paged attention pattern');
    }

    // 3. Prefix caching / cache reuse
    const prefixInfo = this.detectPrefixCaching(source);
    if (prefixInfo.found) {
      addEvidence(evidence, 'prefix_caching', 0.25,
        'Prefix cache reuse');
    }

    // === Secondary Indicators (medium weight) ===

    // 4. Sequence length / position tracking
    const hasSeqLen = /seq_?len|cur_?len|cache_?len|kv_?len|position/i.test(source);
    if (hasSeqLen) {
      addEvidence(evidence, 'seq_len_tracking', 0.15,
        'Sequence length tracking');
    }

    // 5. Key/Value cache arrays
    const kvParams = kernel.parameters.filter(p =>
      /k_?cache|v_?cache|key_?cache|value_?cache|past_?key|past_?value|kv_?buffer/i.test(p.name)
    );
    if (kvParams.length >= 1) {
      addEvidence(evidence, 'kv_cache_params', 0.20,
        `KV cache parameters: ${kvParams.map(p => p.name).join(', ')}`);
    }

    // 6. Cache copy/update operation
    const hasCacheCopy = /cache\s*\[\s*\w+\s*\]\s*=|cache\s*\+\s*offset|memcpy.*cache/i.test(source);
    if (hasCacheCopy) {
      addEvidence(evidence, 'cache_copy', 0.15,
        'Cache copy/update');
    }

    // 7. Block table / page table lookup
    const hasBlockTable = /block_?table|page_?table|slot_?mapping/i.test(source);
    if (hasBlockTable) {
      addEvidence(evidence, 'block_table', 0.20,
        'Block/page table lookup');
    }

    // 8. Name hints
    if (/kv_?cache|cache|paged|vllm|past/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests KV cache');
    }

    // 9. Maximum sequence length handling
    const hasMaxLen = /max_?seq|max_?len|max_?context/i.test(source);
    if (hasMaxLen) {
      addEvidence(evidence, 'max_len_check', 0.10,
        'Maximum sequence length handling');
    }

    // 10. Token position update
    const hasTokenPos = /token_?idx|cur_?pos|new_?tokens?/i.test(source);
    if (hasTokenPos && appendInfo.found) {
      addEvidence(evidence, 'token_pos_update', 0.10,
        'Token position update');
    }

    // 11. Multi-query attention (MQA) or Grouped-query attention (GQA) cache
    const hasMQAGQA = /num_?kv_?heads|kv_?head_?mapping|num_?groups/i.test(source);
    if (hasMQAGQA) {
      addEvidence(evidence, 'mqa_gqa', 0.15,
        'MQA/GQA cache pattern');
    }

    // 12. Rotary embedding with cache
    const hasRoPECache = /rope.*cache|rotary.*past|pos.*embed.*cache/i.test(source);
    if (hasRoPECache) {
      addEvidence(evidence, 'rope_cache', 0.10,
        'RoPE with cache');
    }

    // === Negative Indicators ===

    // Full attention computation (not just cache)
    if (/softmax|attention_?weights|qk.*transpose/i.test(source) && !appendInfo.found) {
      addEvidence(evidence, 'full_attention', -0.15,
        'Full attention (not pure cache operation)');
    }

    const match = createPatternMatch('kvcache' as any, evidence, warnings);

    if (match.confidence > 0.3) {
      match.variant = this.determineVariant(appendInfo, pagedInfo, prefixInfo, hasMQAGQA);
    }

    return match;
  }

  /**
   * Detect cache append operation
   */
  private detectCacheAppend(source: string): { found: boolean; type: string } {
    // Write to cache at sequence position
    if (/cache\s*\[\s*(?:seq|pos|cur|len)\s*[\+\*].*\]\s*=/i.test(source)) {
      return { found: true, type: 'position-indexed append' };
    }

    // Cache update at token position
    if (/k_?cache.*\[\s*\w+\s*\].*=\s*k(?:ey)?|v_?cache.*\[\s*\w+\s*\].*=\s*v(?:alue)?/i.test(source)) {
      return { found: true, type: 'direct KV update' };
    }

    // Copy new KV to cache
    if (/memcpy.*cache|cache.*=.*new_?(?:k|v)|append.*cache/i.test(source)) {
      return { found: true, type: 'memcpy append' };
    }

    // Concatenate style
    if (/concat.*cache|cat\s*\(.*cache|torch\.cat/i.test(source)) {
      return { found: true, type: 'concatenation' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect paged attention pattern (vLLM style)
   */
  private detectPagedAttention(source: string): { found: boolean } {
    // Block table lookup
    const hasBlockLookup = /block_?table\s*\[\s*\w+\s*\]/.test(source);

    // Physical block addressing
    const hasPhysBlock = /physical_?block|block_?offset|slot\s*=/.test(source);

    // Page-based iteration
    const hasPageIteration = /for\s*\([^)]*block|for\s*\([^)]*page/i.test(source);

    // vLLM naming
    const hasVLLMPattern = /vllm|paged_?attention/i.test(source);

    return { found: hasBlockLookup || hasPhysBlock || hasVLLMPattern || (hasPageIteration && hasPhysBlock) };
  }

  /**
   * Detect prefix caching pattern
   */
  private detectPrefixCaching(source: string): { found: boolean } {
    // Prefix/prompt cache reuse
    const hasPrefixReuse = /prefix_?cache|prompt_?cache|reuse_?cache/i.test(source);

    // Cache hit check
    const hasCacheHit = /cache_?hit|cached_?len|prefix_?len\s*>/i.test(source);

    // Skip already cached positions
    const hasSkipCached = /if\s*\([^)]*cached|skip.*prefix/i.test(source);

    return { found: hasPrefixReuse || hasCacheHit || hasSkipCached };
  }

  /**
   * Determine KV cache variant
   */
  private determineVariant(
    appendInfo: { found: boolean; type: string },
    pagedInfo: { found: boolean },
    prefixInfo: { found: boolean },
    hasMQAGQA: boolean
  ): PatternVariant {
    if (pagedInfo.found) {
      return 'kvcache_paged' as PatternVariant;
    }
    if (prefixInfo.found) {
      return 'kvcache_prefix' as PatternVariant;
    }
    if (hasMQAGQA) {
      return 'kvcache_gqa' as PatternVariant;
    }
    if (appendInfo.found) {
      return 'kvcache_append' as PatternVariant;
    }

    return 'kvcache_append' as PatternVariant; // default
  }
}

export const kvcacheMatcher = new KVCacheMatcher();
