/**
 * Embedding Pattern Matcher
 * Detects embedding lookup, embedding bag, and positional embedding patterns
 */

import { CudaKernelInfo, PatternMatch, Evidence, PatternVariant } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class EmbeddingMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;

    // === Primary Indicators (high weight) ===

    // 1. Table lookup: embedding[idx] or embedding[idx * dim + d]
    const lookupInfo = this.detectTableLookup(source, kernel);
    if (lookupInfo.found) {
      addEvidence(evidence, 'table_lookup', 0.35,
        `Table lookup: ${lookupInfo.type}`);
    }

    // 2. Index-based gather operation
    const gatherInfo = this.detectGather(source);
    if (gatherInfo.found) {
      addEvidence(evidence, 'gather_pattern', 0.30,
        'Gather operation detected');
    }

    // 3. Embedding bag (sum/mean of multiple embeddings)
    const bagInfo = this.detectEmbeddingBag(source);
    if (bagInfo.found) {
      addEvidence(evidence, 'embedding_bag', 0.30,
        `Embedding bag: ${bagInfo.type}`);
    }

    // === Secondary Indicators (medium weight) ===

    // 4. Integer indices used to access 2D table
    const hasIntIndices = /\[\s*(?:idx|index|token|id)\s*\]/.test(source) ||
                          /\[\s*\w+\s*\[\s*\w+\s*\]\s*\]/.test(source);
    if (hasIntIndices) {
      addEvidence(evidence, 'int_indices', 0.15,
        'Integer indices for lookup');
    }

    // 5. Embedding dimension loop
    const hasDimLoop = /for\s*\([^)]*embed|for\s*\([^)]*dim|for\s*\([^)]*d\s*=/.test(source);
    if (hasDimLoop && lookupInfo.found) {
      addEvidence(evidence, 'dim_loop', 0.10,
        'Embedding dimension iteration');
    }

    // 6. Name hints
    if (/embed|lookup|vocab|token|position|gather/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests embedding');
    }

    // 7. Parameter names
    const embedParams = kernel.parameters.filter(p =>
      /embed|weight|table|vocab|dictionary/i.test(p.name)
    );
    if (embedParams.length >= 1) {
      addEvidence(evidence, 'embed_params', 0.10,
        'Embedding parameters detected');
    }

    // 8. Copy pattern (memcpy-like for embedding row)
    const hasCopyPattern = /output\s*\[\s*\w+\s*\*\s*dim.*\+\s*\w+\s*\]\s*=\s*embed/.test(source);
    if (hasCopyPattern) {
      addEvidence(evidence, 'copy_pattern', 0.15,
        'Embedding row copy');
    }

    // 9. Positional embedding pattern
    const isPosEmbed = this.detectPositionalEmbedding(source, kernel);
    if (isPosEmbed) {
      addEvidence(evidence, 'pos_embed', 0.20,
        'Positional embedding detected');
    }

    // === Negative Indicators ===

    // Matrix multiplication pattern
    if (/\[\s*i\s*\]\s*\[\s*k\s*\]\s*\*\s*\[\s*k\s*\]\s*\[\s*j\s*\]/.test(source)) {
      addEvidence(evidence, 'matmul_pattern', -0.25,
        'Matrix multiplication (not embedding)');
    }

    // Convolution pattern
    if (/stride|kernel_size|filter/i.test(source) && !lookupInfo.found) {
      addEvidence(evidence, 'conv_pattern', -0.15,
        'Convolution pattern');
    }

    const match = createPatternMatch('embedding' as any, evidence, warnings);

    if (match.confidence > 0.3) {
      match.variant = this.determineVariant(lookupInfo, bagInfo, isPosEmbed);
    }

    return match;
  }

  /**
   * Detect embedding table lookup
   */
  private detectTableLookup(source: string, kernel: CudaKernelInfo): { found: boolean; type: string } {
    // 2D table access with integer index
    if (/\w+\s*\[\s*\w+\s*\]\s*\[\s*\w+\s*\]/.test(source) &&
        kernel.parameters.some(p => /embed|weight|table/i.test(p.name))) {
      return { found: true, type: '2D table access' };
    }

    // Linear index: table[idx * dim + d]
    if (/\w+\s*\[\s*\w+\s*\*\s*(?:dim|embed|hidden)\s*\+\s*\w+\s*\]/i.test(source)) {
      return { found: true, type: 'linear index access' };
    }

    // Gather style: output[i] = table[indices[i]]
    if (/\w+\s*\[\s*\w+\s*\]\s*=\s*\w+\s*\[\s*\w+\s*\[\s*\w+\s*\]\s*\]/.test(source)) {
      return { found: true, type: 'gather-style lookup' };
    }

    // Token embedding
    if (/token.*embed|embed.*token|vocab\s*\[/i.test(source)) {
      return { found: true, type: 'token embedding' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect gather operation
   */
  private detectGather(source: string): { found: boolean } {
    // Index-based gather
    const patterns = [
      /gather|scatter/i,
      /\w+\s*\[\s*indices\s*\[\s*\w+\s*\]\s*\]/,
      /idx\s*=\s*\w+\s*\[\s*\w+\s*\].*\w+\s*\[\s*idx\s*\]/,
    ];

    return { found: patterns.some(p => p.test(source)) };
  }

  /**
   * Detect embedding bag (aggregation of multiple embeddings)
   */
  private detectEmbeddingBag(source: string): { found: boolean; type: string } {
    // Sum mode: sum embeddings for variable-length sequences
    if (/embeddingbag|embedding_bag/i.test(source)) {
      return { found: true, type: 'explicit embedding_bag' };
    }

    // Sum/mean aggregation with offsets
    if (/offset.*\[|bag.*sum|pool.*embed/i.test(source)) {
      return { found: true, type: 'offset-based aggregation' };
    }

    // Accumulate multiple embeddings
    if (/for\s*\([^)]*\)[^}]*embed\s*\[\s*\w+\s*\[\s*\w+\s*\]\s*\][^}]*\+=/.test(source)) {
      return { found: true, type: 'embedding accumulation' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect positional embedding
   */
  private detectPositionalEmbedding(source: string, kernel: CudaKernelInfo): boolean {
    // Explicit positional embedding
    if (/pos_?embed|position|positional/i.test(kernel.name)) {
      return true;
    }

    // Position-based indexing
    if (/pos\s*\*\s*dim|position\s*\[\s*seq|seq_idx\s*\*/.test(source)) {
      return true;
    }

    // Sinusoidal position encoding
    if (/sin\s*\(.*pos|cos\s*\(.*pos|2\s*\*\s*i\s*\/\s*d_model/i.test(source)) {
      return true;
    }

    return false;
  }

  /**
   * Determine embedding variant
   */
  private determineVariant(
    lookupInfo: { found: boolean; type: string },
    bagInfo: { found: boolean; type: string },
    isPosEmbed: boolean
  ): PatternVariant {
    if (bagInfo.found) return 'embedding_bag' as PatternVariant;
    if (isPosEmbed) return 'positional_embedding' as PatternVariant;
    if (lookupInfo.found) return 'embedding_lookup' as PatternVariant;

    return 'embedding_lookup' as PatternVariant; // default
  }
}

export const embeddingMatcher = new EmbeddingMatcher();
