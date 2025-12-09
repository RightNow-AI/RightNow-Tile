// Semantic Validator - Validates generated cuTile code and adjusts confidence

import { CudaKernelInfo, KernelIR, ValidationResult, PatternMatch } from '../ast/types';

export class SemanticValidator {
  /**
   * Validate generated cuTile code against the original kernel
   */
  validate(kernel: CudaKernelInfo, ir: KernelIR, generatedCode: string): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    let confidenceAdjustment = 0;

    // 1. Check parameter preservation
    const paramResult = this.validateParameters(kernel, generatedCode);
    errors.push(...paramResult.errors);
    warnings.push(...paramResult.warnings);
    confidenceAdjustment += paramResult.adjustment;

    // 2. Validate cuTile API usage
    const apiResult = this.validateCuTileAPI(generatedCode);
    errors.push(...apiResult.errors);
    warnings.push(...apiResult.warnings);
    confidenceAdjustment += apiResult.adjustment;

    // 3. Validate tile sizes
    const tileResult = this.validateTileSizes(generatedCode);
    errors.push(...tileResult.errors);
    warnings.push(...tileResult.warnings);
    confidenceAdjustment += tileResult.adjustment;

    // 4. Check archetype-specific requirements
    const archetypeResult = this.validateArchetype(kernel, ir);
    errors.push(...archetypeResult.errors);
    warnings.push(...archetypeResult.warnings);
    confidenceAdjustment += archetypeResult.adjustment;

    // 5. Check for structural completeness
    const structureResult = this.validateStructure(generatedCode, ir);
    errors.push(...structureResult.errors);
    warnings.push(...structureResult.warnings);
    confidenceAdjustment += structureResult.adjustment;

    // Calculate adjusted confidence
    const adjustedConfidence = Math.max(0, Math.min(1,
      ir.confidence + confidenceAdjustment
    ));

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      adjustedConfidence,
    };
  }

  private validateParameters(
    kernel: CudaKernelInfo,
    generatedCode: string
  ): { errors: string[]; warnings: string[]; adjustment: number } {
    const errors: string[] = [];
    const warnings: string[] = [];
    let adjustment = 0;

    for (const param of kernel.parameters) {
      const paramNameLower = param.name.toLowerCase();

      // Check if parameter appears in generated code
      if (!generatedCode.toLowerCase().includes(paramNameLower)) {
        if (param.isPointer) {
          warnings.push(`Array parameter '${param.name}' may not be mapped correctly`);
          adjustment -= 0.05;
        } else {
          // Scalar parameters might be inlined or renamed
          // Only warn, don't penalize heavily
        }
      }
    }

    // Check parameter count matches
    const cudaParamCount = kernel.parameters.filter(p => p.isPointer).length;
    const cuTileParamCount = (generatedCode.match(/ct\.load\(/g) || []).length;

    if (cuTileParamCount < cudaParamCount - 1) {
      warnings.push(`Generated code has fewer loads (${cuTileParamCount}) than CUDA arrays (${cudaParamCount})`);
      adjustment -= 0.05;
    }

    return { errors, warnings, adjustment };
  }

  private validateCuTileAPI(
    generatedCode: string
  ): { errors: string[]; warnings: string[]; adjustment: number } {
    const errors: string[] = [];
    const warnings: string[] = [];
    let adjustment = 0;

    // Valid cuTile functions
    const validAPIs = [
      'ct.kernel',
      'ct.load',
      'ct.store',
      'ct.bid',
      'ct.tid',
      'ct.launch',
      'ct.cdiv',
      'ct.zeros',
      'ct.reduce',
      'ct.cumsum',
      'ct.tile_matmul',
      'ct.atomic_add',
      'ct.Constant',
      'ct.sum',
      'ct.max',
      'ct.min',
      'ct.float32',
      'ct.float64',
      'ct.int32',
    ];

    // Find all ct.* calls in generated code
    const ctCalls = generatedCode.match(/ct\.\w+/g) || [];

    for (const call of ctCalls) {
      const isValid = validAPIs.some(api => call.startsWith(api.split('[')[0]));
      if (!isValid) {
        warnings.push(`Unknown cuTile API: ${call}`);
        adjustment -= 0.02;
      }
    }

    // Check for required elements
    if (!generatedCode.includes('@ct.kernel')) {
      errors.push('Missing @ct.kernel decorator');
      adjustment -= 0.1;
    }

    if (!generatedCode.includes('ct.bid') && !generatedCode.includes('ct.tid')) {
      warnings.push('No block/thread ID access - may not parallelize correctly');
      adjustment -= 0.05;
    }

    return { errors, warnings, adjustment };
  }

  private validateTileSizes(
    generatedCode: string
  ): { errors: string[]; warnings: string[]; adjustment: number } {
    const errors: string[] = [];
    const warnings: string[] = [];
    let adjustment = 0;

    // Extract tile size values
    const tileSizeMatches = generatedCode.matchAll(/(?:TILE_SIZE|BLOCK_[MNK])\s*=\s*(\d+)/g);

    for (const match of tileSizeMatches) {
      const size = parseInt(match[1]);

      // Check power of 2 (recommended but not required)
      if (!this.isPowerOfTwo(size)) {
        warnings.push(`Tile size ${size} is not a power of 2 (may affect performance)`);
        // Don't penalize, just warn
      }

      // Check reasonable range
      if (size < 16) {
        warnings.push(`Tile size ${size} is very small (may be inefficient)`);
        adjustment -= 0.02;
      } else if (size > 1024) {
        warnings.push(`Tile size ${size} is very large (may exceed resources)`);
        adjustment -= 0.02;
      }
    }

    return { errors, warnings, adjustment };
  }

  private validateArchetype(
    kernel: CudaKernelInfo,
    ir: KernelIR
  ): { errors: string[]; warnings: string[]; adjustment: number } {
    const errors: string[] = [];
    const warnings: string[] = [];
    let adjustment = 0;

    switch (ir.archetype) {
      case 'gemm':
        // GEMM should have at least 3 array parameters
        const pointerParams = kernel.parameters.filter(p => p.isPointer);
        if (pointerParams.length < 3) {
          warnings.push('GEMM typically needs 3 arrays (A, B, C)');
          adjustment -= 0.05;
        }

        // Should have multiplication
        if (!kernel.sourceText.includes('*')) {
          warnings.push('GEMM kernel should have multiplication operations');
          adjustment -= 0.1;
        }
        break;

      case 'reduction':
        // Reduction should have shared memory or atomics
        if (kernel.sharedMemoryDecls.length === 0 &&
            kernel.syncPoints.filter(s => s.type === 'atomic').length === 0) {
          warnings.push('Reduction without shared memory or atomics may be incorrect');
          adjustment -= 0.05;
        }
        break;

      case 'scan':
        // Scan should have shared memory
        if (kernel.sharedMemoryDecls.length === 0) {
          warnings.push('Efficient scan typically requires shared memory');
          adjustment -= 0.03;
        }
        break;

      case 'stencil':
        // Stencil should have neighbor accesses
        const neighborAccesses = kernel.memoryAccesses.filter(a => a.hasNeighborOffset);
        if (neighborAccesses.length < 2) {
          warnings.push('Stencil detected but few neighbor accesses found');
          adjustment -= 0.05;
        }
        break;

      case 'elementwise':
        // Elementwise is the fallback - boost confidence if it's clearly simple
        if (kernel.loops.length === 0 &&
            kernel.sharedMemoryDecls.length === 0 &&
            kernel.syncPoints.length === 0) {
          adjustment += 0.05; // Boost for clean elementwise
        }
        break;
    }

    return { errors, warnings, adjustment };
  }

  private validateStructure(
    generatedCode: string,
    ir: KernelIR
  ): { errors: string[]; warnings: string[]; adjustment: number } {
    const errors: string[] = [];
    const warnings: string[] = [];
    let adjustment = 0;

    // Check for balanced loads and stores
    const loadCount = (generatedCode.match(/ct\.load\(/g) || []).length;
    const storeCount = (generatedCode.match(/ct\.store\(/g) || []).length;

    if (loadCount === 0) {
      errors.push('No ct.load calls found - kernel cannot read data');
      adjustment -= 0.2;
    }

    if (storeCount === 0 && !generatedCode.includes('ct.atomic')) {
      warnings.push('No ct.store or atomic calls - kernel may not write results');
      adjustment -= 0.1;
    }

    // Check for kernel decorator
    if (!generatedCode.includes('@ct.kernel')) {
      errors.push('Missing @ct.kernel decorator');
      adjustment -= 0.15;
    }

    // Check for launch function
    if (!generatedCode.includes('ct.launch')) {
      warnings.push('No launch function generated');
      // Don't penalize - launch function is helper code
    }

    return { errors, warnings, adjustment };
  }

  private isPowerOfTwo(n: number): boolean {
    return n > 0 && (n & (n - 1)) === 0;
  }
}

export const semanticValidator = new SemanticValidator();
