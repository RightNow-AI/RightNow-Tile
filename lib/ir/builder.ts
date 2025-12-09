// IR Builder - Transforms CUDA kernel info into Intermediate Representation

import {
  CudaKernelInfo,
  PatternMatch,
  KernelIR,
  IRParameter,
  IRLoad,
  IROperation,
  IRStore,
  TileConfig,
  KernelArchetype,
} from '../ast/types';

export class IRBuilder {
  /**
   * Build IR from kernel info and pattern match
   */
  build(kernel: CudaKernelInfo, pattern: PatternMatch): KernelIR {
    const tileConfig = this.determineTileConfig(kernel, pattern.archetype);

    switch (pattern.archetype) {
      case 'gemm':
        return this.buildGEMMIR(kernel, pattern, tileConfig);
      case 'reduction':
        return this.buildReductionIR(kernel, pattern, tileConfig);
      case 'scan':
        return this.buildScanIR(kernel, pattern, tileConfig);
      case 'stencil':
        return this.buildStencilIR(kernel, pattern, tileConfig);
      case 'elementwise':
      default:
        return this.buildElementwiseIR(kernel, pattern, tileConfig);
    }
  }

  private determineTileConfig(kernel: CudaKernelInfo, archetype: KernelArchetype): TileConfig {
    // Default tile sizes based on pattern
    switch (archetype) {
      case 'gemm':
        return {
          tileSize: 128,
          blockM: 128,
          blockN: 128,
          blockK: 32,
        };
      case 'reduction':
        return { tileSize: 256 };
      case 'scan':
        return { tileSize: 256 };
      case 'stencil':
        return { tileSize: 16 }; // 2D tiles typically smaller
      case 'elementwise':
      default:
        return { tileSize: 256 };
    }
  }

  private buildElementwiseIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): KernelIR {
    // Map parameters
    const parameters = this.mapParameters(kernel);

    // Identify input arrays (read-only) vs output arrays (written)
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    // Build loads
    const loads: IRLoad[] = inputArrays.map(arr => ({
      source: arr,
      target: `${arr}_tile`,
      index: '(pid,)',
      shape: '(tile_size,)',
    }));

    // Detect the operation being performed
    const operations = this.extractElementwiseOperations(kernel, loads);

    // Build stores
    const stores: IRStore[] = outputArrays.map(arr => ({
      source: operations.length > 0 ? operations[operations.length - 1].output : 'result',
      target: arr,
      index: '(pid,)',
    }));

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'elementwise',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
    };
  }

  private buildGEMMIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): KernelIR {
    const parameters = this.mapParameters(kernel);

    // For GEMM, we need A, B input and C output
    const pointerParams = kernel.parameters.filter(p => p.isPointer);
    const matrixNames = pointerParams.map(p => p.name.toLowerCase());

    // Standard A, B, C naming
    const [aName, bName, cName] = matrixNames.length >= 3
      ? matrixNames
      : ['a', 'b', 'c'];

    const loads: IRLoad[] = [
      {
        source: aName,
        target: 'a_tile',
        index: '(pid_m, k)',
        shape: '(block_m, block_k)',
      },
      {
        source: bName,
        target: 'b_tile',
        index: '(k, pid_n)',
        shape: '(block_k, block_n)',
      },
    ];

    const operations: IROperation[] = [
      {
        type: 'matmul',
        inputs: ['a_tile', 'b_tile', 'acc'],
        output: 'acc',
        op: 'matmul',
      },
    ];

    const stores: IRStore[] = [
      {
        source: 'acc',
        target: cName,
        index: '(pid_m, pid_n)',
      },
    ];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'gemm',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
    };
  }

  private buildReductionIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): KernelIR {
    const parameters = this.mapParameters(kernel);

    // Identify input and output arrays
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const inputName = inputArrays[0] || 'input_arr';
    const outputName = outputArrays[0] || 'output_arr';

    const loads: IRLoad[] = [
      {
        source: inputName,
        target: 'tile',
        index: '(pid,)',
        shape: '(tile_size,)',
      },
    ];

    // Detect reduction operation (sum, max, min, etc.)
    const reductionOp = this.detectReductionOp(kernel);

    const operations: IROperation[] = [
      {
        type: 'reduce',
        inputs: ['tile'],
        output: 'result',
        op: reductionOp,
      },
      {
        type: 'atomic',
        inputs: [outputName, 'result'],
        output: outputName,
        op: 'add',
      },
    ];

    const stores: IRStore[] = []; // Atomic handles the store

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'reduction',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
    };
  }

  private buildScanIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): KernelIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const inputName = inputArrays[0] || 'data';
    const outputName = outputArrays[0] || inputName;

    const loads: IRLoad[] = [
      {
        source: inputName,
        target: 'tile',
        index: '(pid,)',
        shape: '(tile_size,)',
      },
    ];

    const operations: IROperation[] = [
      {
        type: 'reduce', // cuTile handles scan via cumsum
        inputs: ['tile'],
        output: 'scanned',
        op: 'cumsum', // Cumulative sum = inclusive scan
      },
    ];

    const stores: IRStore[] = [
      {
        source: 'scanned',
        target: outputName,
        index: '(pid,)',
      },
    ];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'scan',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
    };
  }

  private buildStencilIR(
    kernel: CudaKernelInfo,
    pattern: PatternMatch,
    tileConfig: TileConfig
  ): KernelIR {
    const parameters = this.mapParameters(kernel);
    const inputArrays = this.identifyInputArrays(kernel);
    const outputArrays = this.identifyOutputArrays(kernel);

    const inputName = inputArrays[0] || 'input_arr';
    const outputName = outputArrays[0] || 'output_arr';

    // For stencil, we need to load a tile with halo
    const loads: IRLoad[] = [
      {
        source: inputName,
        target: 'tile',
        index: '(pid_y, pid_x)',
        shape: '(tile_size + 2, tile_size + 2)', // +2 for halo
      },
    ];

    // Stencil operation (simplified as average of neighbors)
    const operations: IROperation[] = [
      {
        type: 'elementwise',
        inputs: ['tile'],
        output: 'result',
        op: 'stencil_apply',
      },
    ];

    const stores: IRStore[] = [
      {
        source: 'result',
        target: outputName,
        index: '(pid_y, pid_x)',
      },
    ];

    return {
      name: this.generateCuTileName(kernel.name),
      originalName: kernel.name,
      archetype: 'stencil',
      confidence: pattern.confidence,
      parameters,
      loads,
      operations,
      stores,
      tileConfig,
    };
  }

  private mapParameters(kernel: CudaKernelInfo): IRParameter[] {
    return kernel.parameters.map(p => {
      const isConstant = !p.isPointer && (p.type === 'int' || p.type === 'size_t');
      return {
        cudaName: p.name,
        cuTileName: p.name.toLowerCase(),
        type: p.type,
        isConstant,
        constantAnnotation: isConstant ? `ct.Constant[${this.mapType(p.type)}]` : undefined,
      };
    });
  }

  private mapType(cudaType: string): string {
    const typeMap: Record<string, string> = {
      'int': 'int',
      'unsigned int': 'int',
      'size_t': 'int',
      'float': 'float',
      'double': 'float',
      'float*': 'float',
      'double*': 'float',
      'int*': 'int',
    };
    return typeMap[cudaType.toLowerCase()] || 'int';
  }

  private generateCuTileName(cudaName: string): string {
    // Convert camelCase or snake_case to a clean name
    return cudaName
      .replace(/([A-Z])/g, '_$1')
      .toLowerCase()
      .replace(/^_/, '')
      .replace(/__+/g, '_');
  }

  private identifyInputArrays(kernel: CudaKernelInfo): string[] {
    const reads = new Set(
      kernel.memoryAccesses
        .filter(a => a.accessType === 'read')
        .map(a => a.array)
    );

    const writes = new Set(
      kernel.memoryAccesses
        .filter(a => a.accessType === 'write')
        .map(a => a.array)
    );

    // Input arrays are read but NOT written (pure inputs)
    return Array.from(reads).filter(arr => !writes.has(arr));
  }

  private identifyOutputArrays(kernel: CudaKernelInfo): string[] {
    const writes = kernel.memoryAccesses
      .filter(a => a.accessType === 'write')
      .map(a => a.array);

    return [...new Set(writes)];
  }

  private extractElementwiseOperations(kernel: CudaKernelInfo, loads: IRLoad[]): IROperation[] {
    const operations: IROperation[] = [];
    const source = kernel.sourceText;

    // Try to detect the operation being performed
    const tileVars = loads.map(l => l.target);

    // Look for common operations
    if (/\+/.test(source) && tileVars.length >= 2) {
      operations.push({
        type: 'elementwise',
        inputs: tileVars.slice(0, 2),
        output: 'result',
        op: '+',
      });
    } else if (/\*/.test(source) && !/\+=/.test(source) && tileVars.length >= 2) {
      operations.push({
        type: 'elementwise',
        inputs: tileVars.slice(0, 2),
        output: 'result',
        op: '*',
      });
    } else if (/-/.test(source) && tileVars.length >= 2) {
      operations.push({
        type: 'elementwise',
        inputs: tileVars.slice(0, 2),
        output: 'result',
        op: '-',
      });
    } else if (tileVars.length === 1) {
      // Unary operation or copy
      operations.push({
        type: 'elementwise',
        inputs: [tileVars[0]],
        output: 'result',
        op: 'identity',
      });
    } else if (tileVars.length >= 2) {
      // Default to addition
      operations.push({
        type: 'elementwise',
        inputs: tileVars.slice(0, 2),
        output: 'result',
        op: '+',
      });
    }

    return operations;
  }

  private detectReductionOp(kernel: CudaKernelInfo): string {
    const source = kernel.sourceText.toLowerCase();

    if (/atomicmax|max\s*\(/.test(source)) return 'max';
    if (/atomicmin|min\s*\(/.test(source)) return 'min';
    if (/\*=|\*\s*=/.test(source) && !/\+/.test(source)) return 'prod';
    // Default to sum
    return 'sum';
  }
}

export const irBuilder = new IRBuilder();
