// CUDA AST Extractor - Analyzes CUDA source code to extract kernel information

import {
  CudaKernelInfo,
  ParameterInfo,
  MemoryAccess,
  LoopInfo,
  SyncPoint,
  ThreadIndexUsage,
} from './types';

export class ASTExtractor {
  /**
   * Extract kernel information from CUDA source code
   */
  extract(source: string): CudaKernelInfo[] {
    const kernels: CudaKernelInfo[] = [];

    // Find all __global__ function definitions
    const kernelMatches = this.findKernelFunctions(source);

    for (const match of kernelMatches) {
      const kernelInfo = this.extractKernelInfo(match.name, match.params, match.body, source);
      kernels.push(kernelInfo);
    }

    // If no explicit kernels found, treat entire source as a kernel
    if (kernels.length === 0) {
      const implicitKernel = this.extractImplicitKernel(source);
      if (implicitKernel) {
        kernels.push(implicitKernel);
      }
    }

    return kernels;
  }

  private findKernelFunctions(source: string): Array<{ name: string; params: string; body: string }> {
    const results: Array<{ name: string; params: string; body: string }> = [];

    // Match __global__ void kernelName(params) { body }
    const kernelRegex = /__global__\s+(?:void|[\w]+)\s+(\w+)\s*\(([^)]*)\)\s*\{/g;
    let match;

    while ((match = kernelRegex.exec(source)) !== null) {
      const name = match[1];
      const params = match[2];
      const bodyStart = match.index + match[0].length;
      const body = this.extractFunctionBody(source, bodyStart);

      results.push({ name, params, body });
    }

    return results;
  }

  private extractFunctionBody(source: string, startIndex: number): string {
    let braceCount = 1;
    let i = startIndex;

    while (i < source.length && braceCount > 0) {
      if (source[i] === '{') braceCount++;
      else if (source[i] === '}') braceCount--;
      i++;
    }

    return source.substring(startIndex, i - 1);
  }

  private extractImplicitKernel(source: string): CudaKernelInfo | null {
    // Look for any function-like structure or treat entire source as kernel body
    const funcMatch = source.match(/(?:void|__device__|__host__)\s+(\w+)\s*\(([^)]*)\)\s*\{/);

    if (funcMatch) {
      const bodyStart = source.indexOf('{', funcMatch.index) + 1;
      const body = this.extractFunctionBody(source, bodyStart);
      return this.extractKernelInfo(funcMatch[1], funcMatch[2], body, source);
    }

    // Treat entire source as kernel body for analysis
    return this.extractKernelInfo('kernel', '', source, source);
  }

  private extractKernelInfo(
    name: string,
    paramsStr: string,
    body: string,
    fullSource: string
  ): CudaKernelInfo {
    return {
      name,
      parameters: this.extractParameters(paramsStr),
      memoryAccesses: this.extractMemoryAccesses(body),
      sharedMemoryDecls: this.extractSharedMemory(body),
      loops: this.extractLoops(body),
      syncPoints: this.extractSyncPoints(body),
      threadIndexUsage: this.extractThreadIndexUsage(body),
      sourceText: fullSource,
    };
  }

  private extractParameters(paramsStr: string): ParameterInfo[] {
    const params: ParameterInfo[] = [];
    if (!paramsStr.trim()) return params;

    // Split by comma, handling nested templates
    const paramList = this.splitParameters(paramsStr);

    for (const param of paramList) {
      const trimmed = param.trim();
      if (!trimmed) continue;

      const isConst = trimmed.includes('const');
      const isPointer = trimmed.includes('*');

      // Extract type and name
      const parts = trimmed.replace(/const/g, '').replace(/\*/g, '').trim().split(/\s+/);
      const name = parts[parts.length - 1];
      const type = parts.slice(0, -1).join(' ').trim() || 'auto';

      params.push({
        name,
        type,
        isPointer,
        isConst,
      });
    }

    return params;
  }

  private splitParameters(paramsStr: string): string[] {
    const params: string[] = [];
    let current = '';
    let depth = 0;

    for (const char of paramsStr) {
      if (char === '<') depth++;
      else if (char === '>') depth--;
      else if (char === ',' && depth === 0) {
        params.push(current);
        current = '';
        continue;
      }
      current += char;
    }

    if (current.trim()) params.push(current);
    return params;
  }

  private extractMemoryAccesses(body: string): MemoryAccess[] {
    const accesses: MemoryAccess[] = [];
    const lines = body.split('\n');

    // Match array access patterns: arr[index]
    const accessRegex = /(\w+)\s*\[([^\]]+)\]/g;

    for (let lineNum = 0; lineNum < lines.length; lineNum++) {
      const line = lines[lineNum];
      let match;

      // Reset regex for each line
      accessRegex.lastIndex = 0;

      while ((match = accessRegex.exec(line)) !== null) {
        const array = match[1];
        const indexExpr = match[2];

        // Skip if it's a declaration (e.g., __shared__ float arr[256])
        if (line.includes('__shared__') && line.indexOf(array) < line.indexOf('[')) {
          continue;
        }

        // Skip local array declarations (e.g., float arr[16])
        const beforeMatch = line.substring(0, match.index);
        if (/(?:float|double|int|half|__half|char|short|long|unsigned)\s*$/.test(beforeMatch.trim())) {
          continue;
        }

        // Determine if read or write with comprehensive detection
        const afterAccess = line.substring(match.index + match[0].length);
        const beforeAccess = line.substring(0, match.index);

        // Check for write operations
        const isDirectAssignment = /^\s*=(?!=)/.test(afterAccess);      // arr[i] = ...
        const isCompoundAssign = /^\s*(\+=|-=|\*=|\/=|%=|&=|\|=|\^=|<<=|>>=)/.test(afterAccess);
        const isPostfixIncDec = /^\s*(\+\+|--)/.test(afterAccess);       // arr[i]++ or arr[i]--
        const isPrefixIncDec = /(\+\+|--)\s*$/.test(beforeAccess);       // ++arr[i] or --arr[i]

        // Check for atomic operations (both read and write)
        const isAtomicOp = /atomic\w+\s*\([^,]*,\s*$/.test(beforeAccess); // First arg to atomic

        const isWrite = isDirectAssignment || isCompoundAssign || isPostfixIncDec || isPrefixIncDec;

        // For compound assignments (+=, etc.) and inc/dec, it's BOTH read and write
        const isReadWrite = isCompoundAssign || isPostfixIncDec || isPrefixIncDec || isAtomicOp;

        // Extract variables used in index
        const indexVars = this.extractVariables(indexExpr);

        // Check for neighbor offsets (i-1, i+1, idx-width, etc.)
        const hasNeighborOffset = /[\w]+\s*[-+]\s*\d+/.test(indexExpr) ||
                                  /[\w]+\s*[-+]\s*\w+/.test(indexExpr);

        // Extract numeric offset if present
        const offsetMatch = indexExpr.match(/[-+]\s*(\d+)\s*$/);
        const offset = offsetMatch ? parseInt(offsetMatch[1]) : undefined;

        // For read-write operations, add both a read and a write access
        if (isReadWrite) {
          // Add read first (happens before the write)
          accesses.push({
            array,
            indexExpression: indexExpr,
            indexVars,
            accessType: 'read',
            hasNeighborOffset,
            offset,
            line: lineNum + 1,
          });
          // Add write
          accesses.push({
            array,
            indexExpression: indexExpr,
            indexVars,
            accessType: 'write',
            hasNeighborOffset,
            offset,
            line: lineNum + 1,
          });
        } else {
          accesses.push({
            array,
            indexExpression: indexExpr,
            indexVars,
            accessType: isWrite ? 'write' : 'read',
            hasNeighborOffset,
            offset,
            line: lineNum + 1,
          });
        }
      }
    }

    return accesses;
  }

  private extractVariables(expr: string): string[] {
    const vars = new Set<string>();
    const varRegex = /\b([a-zA-Z_]\w*)\b/g;
    let match;

    while ((match = varRegex.exec(expr)) !== null) {
      const v = match[1];
      // Filter out common keywords and CUDA intrinsics
      if (!['threadIdx', 'blockIdx', 'blockDim', 'gridDim', 'int', 'float', 'double'].includes(v)) {
        vars.add(v);
      }
    }

    return Array.from(vars);
  }

  private extractSharedMemory(body: string): string[] {
    const shared: string[] = [];
    const sharedRegex = /__shared__\s+[\w\s\*]+\s+(\w+)\s*(?:\[|;|=)/g;
    let match;

    while ((match = sharedRegex.exec(body)) !== null) {
      shared.push(match[1]);
    }

    return shared;
  }

  private extractLoops(body: string): LoopInfo[] {
    const loops: LoopInfo[] = [];
    let nestLevel = 0;

    // Extract for loops
    const forLoops = this.extractForLoops(body, nestLevel);
    loops.push(...forLoops);
    nestLevel += forLoops.length;

    // Extract while loops
    const whileLoops = this.extractWhileLoops(body, nestLevel);
    loops.push(...whileLoops);
    nestLevel += whileLoops.length;

    // Extract do-while loops
    const doWhileLoops = this.extractDoWhileLoops(body, nestLevel);
    loops.push(...doWhileLoops);

    return loops;
  }

  private extractForLoops(body: string, startNestLevel: number): LoopInfo[] {
    const loops: LoopInfo[] = [];
    const forRegex = /for\s*\(\s*([^;]*);([^;]*);([^)]*)\)\s*\{?/g;
    let match;
    let nestLevel = startNestLevel;

    while ((match = forRegex.exec(body)) !== null) {
      const init = match[1].trim();
      const condition = match[2].trim();
      const update = match[3].trim();

      // Extract loop variable from init
      const initVar = this.extractLoopVariable(init);

      // Find loop body
      const loopStart = match.index + match[0].length;
      const loopBody = this.extractFunctionBody(body, loopStart);

      // Check for stride patterns
      const hasStrideHalving = />>=\s*1/.test(update) ||
                               /\/=\s*2/.test(update) ||
                               /=\s*\w+\s*\/\s*2/.test(update) ||
                               /=\s*\w+\s*>>\s*1/.test(update);

      const hasStrideDoubling = /<<=\s*1/.test(update) ||
                                /\*=\s*2/.test(update) ||
                                /=\s*\w+\s*\*\s*2/.test(update) ||
                                /=\s*\w+\s*<<\s*1/.test(update);

      // Check for __syncthreads in loop body
      const containsSyncthreads = loopBody.includes('__syncthreads');

      // Calculate line numbers
      const beforeLoop = body.substring(0, match.index);
      const startLine = beforeLoop.split('\n').length;

      loops.push({
        loopType: 'for',
        initVar,
        condition,
        update,
        body: loopBody,
        nestLevel: nestLevel++,
        containsSyncthreads,
        hasStrideHalving,
        hasStrideDoubling,
        startLine,
        endLine: startLine + loopBody.split('\n').length,
      });
    }

    return loops;
  }

  private extractWhileLoops(body: string, startNestLevel: number): LoopInfo[] {
    const loops: LoopInfo[] = [];
    // Match while (condition) { - but not do-while
    const whileRegex = /(?<!do\s*)while\s*\(([^)]+)\)\s*\{/g;
    let match;
    let nestLevel = startNestLevel;

    while ((match = whileRegex.exec(body)) !== null) {
      const condition = match[1].trim();

      // Find loop body
      const loopStart = match.index + match[0].length;
      const loopBody = this.extractFunctionBody(body, loopStart);

      // Analyze condition for potential loop variable
      const initVar = this.extractConditionVariable(condition);

      // Check for stride patterns in loop body
      const hasStrideHalving = />>=\s*1/.test(loopBody) ||
                               /\/=\s*2/.test(loopBody);

      const hasStrideDoubling = /<<=\s*1/.test(loopBody) ||
                                /\*=\s*2/.test(loopBody);

      const containsSyncthreads = loopBody.includes('__syncthreads');

      const beforeLoop = body.substring(0, match.index);
      const startLine = beforeLoop.split('\n').length;

      loops.push({
        loopType: 'while',
        initVar,
        condition,
        update: '', // while loops don't have explicit update
        body: loopBody,
        nestLevel: nestLevel++,
        containsSyncthreads,
        hasStrideHalving,
        hasStrideDoubling,
        startLine,
        endLine: startLine + loopBody.split('\n').length,
      });
    }

    return loops;
  }

  private extractDoWhileLoops(body: string, startNestLevel: number): LoopInfo[] {
    const loops: LoopInfo[] = [];
    // Match do { ... } while (condition);
    const doWhileRegex = /do\s*\{/g;
    let match;
    let nestLevel = startNestLevel;

    while ((match = doWhileRegex.exec(body)) !== null) {
      // Find loop body
      const loopStart = match.index + match[0].length;
      const loopBody = this.extractFunctionBody(body, loopStart);

      // Find the while condition after the body
      const afterBody = body.substring(loopStart + loopBody.length);
      const conditionMatch = afterBody.match(/\}\s*while\s*\(([^)]+)\)/);
      const condition = conditionMatch ? conditionMatch[1].trim() : '';

      const initVar = this.extractConditionVariable(condition);

      const hasStrideHalving = />>=\s*1/.test(loopBody) ||
                               /\/=\s*2/.test(loopBody);

      const hasStrideDoubling = /<<=\s*1/.test(loopBody) ||
                                /\*=\s*2/.test(loopBody);

      const containsSyncthreads = loopBody.includes('__syncthreads');

      const beforeLoop = body.substring(0, match.index);
      const startLine = beforeLoop.split('\n').length;

      loops.push({
        loopType: 'do-while',
        initVar,
        condition,
        update: '',
        body: loopBody,
        nestLevel: nestLevel++,
        containsSyncthreads,
        hasStrideHalving,
        hasStrideDoubling,
        startLine,
        endLine: startLine + loopBody.split('\n').length,
      });
    }

    return loops;
  }

  private extractConditionVariable(condition: string): string {
    // Extract the main variable from conditions like "i < n", "stride > 0", etc.
    const match = condition.match(/(\w+)\s*[<>=!]/);
    return match ? match[1] : '';
  }

  private extractLoopVariable(init: string): string {
    // Extract variable name from init like "int i = 0" or "i = 0"
    const match = init.match(/(?:int|size_t|unsigned)?\s*(\w+)\s*=/);
    return match ? match[1] : '';
  }

  private extractSyncPoints(body: string): SyncPoint[] {
    const syncPoints: SyncPoint[] = [];
    const lines = body.split('\n');

    const syncPatterns = [
      { regex: /__syncthreads\s*\(\s*\)/, type: 'syncthreads' as const },
      { regex: /__syncwarp\s*\([^)]*\)/, type: 'syncwarp' as const },
      { regex: /atomic(\w+)\s*\([^)]+\)/, type: 'atomic' as const },
    ];

    for (let lineNum = 0; lineNum < lines.length; lineNum++) {
      const line = lines[lineNum];

      for (const { regex, type } of syncPatterns) {
        const match = line.match(regex);
        if (match) {
          syncPoints.push({
            type,
            name: match[0],
            line: lineNum + 1,
          });
        }
      }
    }

    return syncPoints;
  }

  private extractThreadIndexUsage(body: string): ThreadIndexUsage {
    const usage: ThreadIndexUsage = {
      usesThreadIdxX: /threadIdx\.x/.test(body),
      usesThreadIdxY: /threadIdx\.y/.test(body),
      usesThreadIdxZ: /threadIdx\.z/.test(body),
      usesBlockIdxX: /blockIdx\.x/.test(body),
      usesBlockIdxY: /blockIdx\.y/.test(body),
      usesBlockIdxZ: /blockIdx\.z/.test(body),
      usesBlockDim: /blockDim\./.test(body),
    };

    // Try to extract global ID expression
    const globalIdMatch = body.match(
      /(\w+)\s*=\s*threadIdx\.x\s*\+\s*blockIdx\.x\s*\*\s*blockDim\.x/
    ) || body.match(
      /(\w+)\s*=\s*blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x/
    );

    if (globalIdMatch) {
      usage.globalIdExpression = globalIdMatch[0];
    }

    return usage;
  }
}

export const astExtractor = new ASTExtractor();
