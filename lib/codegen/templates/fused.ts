/**
 * Fused Pattern Templates
 * Specialized code generation for fused kernels (matmul+activation, normalization+residual, etc.)
 */

import { EnhancedKernelIR } from '../../ir/types';
import { FusedIR } from '../../ir/builder';

/**
 * Generate Matmul + Activation fused kernel
 * Common in transformer feed-forward networks
 */
export function generateMatmulActivation(ir: EnhancedKernelIR & Partial<FusedIR>): string {
  const blockM = ir.tileConfig.blockM || 128;
  const blockN = ir.tileConfig.blockN || 128;
  const blockK = ir.tileConfig.blockK || 32;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const weightName = ir.loads[1]?.source || 'weight';
  const outputName = ir.stores[0]?.target || 'output';

  // Detect activation type from operations
  const activationOp = ir.operations.find(op => op.op === 'activation')?.op || 'relu';
  const activationCode = getActivationCode(activationOp);

  return `import cuda_tile as ct
import cupy
import math

BLOCK_M = ${blockM}
BLOCK_N = ${blockN}
BLOCK_K = ${blockK}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${weightName}, ${outputName},
    M: ct.Constant[int],
    N: ct.Constant[int],
    K: ct.Constant[int]
):
    """
    Fused Matmul + Activation kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: matmul_activation

    Fuses matrix multiplication with activation function
    to avoid extra memory roundtrip.
    """
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    # Initialize accumulator
    acc = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)

    # Tile over K dimension
    for k_block in range(ct.cdiv(K, BLOCK_K)):
        k_start = k_block * BLOCK_K

        # Load input tile
        a_tile = ct.load(
            ${inputName},
            index=(pid_m * BLOCK_M, k_start),
            shape=(BLOCK_M, BLOCK_K),
            mask=(pid_m * BLOCK_M + ct.arange(BLOCK_M)[:, None] < M) &
                 (k_start + ct.arange(BLOCK_K)[None, :] < K)
        )

        # Load weight tile
        b_tile = ct.load(
            ${weightName},
            index=(k_start, pid_n * BLOCK_N),
            shape=(BLOCK_K, BLOCK_N),
            mask=(k_start + ct.arange(BLOCK_K)[:, None] < K) &
                 (pid_n * BLOCK_N + ct.arange(BLOCK_N)[None, :] < N)
        )

        # Accumulate matmul
        acc = acc + ct.tile_matmul(a_tile, b_tile)

    # Apply activation function (fused - no extra memory access)
    ${activationCode}

    # Store result
    ct.store(
        ${outputName},
        index=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        tile=result,
        mask=(pid_m * BLOCK_M + ct.arange(BLOCK_M)[:, None] < M) &
             (pid_n * BLOCK_N + ct.arange(BLOCK_N)[None, :] < N)
    )


def launch_${ir.name}(${inputName}, ${weightName}, ${outputName}):
    """Launch the ${ir.name} fused matmul + activation kernel"""
    M, K = ${inputName}.shape
    K2, N = ${weightName}.shape
    assert K == K2, "Inner dimensions must match"

    grid = (ct.cdiv(M, BLOCK_M), ct.cdiv(N, BLOCK_N), 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${weightName}, ${outputName}, M, N, K))`;
}

/**
 * Generate Matmul + Bias + Activation fused kernel
 * Common in linear layers with bias
 */
export function generateMatmulBiasActivation(ir: EnhancedKernelIR & Partial<FusedIR>): string {
  const blockM = ir.tileConfig.blockM || 128;
  const blockN = ir.tileConfig.blockN || 128;
  const blockK = ir.tileConfig.blockK || 32;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const weightName = ir.loads[1]?.source || 'weight';
  const biasName = ir.loads[2]?.source || 'bias';
  const outputName = ir.stores[0]?.target || 'output';

  const activationCode = getActivationCode('gelu');

  return `import cuda_tile as ct
import cupy
import math

BLOCK_M = ${blockM}
BLOCK_N = ${blockN}
BLOCK_K = ${blockK}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${weightName}, ${biasName}, ${outputName},
    M: ct.Constant[int],
    N: ct.Constant[int],
    K: ct.Constant[int]
):
    """
    Fused Matmul + Bias + Activation kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: matmul_bias_activation

    Fuses: output = activation(input @ weight + bias)
    Common in transformer FFN layers.
    """
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    # Initialize accumulator
    acc = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)

    # Tile over K dimension
    for k_block in range(ct.cdiv(K, BLOCK_K)):
        k_start = k_block * BLOCK_K

        a_tile = ct.load(
            ${inputName},
            index=(pid_m * BLOCK_M, k_start),
            shape=(BLOCK_M, BLOCK_K)
        )
        b_tile = ct.load(
            ${weightName},
            index=(k_start, pid_n * BLOCK_N),
            shape=(BLOCK_K, BLOCK_N)
        )

        acc = acc + ct.tile_matmul(a_tile, b_tile)

    # Load and add bias (broadcast over M dimension)
    bias_tile = ct.load(${biasName}, index=(pid_n * BLOCK_N,), shape=(BLOCK_N,))
    acc = acc + bias_tile[None, :]

    # Apply activation function
    ${activationCode}

    # Store result
    ct.store(${outputName}, index=(pid_m * BLOCK_M, pid_n * BLOCK_N), tile=result)


def launch_${ir.name}(${inputName}, ${weightName}, ${biasName}, ${outputName}):
    """Launch the ${ir.name} fused matmul + bias + activation kernel"""
    M, K = ${inputName}.shape
    K2, N = ${weightName}.shape
    grid = (ct.cdiv(M, BLOCK_M), ct.cdiv(N, BLOCK_N), 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${weightName}, ${biasName}, ${outputName}, M, N, K))`;
}

/**
 * Generate LayerNorm + Residual fused kernel
 * Common in transformer blocks
 */
export function generateLayerNormResidual(ir: EnhancedKernelIR & Partial<FusedIR>): string {
  const tileSize = ir.tileConfig.tileSize || 256;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const residualName = ir.loads[1]?.source || 'residual';
  const gammaName = 'gamma';
  const betaName = 'beta';
  const outputName = ir.stores[0]?.target || 'output';

  return `import cuda_tile as ct
import cupy
import math

TILE_SIZE = ${tileSize}
EPSILON = 1e-5

@ct.kernel
def ${ir.name}(
    ${inputName}, ${residualName}, ${gammaName}, ${betaName}, ${outputName},
    batch_size: ct.Constant[int],
    hidden_size: ct.Constant[int]
):
    """
    Fused LayerNorm + Residual kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: layernorm_residual

    Fuses: output = LayerNorm(input + residual)
    Saves memory bandwidth by avoiding intermediate storage.
    """
    pid = ct.bid(0)  # Batch/sequence index
    tid = ct.tid(0)

    if pid >= batch_size:
        return

    # First pass: compute mean
    sum_val = ct.float32(0.0)
    for i in range(tid, hidden_size, TILE_SIZE):
        val = ${inputName}[pid * hidden_size + i] + ${residualName}[pid * hidden_size + i]
        sum_val = sum_val + val

    # Warp-level reduction for mean
    for offset in [16, 8, 4, 2, 1]:
        sum_val = sum_val + ct.shfl_down(sum_val, offset)

    # Share mean across block
    shared_mean = ct.shared_zeros((1,), dtype=ct.float32)
    if tid == 0:
        shared_mean[0] = sum_val / hidden_size
    ct.sync_threads()
    mean = shared_mean[0]

    # Second pass: compute variance
    var_sum = ct.float32(0.0)
    for i in range(tid, hidden_size, TILE_SIZE):
        val = ${inputName}[pid * hidden_size + i] + ${residualName}[pid * hidden_size + i]
        diff = val - mean
        var_sum = var_sum + diff * diff

    for offset in [16, 8, 4, 2, 1]:
        var_sum = var_sum + ct.shfl_down(var_sum, offset)

    shared_var = ct.shared_zeros((1,), dtype=ct.float32)
    if tid == 0:
        shared_var[0] = ct.rsqrt(var_sum / hidden_size + EPSILON)
    ct.sync_threads()
    rstd = shared_var[0]

    # Third pass: normalize and apply affine transform
    for i in range(tid, hidden_size, TILE_SIZE):
        val = ${inputName}[pid * hidden_size + i] + ${residualName}[pid * hidden_size + i]
        normalized = (val - mean) * rstd
        ${outputName}[pid * hidden_size + i] = ${gammaName}[i] * normalized + ${betaName}[i]


def launch_${ir.name}(${inputName}, ${residualName}, ${gammaName}, ${betaName}, ${outputName}):
    """Launch the ${ir.name} fused layernorm + residual kernel"""
    batch_size, hidden_size = ${inputName}.shape
    grid = (batch_size, 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${residualName}, ${gammaName}, ${betaName}, ${outputName}, batch_size, hidden_size))`;
}

/**
 * Generate generic multi-phase fused kernel
 * For arbitrary combinations of operations
 */
export function generateMultiPhaseFused(ir: EnhancedKernelIR & Partial<FusedIR>): string {
  const tileSize = ir.tileConfig.tileSize || 256;

  const inputName = ir.loads[0]?.source || 'input_arr';
  const outputName = ir.stores[0]?.target || 'output';

  // Generate operation chain from fusedOperations
  const operationCode = generateOperationChain(ir);

  return `import cuda_tile as ct
import cupy

TILE_SIZE = ${tileSize}

@ct.kernel
def ${ir.name}(
    ${inputName}, ${outputName},
    n: ct.Constant[int],
    tile_size: ct.Constant[int]
):
    """
    Multi-Phase Fused kernel - auto-transpiled from CUDA
    Original: ${ir.originalName}
    Confidence: ${Math.round(ir.confidence * 100)}%
    Variant: multi_phase_fused

    Fused operations: ${ir.fusedOperations?.map(op => op.type).join(' -> ') || 'unknown'}
    """
    pid = ct.bid(0)

    # Load input tile
    tile = ct.load(${inputName}, index=(pid,), shape=(tile_size,))

    # Apply fused operations
${operationCode}

    # Store result
    ct.store(${outputName}, index=(pid,), tile=result)


def launch_${ir.name}(${inputName}, ${outputName}):
    """Launch the ${ir.name} multi-phase fused kernel"""
    n = ${inputName}.shape[0]
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream, grid, ${ir.name}, (${inputName}, ${outputName}, n, TILE_SIZE))`;
}

/**
 * Helper: Generate activation function code
 */
function getActivationCode(activation: string): string {
  switch (activation.toLowerCase()) {
    case 'relu':
      return `result = ct.maximum(acc, 0.0)`;
    case 'gelu':
      return `# GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x = acc
    result = x * 0.5 * (1.0 + ct.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))`;
    case 'silu':
    case 'swish':
      return `# SiLU/Swish: x * sigmoid(x)
    result = acc * (1.0 / (1.0 + ct.exp(-acc)))`;
    case 'sigmoid':
      return `result = 1.0 / (1.0 + ct.exp(-acc))`;
    case 'tanh':
      return `result = ct.tanh(acc)`;
    case 'leaky_relu':
      return `result = ct.where(acc > 0, acc, 0.01 * acc)`;
    default:
      return `result = acc  # Identity (no activation)`;
  }
}

/**
 * Helper: Generate operation chain code
 */
function generateOperationChain(ir: EnhancedKernelIR & Partial<FusedIR>): string {
  if (!ir.fusedOperations || ir.fusedOperations.length === 0) {
    return `    result = tile`;
  }

  const lines: string[] = [];
  let currentVar = 'tile';

  for (const op of ir.fusedOperations) {
    const outputVar = `result_${op.order}`;

    switch (op.type) {
      case 'matmul':
        lines.push(`    ${outputVar} = ct.tile_matmul(${currentVar}, weight)  # Matmul phase`);
        break;
      case 'softmax':
        lines.push(`    max_val = ct.reduce(${currentVar}, op=ct.max)`);
        lines.push(`    exp_val = ct.exp(${currentVar} - max_val)`);
        lines.push(`    ${outputVar} = exp_val / ct.reduce(exp_val, op=ct.sum)  # Softmax phase`);
        break;
      case 'reduction':
        lines.push(`    ${outputVar} = ct.reduce(${currentVar}, op=ct.sum)  # Reduction phase`);
        break;
      case 'elementwise':
      default:
        lines.push(`    ${outputVar} = ${currentVar}  # Elementwise phase`);
        break;
    }

    currentVar = outputVar;
  }

  lines.push(`    result = ${currentVar}`);
  return lines.join('\n');
}

/**
 * Get the appropriate fused template generator based on variant
 */
export function getFusedGenerator(variant?: string): (ir: EnhancedKernelIR & Partial<FusedIR>) => string {
  switch (variant) {
    case 'matmul_activation':
      return generateMatmulActivation;
    case 'matmul_bias_activation':
      return generateMatmulBiasActivation;
    case 'layernorm_residual':
      return generateLayerNormResidual;
    case 'multi_phase_fused':
    default:
      return generateMultiPhaseFused;
  }
}
