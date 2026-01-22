#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 64
#define WARP_SIZE 32

// Online softmax tracking for numerical stability
struct SoftmaxState {
    float max_val;
    float sum_exp;
};

// Update running statistics for online softmax
__device__ __forceinline__ void update_softmax_state(
    SoftmaxState& state, 
    float new_val
) {
    float old_max = state.max_val;
    state.max_val = fmaxf(state.max_val, new_val);
    state.sum_exp = state.sum_exp * expf(old_max - state.max_val) + 
                    expf(new_val - state.max_val);
}

// Warp-level reduction for maximum
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// FlashAttention forward kernel
__global__ void flash_attention_forward_kernel(
    const float* Q,
    const float* K, 
    const float* V,
    float* O,
    float* L,  // Log-sum-exp for backward
    float* M,  // Max values for backward
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal
) {
    // Block and thread indices
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory for tiles
    __shared__ float Q_tile[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float K_tile[BLOCK_SIZE_N][BLOCK_SIZE_K];
    __shared__ float V_tile[BLOCK_SIZE_N][BLOCK_SIZE_K];
    __shared__ float S_tile[BLOCK_SIZE_M][BLOCK_SIZE_N];
    
    // Register storage for output accumulation
    float O_reg[BLOCK_SIZE_M] = {0.0f};
    
    // Softmax state for each Q row this thread handles
    SoftmaxState state[BLOCK_SIZE_M];
    for (int i = 0; i < BLOCK_SIZE_M; i++) {
        state[i].max_val = -INFINITY;
        state[i].sum_exp = 0.0f;
    }
    
    int q_start = q_block_idx * BLOCK_SIZE_M;
    int total_k_blocks = (seq_len + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    
    // Iterate over K, V blocks
    for (int k_block_idx = 0; k_block_idx < total_k_blocks; k_block_idx++) {
        int k_start = k_block_idx * BLOCK_SIZE_N;
        
        // Load Q tile to shared memory
        if (tid < BLOCK_SIZE_K) {
            for (int i = 0; i < BLOCK_SIZE_M; i++) {
                int q_row = q_start + i;
                if (q_row < seq_len) {
                    int q_idx = ((batch_idx * num_heads + head_idx) * seq_len + q_row) 
                               * head_dim + tid;
                    Q_tile[i][tid] = Q[q_idx];
                } else {
                    Q_tile[i][tid] = 0.0f;
                }
            }
        }
        
        // Load K tile to shared memory
        if (tid < BLOCK_SIZE_K) {
            for (int j = 0; j < BLOCK_SIZE_N; j++) {
                int k_row = k_start + j;
                if (k_row < seq_len) {
                    int k_idx = ((batch_idx * num_heads + head_idx) * seq_len + k_row) 
                               * head_dim + tid;
                    K_tile[j][tid] = K[k_idx];
                } else {
                    K_tile[j][tid] = 0.0f;
                }
            }
        }
        
        // Load V tile to shared memory
        if (tid < BLOCK_SIZE_K) {
            for (int j = 0; j < BLOCK_SIZE_N; j++) {
                int v_row = k_start + j;
                if (v_row < seq_len) {
                    int v_idx = ((batch_idx * num_heads + head_idx) * seq_len + v_row) 
                               * head_dim + tid;
                    V_tile[j][tid] = V[v_idx];
                } else {
                    V_tile[j][tid] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Compute S = Q @ K^T (attention scores)
        for (int i = 0; i < BLOCK_SIZE_M; i++) {
            for (int j = tid; j < BLOCK_SIZE_N; j += blockDim.x) {
                float score = 0.0f;
                for (int k = 0; k < BLOCK_SIZE_K && k < head_dim; k++) {
                    score += Q_tile[i][k] * K_tile[j][k];
                }
                score *= scale;
                
                // Apply causal mask if needed
                int q_pos = q_start + i;
                int k_pos = k_start + j;
                if (causal && k_pos > q_pos) {
                    score = -INFINITY;
                }
                
                S_tile[i][j] = score;
            }
        }
        
        __syncthreads();
        
        // Update softmax statistics and compute attention-weighted sum
        for (int i = 0; i < BLOCK_SIZE_M; i++) {
            int q_row = q_start + i;
            if (q_row >= seq_len) continue;
            
            // Find max in this block for current Q row
            float block_max = -INFINITY;
            for (int j = 0; j < BLOCK_SIZE_N; j++) {
                block_max = fmaxf(block_max, S_tile[i][j]);
            }
            
            // Update global softmax state
            float old_max = state[i].max_val;
            state[i].max_val = fmaxf(state[i].max_val, block_max);
            float correction = expf(old_max - state[i].max_val);
            
            // Correct previous accumulation
            O_reg[i] *= correction;
            state[i].sum_exp *= correction;
            
            // Accumulate new values
            for (int d = tid; d < head_dim; d += blockDim.x) {
                float acc = 0.0f;
                for (int j = 0; j < BLOCK_SIZE_N; j++) {
                    int k_pos = k_start + j;
                    if (k_pos < seq_len) {
                        float exp_score = expf(S_tile[i][j] - state[i].max_val);
                        acc += exp_score * V_tile[j][d];
                        
                        if (d == 0) {
                            state[i].sum_exp += exp_score;
                        }
                    }
                }
                
                if (d < head_dim) {
                    O_reg[i] += acc;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write final output with softmax normalization
    for (int i = 0; i < BLOCK_SIZE_M; i++) {
        int q_row = q_start + i;
        if (q_row < seq_len) {
            for (int d = tid; d < head_dim; d += blockDim.x) {
                int o_idx = ((batch_idx * num_heads + head_idx) * seq_len + q_row) 
                           * head_dim + d;
                O[o_idx] = O_reg[i] / state[i].sum_exp;
            }
            
            // Store statistics for backward pass
            if (tid == 0) {
                int stat_idx = (batch_idx * num_heads + head_idx) * seq_len + q_row;
                L[stat_idx] = logf(state[i].sum_exp) + state[i].max_val;
                M[stat_idx] = state[i].max_val;
            }
        }
    }
}

// Forward pass wrapper
torch::Tensor flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal
) {
    const auto batch_size = Q.size(0);
    const auto num_heads = Q.size(1);
    const auto seq_len = Q.size(2);
    const auto head_dim = Q.size(3);
    
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({batch_size, num_heads, seq_len}, Q.options());
    auto M = torch::zeros({batch_size, num_heads, seq_len}, Q.options());
    
    const at::cuda::OptionalCUDAGuard device_guard(Q.device());
    
    dim3 grid(
        (seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
        num_heads,
        batch_size
    );
    dim3 block(256);
    
    flash_attention_forward_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        L.data_ptr<float>(),
        M.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale,
        causal
    );
    
    return O;
}

// Backward kernel (simplified version)
__global__ void flash_attention_backward_kernel(
    const float* Q,
    const float* K,
    const float* V,
    const float* O,
    const float* dO,
    const float* L,
    float* dQ,
    float* dK,
    float* dV,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal
) {
    // Simplified backward pass - recomputes attention on the fly
    // Full implementation would include all gradient computations
    
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    // This is a placeholder - full backward requires careful
    // implementation of gradient flow through softmax
}

torch::Tensor flash_attention_backward(
    torch::Tensor dO,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor L,
    bool causal
) {
    // Return gradient tensors
    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);
    
    // Note: Full backward implementation omitted for brevity
    // Would include proper gradient computation
    
    return torch::stack({dQ, dK, dV});
}
