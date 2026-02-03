/**
 * PROJECT: Urban Heat Island Risk Map
 * FILE: src/profiling_harness.cu
 * DESCRIPTION: Simplified kernel harness for Nsight Compute Profiling.
 * Includes 'loops' parameter to extend kernel duration for accurate sampling.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

#define WIDTH  758
#define HEIGHT 502
#define BLOCK_DIM 32
#define RADIUS 1
#define SHARED_DIM (BLOCK_DIM + 2 * RADIUS)

// --- KERNEL 1: NAIVE (Baseline) ---
// Global Memory intensive implementation
__global__ void profileNaive(float* lst, float* ndvi, float* albedo, float* output, int loops) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    for (int k = 0; k < loops; k++) {
        if (col < WIDTH && row < HEIGHT) {
            float sum = 0.0f; int count = 0;
            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    int c = col + dx; int r = row + dy;
                    if (c >= 0 && c < WIDTH && r >= 0 && r < HEIGHT) {
                        int idx = r * WIDTH + c;
                        sum += lst[idx] + (1.0f - ndvi[idx]) + (1.0f - albedo[idx]);
                        count++;
                    }
                }
            }
            // Write only on last iteration to prevent compiler optimizing away the loop
            if (k == loops - 1) output[row * WIDTH + col] = sum / (float)count;
        }
        // Barrier to simulate temporal dependency
        if (loops > 1) __syncthreads();
    }
}

// --- KERNEL 2: SHARED (Optimized Logic) ---
// Tiled Memory Access implementation
__global__ void profileShared(float* lst, float* ndvi, float* albedo, float* output, int loops) {
    __shared__ float s_lst[SHARED_DIM][SHARED_DIM];
    __shared__ float s_ndvi[SHARED_DIM][SHARED_DIM];
    __shared__ float s_alb[SHARED_DIM][SHARED_DIM];

    int tileTopX = blockIdx.x * blockDim.x - RADIUS;
    int tileTopY = blockIdx.y * blockDim.y - RADIUS;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    int sharedSize = SHARED_DIM * SHARED_DIM;
    
    int g_col = blockIdx.x * blockDim.x + threadIdx.x;
    int g_row = blockIdx.y * blockDim.y + threadIdx.y;

    for (int k = 0; k < loops; k++) {
        // Cooperative Loading
        for (int i = tid; i < sharedSize; i += blockSize) {
            int s_y = i / SHARED_DIM; int s_x = i % SHARED_DIM;
            int g_x = tileTopX + s_x; int g_y = tileTopY + s_y;
            if (g_x >= 0 && g_x < WIDTH && g_y >= 0 && g_y < HEIGHT) {
                int idx = g_y * WIDTH + g_x;
                s_lst[s_y][s_x] = lst[idx]; s_ndvi[s_y][s_x] = ndvi[idx]; s_alb[s_y][s_x] = albedo[idx];
            } else {
                s_lst[s_y][s_x] = 0.0f; s_ndvi[s_y][s_x] = 0.0f; s_alb[s_y][s_x] = 0.0f;
            }
        }
        __syncthreads();

        // Compute
        if (g_col < WIDTH && g_row < HEIGHT) {
            float sum = 0.0f; int count = 0;
            int s_c = threadIdx.x + RADIUS; int s_r = threadIdx.y + RADIUS;
            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    int nx = g_col + dx; int ny = g_row + dy;
                    if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT) {
                        sum += s_lst[s_r+dy][s_c+dx] + (1.0f - s_ndvi[s_r+dy][s_c+dx]) + (1.0f - s_alb[s_r+dy][s_c+dx]);
                        count++;
                    }
                }
            }
            if (k == loops - 1) output[g_row * WIDTH + g_col] = sum / (float)count;
        }
        __syncthreads();
    }
}

int main(int argc, char **argv) {
    // Mode Argument: 0 = Naive, 1 = Shared
    int mode = 0; 
    if (argc > 1) mode = atoi(argv[1]);

    size_t n = WIDTH*HEIGHT; size_t bytes = n*sizeof(float);
    int loops = 100; // Stabilize profiler

    float *d_l, *d_n, *d_a, *d_o;
    CHECK_CUDA(cudaMalloc(&d_l, bytes)); 
    CHECK_CUDA(cudaMalloc(&d_n, bytes)); 
    CHECK_CUDA(cudaMalloc(&d_a, bytes)); 
    CHECK_CUDA(cudaMalloc(&d_o, bytes));
    
    // Init Dummy Data (Profiling focuses on performance, not correctness)
    CHECK_CUDA(cudaMemset(d_l, 0, bytes)); 
    CHECK_CUDA(cudaMemset(d_n, 0, bytes)); 
    CHECK_CUDA(cudaMemset(d_a, 0, bytes));

    // Carveout Configuration for Fair Comparison
    // Naive: Maximize L1 Cache (benefits global memory accesses)
    // Shared: Maximize Shared Memory (needs explicit shared memory space)
    CHECK_CUDA(cudaFuncSetAttribute(profileNaive, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1));
    CHECK_CUDA(cudaFuncSetAttribute(profileShared, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((WIDTH + block.x - 1)/block.x, (HEIGHT + block.y - 1)/block.y);

    if (mode == 0) {
        profileNaive<<<grid, block>>>(d_l, d_n, d_a, d_o, loops);
    } else {
        profileShared<<<grid, block>>>(d_l, d_n, d_a, d_o, loops);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaFree(d_l); cudaFree(d_n); cudaFree(d_a); cudaFree(d_o);
    cudaDeviceReset();
    return 0;
}