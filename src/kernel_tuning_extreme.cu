/**
 * PROJECT: Urban Heat Island Risk Map
 * FILE: src/kernel_tuning_extreme.cu
 * DESCRIPTION: "Deathmatch" Benchmark.
 * Compares standard Naive kernel vs. Extreme Optimized Naive kernel
 * to identify L2 Cache bandwidth limits vs Compute limits.
 * 
 * IMPORTANT NOTE - ALGORITHM CHARACTERISTICS:
 * This stencil algorithm is MEMORY-BOUND with low arithmetic intensity (~1 FLOP/byte).
 * The T4 GPU has L2 cache bandwidth of ~260 GB/s and compute capability of ~8.1 TFLOPs.
 * With 1 FLOP/byte, the maximum achievable performance is ~260 GFLOP/s, well below
 * the compute ceiling. Therefore, low-level optimizations (__restrict__, #pragma unroll,
 * __launch_bounds__) will have LIMITED impact (<5-10% improvement) since the bottleneck
 * is memory bandwidth, NOT instruction throughput or register pressure.
 * 
 * The primary optimization for this class of problems is SHARED MEMORY tiling, which
 * reduces global memory accesses by ~90%, as demonstrated in benchmark_suite.cu.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

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

// --- UTILS: CACHE FLUSH ---
void flushL2Cache() {
    int size = 6 * 1024 * 1024; 
    int *d_dummy;
    CHECK_CUDA(cudaMalloc(&d_dummy, size * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_dummy, 0, size * sizeof(int)));
    CHECK_CUDA(cudaFree(d_dummy));
}

// ---------------------------------------------------------
// 1. KERNEL STANDARD (Baseline)
// ---------------------------------------------------------
__global__ void kernelNaiveStandard(float* lst, float* ndvi, float* albedo, float* output, int loops) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    for(int k=0; k<loops; k++) {
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
            if(k == loops-1) output[row * WIDTH + col] = sum / (float)count;
        }
        if(loops > 1) __syncthreads();
    }
}

// ---------------------------------------------------------
// 2. KERNEL EXTREME (Instruction Level Optimization)
// ---------------------------------------------------------
// __launch_bounds__: Hint to compiler to limit register usage for high occupancy
// __restrict__: Hint that pointers do not alias (enables better read-only caching)
__global__ void __launch_bounds__(1024) kernelNaiveExtreme(
    const float* __restrict__ lst, 
    const float* __restrict__ ndvi, 
    const float* __restrict__ albedo, 
    float* __restrict__ output, 
    int loops) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_out = row * WIDTH + col;

    for(int k=0; k<loops; k++) {
        if (col < WIDTH && row < HEIGHT) {
            float sum = 0.0f; 
            int count = 0;

            // #pragma unroll: Forces loop unrolling to remove branch instructions
            // Increases instruction throughput by keeping the pipeline full
            #pragma unroll
            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                #pragma unroll
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    int c = col + dx; 
                    int r = row + dy;
                    
                    if (c >= 0 && c < WIDTH && r >= 0 && r < HEIGHT) {
                        int idx = r * WIDTH + c;
                        sum += lst[idx] + (1.0f - ndvi[idx]) + (1.0f - albedo[idx]);
                        count++;
                    }
                }
            }
            if(k == loops-1) output[idx_out] = sum / (float)count;
        }
        if (loops > 1) __syncthreads();
    }
}

// ---------------------------------------------------------
// MAIN DRIVER
// ---------------------------------------------------------
int main() {
    size_t n = WIDTH * HEIGHT;
    size_t bytes = n * sizeof(float);
    
    // Allocazione Dummy (Non carichiamo i file veri per velocità, ci interessa il tempo di calcolo)
    float *d_l, *d_n, *d_a, *d_o;
    CHECK_CUDA(cudaMalloc(&d_l, bytes)); 
    CHECK_CUDA(cudaMalloc(&d_n, bytes)); 
    CHECK_CUDA(cudaMalloc(&d_a, bytes)); 
    CHECK_CUDA(cudaMalloc(&d_o, bytes));
    CHECK_CUDA(cudaMemset(d_l, 0, bytes)); 

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((WIDTH + block.x - 1)/block.x, (HEIGHT + block.y - 1)/block.y);

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms_std = 0, ms_ext = 0;

    // --- ROUND 1: STANDARD ---
    flushL2Cache();
    cudaEventRecord(start);
    kernelNaiveStandard<<<grid, block>>>(d_l, d_n, d_a, d_o, 1000);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_std, start, stop);

    // --- ROUND 2: EXTREME ---
    flushL2Cache();
    cudaEventRecord(start);
    kernelNaiveExtreme<<<grid, block>>>(d_l, d_n, d_a, d_o, 1000);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_ext, start, stop);

    // REPORT
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Kernel Standard: " << std::fixed << std::setprecision(3) << ms_std << " ms" << std::endl;
    std::cout << "Kernel Extreme:  " << std::fixed << std::setprecision(3) << ms_ext << " ms" << std::endl;
    
    float diff = ms_std - ms_ext;
    float pct = (diff / ms_std) * 100.0f;
    
    std::cout << "--------------------------------------------------" << std::endl;
    if (diff > 0) std::cout << "MIGLIORAMENTO: -" << diff << " ms (" << pct << "%)" << std::endl;
    else          std::cout << "PEGGIORAMENTO: +" << -diff << " ms" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    cudaDeviceReset();
    return 0;
}