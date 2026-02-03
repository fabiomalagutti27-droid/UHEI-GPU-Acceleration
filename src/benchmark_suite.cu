/**
 * PROJECT: Urban Heat Island Risk Map
 * FILE: src/benchmark_suite.cu
 * DESCRIPTION: Unified Benchmark Suite.
 * Switches kernels via Preprocessor Flags:
 * - USE_NAIVE: Runs global memory kernel
 * - USE_PINNED: Uses cudaMallocHost instead of malloc
 * - USE_CARVEOUT: Maximizes Shared Memory config
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// Dimensioni fisse del Dataset Bologna
#define WIDTH  758
#define HEIGHT 502

#ifndef BLOCK_DIM
#define BLOCK_DIM 32
#endif

#define RADIUS 1
#define SHARED_DIM (BLOCK_DIM + 2 * RADIUS)

// Utility: Flush Cache L2 per test "Cold Start"
void flushL2Cache() {
    int size = 6 * 1024 * 1024; // 6MB (T4 L2 is 4MB)
    int *d_dummy;
    CHECK_CUDA(cudaMalloc(&d_dummy, size * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_dummy, 0, size * sizeof(int)));
    CHECK_CUDA(cudaFree(d_dummy));
}

// ------------------------------------------------------------------
// 1. CPU REFERENCE (GOLD STANDARD)
// ------------------------------------------------------------------
void cpuKernel(const float* lst, const float* ndvi, const float* alb, float* out) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            float sum = 0.0f; int count = 0;
            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    int nx = x + dx; int ny = y + dy;
                    if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT) {
                        int idx = ny * WIDTH + nx;
                        sum += lst[idx] + (1.0f - ndvi[idx]) + (1.0f - alb[idx]);
                        count++;
                    }
                }
            }
            out[y * WIDTH + x] = sum / (float)count;
        }
    }
}

// ------------------------------------------------------------------
// 2. GPU KERNELS
// ------------------------------------------------------------------

// Kernel A: Naive (Global Memory Access)
__global__ void kernelNaive(float* lst, float* ndvi, float* albedo, float* output, int loops) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Loop temporale simulato per testare throughput
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
            // Scrittura solo all'ultima iterazione
            if(k == loops-1) output[row * WIDTH + col] = sum / (float)count;
        }
        // Barriera per evitare che il compilatore rimuova il loop
        if(loops > 1) __syncthreads();
    }
}

// Kernel B: Shared Memory (Tiled Access)
__global__ void kernelShared(float* lst, float* ndvi, float* albedo, float* output, int loops) {
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

    for(int k=0; k<loops; k++) {
        // 1. Cooperative Load
        for (int i = tid; i < sharedSize; i += blockSize) {
            int s_y = i / SHARED_DIM; int s_x = i % SHARED_DIM;
            int g_x = tileTopX + s_x; int g_y = tileTopY + s_y;
            
            if (g_x >= 0 && g_x < WIDTH && g_y >= 0 && g_y < HEIGHT) {
                int idx = g_y * WIDTH + g_x;
                s_lst[s_y][s_x] = lst[idx]; 
                s_ndvi[s_y][s_x] = ndvi[idx]; 
                s_alb[s_y][s_x] = albedo[idx];
            } else {
                s_lst[s_y][s_x] = 0.0f; s_ndvi[s_y][s_x] = 0.0f; s_alb[s_y][s_x] = 0.0f;
            }
        }
        __syncthreads();

        // 2. Compute
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
            if(k == loops-1) output[g_row * WIDTH + g_col] = sum / (float)count;
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------
// MAIN BENCHMARK DRIVER
// ------------------------------------------------------------------
int main() {
    size_t n = WIDTH * HEIGHT; 
    size_t bytes = n * sizeof(float);
    
    float *h_l, *h_n, *h_a, *h_o_cpu, *h_o_gpu;

    // A. Memory Allocation (Pinned vs Standard)
    #ifdef USE_PINNED
        CHECK_CUDA(cudaMallocHost((void**)&h_l, bytes));
        CHECK_CUDA(cudaMallocHost((void**)&h_n, bytes));
        CHECK_CUDA(cudaMallocHost((void**)&h_a, bytes));
        CHECK_CUDA(cudaMallocHost((void**)&h_o_cpu, bytes));
        CHECK_CUDA(cudaMallocHost((void**)&h_o_gpu, bytes));
    #else
        h_l = (float*)malloc(bytes); h_n = (float*)malloc(bytes);
        h_a = (float*)malloc(bytes); h_o_cpu = (float*)malloc(bytes);
        h_o_gpu = (float*)malloc(bytes);
    #endif

    // Init Dummy Data
    for(int i=0; i<n; i++) { h_l[i]=1.0f; h_n[i]=0.5f; h_a[i]=0.2f; }

    // B. CPU Benchmark
    clock_t c_start = clock();
    cpuKernel(h_l, h_n, h_a, h_o_cpu);
    float ms_cpu = 1000.0 * (clock() - c_start) / CLOCKS_PER_SEC;

    // GPU Setup
    float *d_l, *d_n, *d_a, *d_o;
    CHECK_CUDA(cudaMalloc(&d_l, bytes));
    CHECK_CUDA(cudaMalloc(&d_n, bytes));
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_o, bytes));
    
    // Note: Initial memcpy moved inside benchmark loop for Total Time measurement

    // Configurazione Carveout (L1/Shared Split)
    #ifdef USE_CARVEOUT
        CHECK_CUDA(cudaFuncSetAttribute(kernelShared, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    #endif

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((WIDTH + block.x - 1)/block.x, (HEIGHT + block.y - 1)/block.y);
    
    // Event pairs: kernel-only vs total (w/ transfers)
    cudaEvent_t start_kernel, stop_kernel, start_total, stop_total;
    cudaEvent_t start_batch, stop_batch;
    CHECK_CUDA(cudaEventCreate(&start_kernel));
    CHECK_CUDA(cudaEventCreate(&stop_kernel));
    CHECK_CUDA(cudaEventCreate(&start_total));
    CHECK_CUDA(cudaEventCreate(&stop_total));
    CHECK_CUDA(cudaEventCreate(&start_batch));
    CHECK_CUDA(cudaEventCreate(&stop_batch));

    // C. Single Shot Benchmark (Latenza) - 5 runs for statistical rigor
    float ms_kernel_runs[5], ms_total_runs[5];
    for (int run = 0; run < 5; run++) {
        // CRITICAL: flushL2Cache() MUST be first for Cold Start guarantee
        flushL2Cache();
        
        CHECK_CUDA(cudaEventRecord(start_total));
        
        // Transfer data to GPU (moved inside loop for total time measurement)
        CHECK_CUDA(cudaMemcpy(d_l, h_l, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_n, h_n, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        
        CHECK_CUDA(cudaEventRecord(start_kernel));
        #ifdef USE_NAIVE
            kernelNaive<<<grid, block>>>(d_l, d_n, d_a, d_o, 1);
        #else
            kernelShared<<<grid, block>>>(d_l, d_n, d_a, d_o, 1);
        #endif
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop_kernel));
        
        // Transfer result back to host
        CHECK_CUDA(cudaMemcpy(h_o_gpu, d_o, bytes, cudaMemcpyDeviceToHost));
        
        CHECK_CUDA(cudaEventRecord(stop_total));
        CHECK_CUDA(cudaEventSynchronize(stop_total));
        
        cudaEventElapsedTime(&ms_kernel_runs[run], start_kernel, stop_kernel);
        cudaEventElapsedTime(&ms_total_runs[run], start_total, stop_total);
    }
    
    // Calculate means
    float ms_kernel = 0.0f, ms_total = 0.0f;
    for (int run = 0; run < 5; run++) {
        ms_kernel += ms_kernel_runs[run];
        ms_total += ms_total_runs[run];
    }
    ms_kernel /= 5.0f;
    ms_total /= 5.0f;
    float ms_transfer = ms_total - ms_kernel;
    if (ms_transfer < 0.0f) ms_transfer = 0.0f;

    // Rigorous Validation with EPSILON check (h_o_gpu already contains last run's result)
    // Note: USE_NAIVE may have slightly higher error due to different computation order
    const float EPSILON = 1.0e-5f;
    int errors = 0;
    double total_diff = 0.0;
    float max_error = 0.0f;
    
    for(int i = 0; i < n; i++) {
        float diff = h_o_cpu[i] - h_o_gpu[i];
        float abs_diff = std::abs(diff);
        
        if (abs_diff > EPSILON) {
            errors++;
        }
        
        total_diff += diff * diff;
        max_error = std::max(max_error, abs_diff);
    }
    float rmse = std::sqrt(total_diff / n);
    
    // Validation Report
    if (errors == 0) {
        std::cout << "TEST PASSED: All pixels within epsilon tolerance" << std::endl;
    } else {
        std::cout << "TEST FAILED: " << errors << " pixels exceed epsilon tolerance" << std::endl;
        #ifdef USE_NAIVE
            std::cout << "Note: Naive kernel may have acceptable rounding differences" << std::endl;
        #endif
    }
    std::cout << "Max Error: " << max_error << " | RMSE: " << rmse << std::endl;

    // D. Batch Benchmark (Throughput - 1000 iterazioni) - 5 runs for statistical rigor
    float ms_batch_runs[5];
    for (int run = 0; run < 5; run++) {
        flushL2Cache();
        CHECK_CUDA(cudaEventRecord(start_batch));
        #ifdef USE_NAIVE
            kernelNaive<<<grid, block>>>(d_l, d_n, d_a, d_o, 1000);
        #else
            kernelShared<<<grid, block>>>(d_l, d_n, d_a, d_o, 1000);
        #endif
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop_batch));
        CHECK_CUDA(cudaEventSynchronize(stop_batch));
        cudaEventElapsedTime(&ms_batch_runs[run], start_batch, stop_batch);
    }
    // Calculate mean
    float ms_batch = 0.0f;
    for (int run = 0; run < 5; run++) ms_batch += ms_batch_runs[run];
    ms_batch /= 5.0f;

    // Output with configuration flags for Python parsing
    std::cout << "BENCHMARK,";
    
    // Kernel type
    #ifdef USE_NAIVE
        std::cout << "NAIVE,";
    #else
        std::cout << "SHARED,";
    #endif
    
    // Memory type
    #ifdef USE_PINNED
        std::cout << "PINNED,";
    #else
        std::cout << "PAGEABLE,";
    #endif
    
    // Carveout configuration
    #ifdef USE_CARVEOUT
        std::cout << "CARVEOUT,";
    #else
        std::cout << "DEFAULT,";
    #endif
    
    // Performance metrics: CPU, KernelOnly, TotalWithTransfers, TransferOnly, BatchGPU, RMSE, MaxError
    std::cout << ms_cpu << "," << ms_kernel << "," << ms_total << "," << ms_transfer << "," << ms_batch << "," << rmse << "," << max_error << std::endl;
    
    // Cleanup (Simplified)
    CHECK_CUDA(cudaFree(d_l));
    CHECK_CUDA(cudaFree(d_n));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_o));
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}