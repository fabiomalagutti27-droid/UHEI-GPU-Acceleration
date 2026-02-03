/**
 * PROJECT: Urban Heat Island Risk Map
 * FILE: src/gpu_optimized_test.cu
 * DESCRIPTION: System-level optimization test.
 * Features: Pinned Memory (Zero-Copy), L2 Cache Flushing, Cooperative Loading.
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>

#define WIDTH  758
#define HEIGHT 502
#define RADIUS 1

// Parametro dinamico da compilatore (default 32 se non specificato)
#ifndef BLOCK_DIM
#define BLOCK_DIM 32 
#endif

#define SHARED_DIM (BLOCK_DIM + 2 * RADIUS)

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// --- SYSTEM UTILS: CACHE FLUSH ---
// Fondamentale per benchmark scientifici "Cold Start"
void flushL2Cache() {
    int size = 6 * 1024 * 1024; // 6MB (T4 L2 is 4MB)
    int *d_dummy;
    CHECK(cudaMalloc(&d_dummy, size * sizeof(int)));
    CHECK(cudaMemset(d_dummy, 0, size * sizeof(int)));
    CHECK(cudaFree(d_dummy));
}

// --- OPTIMIZATION: PINNED MEMORY READER ---
void readBinaryToPinned(const std::string& path, float* pinned_ptr, size_t size) {
    std::ifstream file(path, std::ios::binary);
    if(!file) { std::cerr << "ERR: Missing " << path << std::endl; exit(1); }
    file.read(reinterpret_cast<char*>(pinned_ptr), size * sizeof(float));
}

// 0. CPU Reference (per validazione RMSE)
void heatMapCPU(const float* lst, const float* ndvi, const float* alb, float* out, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum_risk = 0.0f; int count = 0;
            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    int ny = y + dy; int nx = x + dx;
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                        int idx = ny * w + nx;
                        sum_risk += lst[idx] + (1.0f - ndvi[idx]) + (1.0f - alb[idx]);
                        count++;
                    }
                }
            }
            out[y * w + x] = sum_risk / (float)count;
        }
    }
}

// 1. Kernel Shared Ottimizzato
__global__ void heatMapShared(float* lst, float* ndvi, float* albedo, float* output, int width, int height) {
    __shared__ float s_lst[SHARED_DIM][SHARED_DIM];
    __shared__ float s_ndvi[SHARED_DIM][SHARED_DIM];
    __shared__ float s_alb[SHARED_DIM][SHARED_DIM];

    int tileTopX = blockIdx.x * blockDim.x - RADIUS;
    int tileTopY = blockIdx.y * blockDim.y - RADIUS;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    int sharedSize = SHARED_DIM * SHARED_DIM;

    // Caricamento Cooperativo
    for (int i = tid; i < sharedSize; i += blockSize) {
        int s_y = i / SHARED_DIM;
        int s_x = i % SHARED_DIM;
        int g_x = tileTopX + s_x;
        int g_y = tileTopY + s_y;

        if (g_x >= 0 && g_x < width && g_y >= 0 && g_y < height) {
            int idx = g_y * width + g_x;
            s_lst[s_y][s_x] = lst[idx];
            s_ndvi[s_y][s_x] = ndvi[idx];
            s_alb[s_y][s_x] = albedo[idx];
        } else {
            s_lst[s_y][s_x] = 0.0f; s_ndvi[s_y][s_x] = 0.0f; s_alb[s_y][s_x] = 0.0f;
        }
    }
    __syncthreads();

    // Calcolo
    int g_col = blockIdx.x * blockDim.x + threadIdx.x;
    int g_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (g_col < width && g_row < height) {
        float sum = 0.0f; int count = 0;
        int s_c = threadIdx.x + RADIUS;
        int s_r = threadIdx.y + RADIUS;

        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                int neighbor_x = g_col + dx;
                int neighbor_y = g_row + dy;
                if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                    sum += s_lst[s_r+dy][s_c+dx] + (1.0f - s_ndvi[s_r+dy][s_c+dx]) + (1.0f - s_alb[s_r+dy][s_c+dx]);
                    count++;
                }
            }
        }
        output[g_row * width + g_col] = sum / (float)count;
    }
}

int main() {
    size_t n = WIDTH*HEIGHT; size_t bytes = n*sizeof(float);

    // 1. PINNED MEMORY ALLOCATION (cudaMallocHost)
    // Velocizza il trasferimento PCIe
    float *h_l, *h_n, *h_a, *h_out;
    CHECK(cudaMallocHost((void**)&h_l, bytes));
    CHECK(cudaMallocHost((void**)&h_n, bytes));
    CHECK(cudaMallocHost((void**)&h_a, bytes));
    CHECK(cudaMallocHost((void**)&h_out, bytes));

    readBinaryToPinned("input_lst.bin", h_l, n);
    readBinaryToPinned("input_ndvi.bin", h_n, n);
    readBinaryToPinned("input_albedo.bin", h_a, n);

    // CPU Validation
    std::vector<float> h_cpu_out(n);
    heatMapCPU(h_l, h_n, h_a, h_cpu_out.data(), WIDTH, HEIGHT);

    float *d_l, *d_n, *d_a, *d_o;
    CHECK(cudaMalloc(&d_l, bytes)); CHECK(cudaMalloc(&d_n, bytes));
    CHECK(cudaMalloc(&d_a, bytes)); CHECK(cudaMalloc(&d_o, bytes));

    CHECK(cudaMemcpy(d_l, h_l, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_n, h_n, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // Carveout configuration for T4/L4
    CHECK(cudaFuncSetAttribute(heatMapShared, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((WIDTH + block.x - 1)/block.x, (HEIGHT + block.y - 1)/block.y);

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

    // FLUSH CACHE (Per test rigoroso)
    flushL2Cache();

    cudaEventRecord(start);
    heatMapShared<<<grid, block>>>(d_l, d_n, d_a, d_o, WIDTH, HEIGHT);
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    float ms_opt = 0; cudaEventElapsedTime(&ms_opt, start, stop);
    CHECK(cudaMemcpy(h_out, d_o, bytes, cudaMemcpyDeviceToHost));

    // Rigorous Validation with EPSILON check
    const float EPSILON = 1.0e-5f;
    int errors = 0;
    double total_diff = 0.0;
    float max_error = 0.0f;
    
    for(size_t i = 0; i < n; i++) {
        float diff = h_cpu_out[i] - h_out[i];
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
    }
    std::cout << "Max Error: " << max_error << " | RMSE: " << rmse << std::endl;

    // Output formattato per il parsing Python
    // Format: TEST_RESULT, BLOCK_DIM, GPU_MS, RMSE, MAX_ERROR
    std::cout << "TEST_RESULT," << BLOCK_DIM << "," << ms_opt << "," << rmse << "," << max_error << std::endl;

    cudaFreeHost(h_l); cudaFreeHost(h_n); cudaFreeHost(h_a); cudaFreeHost(h_out);
    cudaFree(d_l); cudaFree(d_n); cudaFree(d_a); cudaFree(d_o);
    CHECK(cudaDeviceReset());
    return 0;
}