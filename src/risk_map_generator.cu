/**
 * PROJECT: Urban Heat Island Risk Map (CUDA Optimized)
 * FILE: risk_map_generator.cu
 * DESCRIPTION: Main algorithmic implementation with Shared Memory tiling.
 * Includes CPU baseline for correctness verification and RMSE calculation.
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>

// --- CONFIGURATION ---
#define WIDTH  758
#define HEIGHT 502
#define RADIUS 1

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

// 0. BASELINE CPU
void heatMapCPU(const std::vector<float>& lst, const std::vector<float>& ndvi,
                const std::vector<float>& alb, std::vector<float>& out, int w, int h) {
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

// 1. KERNEL SHARED
__global__ void heatMapShared(float* lst, float* ndvi, float* albedo, float* output, int width, int height) {
    __shared__ float s_lst[SHARED_DIM][SHARED_DIM];
    __shared__ float s_ndvi[SHARED_DIM][SHARED_DIM];
    __shared__ float s_alb[SHARED_DIM][SHARED_DIM];

    int tileTopX = blockIdx.x * blockDim.x - RADIUS;
    int tileTopY = blockIdx.y * blockDim.y - RADIUS;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    int sharedSize = SHARED_DIM * SHARED_DIM;

    // Cooperative Loading
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

    // Computation
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

void readBinary(const std::string& path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    if(!file) { std::cerr << "ERR: Missing " << path << std::endl; exit(1); }
    file.read(reinterpret_cast<char*>(data.data()), data.size()*sizeof(float));
}

bool verifyResults(const std::vector<float>& cpu_out, const std::vector<float>& gpu_out,
                   size_t n, float& rmse, float& max_error) {
    const float EPSILON = 1.0e-5f;
    int errors = 0;
    double total_diff = 0.0;
    max_error = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float diff = cpu_out[i] - gpu_out[i];
        float abs_diff = std::abs(diff);

        if (abs_diff > EPSILON) {
            errors++;
        }

        total_diff += diff * diff;
        max_error = std::max(max_error, abs_diff);
    }
    rmse = std::sqrt(total_diff / n);

    if (errors == 0) {
        std::cout << "TEST PASSED: All pixels within epsilon tolerance" << std::endl;
    } else {
        std::cout << "TEST FAILED: " << errors << " pixels exceed epsilon tolerance" << std::endl;
    }
    std::cout << "Max Error: " << max_error << " | RMSE: " << rmse << std::endl;

    return errors == 0;
}

int main() {
    size_t n = WIDTH*HEIGHT; 
    size_t bytes = n*sizeof(float);

    // Host Allocation
    std::vector<float> h_l(n), h_n(n), h_a(n);
    std::vector<float> h_gpu_out(n);
    std::vector<float> h_cpu_out(n);

    readBinary("input_lst.bin", h_l);
    readBinary("input_ndvi.bin", h_n);
    readBinary("input_albedo.bin", h_a);

    // 1. CPU RUN (Baseline)
    clock_t c_start = clock();
    heatMapCPU(h_l, h_n, h_a, h_cpu_out, WIDTH, HEIGHT);
    float ms_cpu = 1000.0 * (clock() - c_start) / CLOCKS_PER_SEC;

    // 2. GPU RUN
    float *d_l, *d_n, *d_a, *d_o;
    CHECK(cudaMalloc(&d_l, bytes)); CHECK(cudaMalloc(&d_n, bytes)); 
    CHECK(cudaMalloc(&d_a, bytes)); CHECK(cudaMalloc(&d_o, bytes));

    CHECK(cudaMemcpy(d_l, h_l.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_n, h_n.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((WIDTH + block.x - 1)/block.x, (HEIGHT + block.y - 1)/block.y);

    cudaEvent_t start, stop; 
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    heatMapShared<<<grid, block>>>(d_l, d_n, d_a, d_o, WIDTH, HEIGHT);
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    float ms_gpu = 0; 
    cudaEventElapsedTime(&ms_gpu, start, stop);

    // 3. VALIDATION
    CHECK(cudaMemcpy(h_gpu_out.data(), d_o, bytes, cudaMemcpyDeviceToHost));

    float rmse = 0.0f;
    float max_error = 0.0f;
    verifyResults(h_cpu_out, h_gpu_out, n, rmse, max_error);

    // 4. OUTPUT FORMATTATO PER PYTHON
    // Format: CSV_DATA, BLOCK_DIM, CPU_MS, GPU_MS, RMSE, MAX_ERROR
    std::cout << "CSV_DATA," << BLOCK_DIM << "," << ms_cpu << "," << ms_gpu << "," << rmse << "," << max_error << std::endl;

    // Save Output for Visualization
    std::ofstream outfile("output_risk_map.bin", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(h_gpu_out.data()), bytes);
    outfile.close();

    cudaFree(d_l); cudaFree(d_n); cudaFree(d_a); cudaFree(d_o);
    CHECK(cudaDeviceReset());
    return 0;
}