#include <vector>
#include <cuda.h>
#include <stdio.h>
#include <chrono>

#define CHECK() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if (e!=cudaSuccess) {                                              \
         printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
         exit(0); \
    }                                                                 \
}

__global__ void matMulKernel(float* d_A, float* d_B, float* d_C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int gid_1d = row * k + col;

    if (row < m && col < k)
    {
        float result = 0;
        for (size_t i = 0; i < n; ++i)
        {
            result += d_A[row * n + i] * d_B[i * k + col];
        }
        d_C[gid_1d] = result;
    }
}

void matMulCuda(float* h_A, float* h_B, float* h_C, int m, int n, int k)
{

    float* d_A;
    float* d_B;
    float* d_C;
    
    cudaMalloc((void**)&d_A, m * n * sizeof(float));
    CHECK();
    cudaMalloc((void**)&d_B, n * k * sizeof(float));
    CHECK();
    cudaMalloc((void**)&d_C, m * k * sizeof(float));
    CHECK();

    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    CHECK();
    cudaMemcpy(d_B, h_B, n * k * sizeof(float), cudaMemcpyHostToDevice);
    CHECK();

    dim3 gridSize((k - 1) / 32 + 1, (m - 1) / 32 + 1, 1);
    dim3 blockSize(32, 32, 1);
    matMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
    CHECK();

    cudaMemcpy(h_C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    CHECK();

    // for (size_t i = 0; i < m; ++i)
    // {
    //     for (size_t j = 0; j < k; ++j)
    //     {
    //         printf("%.2f ", h_C[i * k + j]);
    //     }
    //     printf("\n");
    // }
}

void matMulCpu(float* h_A, float* h_B, float* h_C, int m, int n, int k)
{
    for (size_t row = 0; row < m; ++row)
    {
        for (size_t col = 0; col < k; ++col)
        {
            int total = 0;
            for (size_t i = 0; i < n; ++i)
            {
                total += h_A[row * n + i] * h_B[i * k + col];
            }
            h_C[row * k + col] = total;
        }
    }
}

int main()
{
    int m = 20;
    int n = 33;
    int k = 65;

    std::vector<float> h_A(m * n);
    for (size_t i = 0; i < m * n; ++i)
    {
        h_A[i] = 1;
    }
    std::vector<float> h_B(n * k, 2);
    for (size_t i = 0; i < n * k; ++i)
    {
        h_B[i] = 2;
    }
    std::vector<float> h_CCuda(m * k);
    std::vector<float> h_CCpu(m * k);
    auto t1 = std::chrono::high_resolution_clock::now();
    matMulCuda(h_A.data(), h_B.data(), h_CCuda.data(), m, n, k);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    printf("Cuda execution time take %.2fms\n", ms_double.count());

    t1 = std::chrono::high_resolution_clock::now();
    matMulCpu(h_A.data(), h_B.data(), h_CCpu.data(), m, n, k);
    t2 = std::chrono::high_resolution_clock::now();
    ms_double = t2 - t1;
    printf("CPU execution time take %.2fms\n", ms_double.count());

    for (size_t i = 0; i < m * k; ++i)
    {
        if (h_CCuda[i] != h_CCpu[i])
        {
            printf("False at i = %zu with %.2f (GPU side) and %.2f (CPU side)\n", i, h_CCuda[i], h_CCpu[i]);
            exit(0);
        }
    }
    printf("Passed\n");


    return 0;
}
