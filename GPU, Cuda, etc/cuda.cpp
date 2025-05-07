#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Declares a function that runs on GPU
__global__ void vectorAdd(int *a, int *b, int *c, int n) {

    // Calculates thread's global ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

int main() {
    int n = 1<<20; // 1M elements
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    size_t size = n * sizeof(int);

    // Allocate host memory
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Copy to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    // Launch the GPU kernel with a grid of threads
    vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy back
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify 
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n". a[i], b[i], c[i]);
    }

    // Free memory
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}