#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int numRowsX = 1024;
    int numColsX = 512;
    int numColsY = 2048;

    float* hostX = (float*)malloc(numRowsX * numColsX * sizeof(float));
    float* hostY = (float*)malloc(numColsX * numColsY * sizeof(float));
    float* hostZ = (float*)malloc(numRowsX * numColsY * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < numRowsX * numColsX; i++) {
        hostX[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < numColsX * numColsY; i++) {
        hostY[i] = rand() / (float)RAND_MAX;
    }

    float* deviceX, * deviceY, * deviceZ;
    cudaMalloc((void**)&deviceX, numRowsX * numColsX * sizeof(float));
    cudaMalloc((void**)&deviceY, numColsX * numColsY * sizeof(float));
    cudaMalloc((void**)&deviceZ, numRowsX * numColsY * sizeof(float));

    cudaMemcpy(deviceX, hostX, numRowsX * numColsX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceY, hostY, numColsX * numColsY * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 16;
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks(ceil(numColsY / (float)blockSize), ceil(numRowsX / (float)blockSize));

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    MultiplyMatrix<<<numBlocks, threadsPerBlock>>>(deviceX, deviceY, deviceZ, numRowsX, numColsX, numColsY);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(hostZ, deviceZ, numRowsX * numColsY * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Elapsed time: %f ms\n", elapsedTime);

    free(hostX);
    free(hostY);
    free(hostZ);
    cudaFree(deviceX);
    cudaFree(deviceY);
    cudaFree(deviceZ);

    return 0;
}
void printMatrix(float* matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            printf("%f ", matrix[i * numCols + j]);
        }
        printf("\n");
    }
}
__global__ void MultiplyMatrix(float* X, float* Y, float* Z, int h, int w, int d)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < h && col < d) {
        float output = 0;
        for (int k = 0; k < w; k++) {
            output +=X[row * w + k] * Y[k * d + col];
        }
        Z[row* d + col] = output;
    }
}