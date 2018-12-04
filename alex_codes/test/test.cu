#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "iostream"

__global__ void test_min(int * min_num) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < 5) {
        return;
    }
    printf("%d %d\n", *min_num, thread_id);
    atomicMin(min_num, thread_id);
}

int main(int argc, char * argv[]) {
    int * min_num;
    int num = 220;
    cudaMalloc((void **) &min_num, sizeof(int));
    cudaMemcpy(min_num, &num, sizeof(int), cudaMemcpyHostToDevice);
    test_min <<<1000, 1>>> (min_num);
    cudaMemcpy(&num, min_num, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << num << std::endl;;
}
