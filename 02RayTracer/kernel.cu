#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void render(float *fb, int max_x, int max_y) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x * 4 + i * 4;
	
	fb[pixel_index + 0] = float(i) / max_x;
	fb[pixel_index + 1] = float(j) / max_y;
	fb[pixel_index + 2] = 0.2;
	fb[pixel_index + 3] = 1.0;
}

extern "C" void launch_cudaProcess(dim3 blocks, dim3 threads, float *fb, int max_x, int max_y) {
	render << <blocks, threads >> >(fb, max_x, max_y);
}
