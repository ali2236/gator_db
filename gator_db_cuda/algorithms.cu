#include "algorithms.cuh"
#include "cuda_runtime.h"
#include "device_functions.h"
#include <stdio.h>

__device__ bool compare_rows(int* a, int* b, int width, int* compare_fields, int compare_count, int asc) {
	int acc = 0;
	for (size_t i = 0; i < compare_count; i++)
	{
		int c = compare_fields[i];
		acc += a[c] - b[c];
	}
	if (asc) {
		return acc < 0;
	}
	else {
		return acc > 0;
	}
};

__global__ void oddEvenSort(int* table, int width, int length, int* sort_keys, int sort_keys_len, int asc) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int phase, i;
	int temp;

	for (phase = 0; phase < length; phase++) {
		if (phase % 2 == 0) {
			if (tid % 2 == 0) {
				if (tid < length - 1 && compare_rows(table + (tid * width), table + ((tid + 1) * width), width, sort_keys, sort_keys_len, asc)) {
					// swap
					for (size_t j = 0; j < width; j++)
					{
						temp = table[tid * width + j];
						table[tid * width + j] = table[(tid + 1) * width + j];
						table[(tid + 1) * width + j] = temp;
					}
				}
			}
		}
		else {
			if (tid % 2 != 0) {
				if (tid < length - 1 && compare_rows(table + (tid * width), table + ((tid + 1) * width), width, sort_keys, sort_keys_len, asc)) {
					// swap
					for (size_t j = 0; j < width; j++)
					{
						temp = table[tid * width + j];
						table[tid * width + j] = table[(tid + 1) * width + j];
						table[(tid + 1) * width + j] = temp;
					}
				}
			}
		}
		__syncthreads();
	}
}

cudaError_t sort(int* data, int width, int length, int* order_cols, int order_cols_count, int asc) {
	int* device_table;
	int* device_sort_keys;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMalloc((void**)&device_table, sizeof(int) * length * width);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMalloc((void**)&device_sort_keys, sizeof(int) * order_cols_count);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMemcpy(device_table, data, sizeof(int) * length * width, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMemcpy(device_sort_keys, order_cols, sizeof(int) * order_cols_count, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	oddEvenSort<<<1,1>>>(device_table, width, length, device_sort_keys, order_cols_count, asc);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMemcpy(data, device_table, sizeof(int) * length * width, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaFree(&device_table);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaFree(&device_sort_keys);
	if (cudaStatus != cudaSuccess) return cudaStatus;
}

void limit(int* table, int& length, int new_length) {
	length = new_length > length ? length : new_length;
}