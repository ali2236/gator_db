/*
 * By: Ali Ghanbari
 * Student ID: 40110524
 * Professor: Dr. Mossavi nia
 * Course: Prallel Proccessing
 *
**/
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

///////////////////////////////////////////////////////////
//// GPU Kernels
///////////////////////////////////////////////////////////

inline __device__ int Min(int a, int b)
{
	return a < b ? a : b;
}

__device__ __host__ bool compare_rows(int* a, int* b, int compare_field, int asc) {
	int acc = a[compare_field] - b[compare_field];
	if (asc) {
		return acc < 0;
	}
	else {
		return acc > 0;
	}
}

__device__ void swap(int* a, int* b, int len) {
	int t;
	for (int i = 0; i < len; i++)
	{
		t = a[i];
		a[i] = b[i];
		b[i] = t;
	}
}

__global__ void even_sort_kernel(int* table, int width, int length, int sort_key, int asc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k, t;

	if (i % 2 == 0 && i < length) {
		// Even phase
		if (compare_rows(table + (i * width), table + ((i - 1) * width), sort_key, asc)) {
			// swap
			for (k = 0; k < width; k++)
			{
				t = table[(i - 1) * width + k];
				table[(i - 1) * width + k] = table[i * width + k];
				table[i * width + k] = t;
			}
		}
	}
}

__global__ void odd_sort_kernel(int* table, int width, int length, int sort_key, int asc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k, t;

	if (i % 2 != 0 && i < length) {
		// Odd phase
		if (compare_rows(table + (i * width), table + ((i - 1) * width), sort_key, asc)) {
			// swap
			for (k = 0; k < width; k++)
			{
				t = table[(i - 1) * width + k];
				table[(i - 1) * width + k] = table[i * width + k];
				table[i * width + k] = t;
			}
		}
	}
}

__global__ void merge_sort_kernel(int* table, int width, int length, int sort_key, int asc) {
	int left = 0;
	int right = length - 1;
	int mid = (left + right) / 2;

	int l = left, r = mid + 1;
	while (l <= mid && r <= right) {
		/*
		if (table[mid * width + sort_key] <= table[r * width + sort_key]) {
			return;
		}
		*/
		// If element 1 is in right place
		if (table[l * width + sort_key] <= table[r * width + sort_key]) {
			l++;
		}
		else {
			int value_key = r;
			int index = r;

			// Shift all the elements between element 1
			// element 2, right by 1.
			while (index != l) {
				for (int k = 0; k < width; k++)
				{
					int t = table[index * width + k];
					table[index * width + k] = table[(index - 1) * width + k];
					table[(index - 1) * width + k] = t;
				}
				index--;
			}
			for (int k = 0; k < width; k++)
			{
				table[l * width + k] = table[value_key * width + k];
			}

			// Update all the pointers
			l++;
			mid++;
			r++;
		}
	}
}

// restrict kernel 
__global__ void filter_kernel(int* data, int width, int length, int filter_col, int op, int value, int* match) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < length)
	{
		int v = data[tid * width + filter_col];
		if (op == 0 /*=*/)
		{
			match[tid] = (v == value);
		}
		else if (op == 1 /*>*/) {
			match[tid] = (v > value);
		}
		else if (op == 2 /*<*/) {
			match[tid] = (v < value);
		}
	}
}

#define SUM 0
#define COUNT 1
#define MAX 2
#define MIN 3
#define AVG 4

// { "sum", "count", "max", "min", "avg" }
__global__ void group_kernel(int* data, int raw_width, int width, int length, int group_by, int* proccessed, int init) {
	int block_start_index = blockIdx.x * blockDim.x;
	int block_end_index = block_start_index + blockDim.x;
	int global_index = block_start_index + threadIdx.x;
	const int debug_clear = false;
	__shared__ int parent_count[1];

	if (global_index >= length) return;
	if (proccessed[global_index]) return;

	if (init) {
		// init expansion data { "sum", "count", "max", "min", "avg" } with default values
		int a = 0;
		for (int k = 0; k < raw_width; k++)
		{
			if (k == group_by) {
				continue;
			}

			// sum
			data[(global_index * width) + raw_width + (a * 5) + SUM] = data[(global_index * width) + k];
			// count
			data[(global_index * width) + raw_width + (a * 5) + COUNT] = 1;
			// max
			data[(global_index * width) + raw_width + (a * 5) + MAX] = data[(global_index * width) + k];
			// min
			data[(global_index * width) + raw_width + (a * 5) + MIN] = data[(global_index * width) + k];
			// avg
			data[(global_index * width) + raw_width + (a * 5) + AVG] = 0;
			// next expansion position
			a++;
		}
	}

	// wait for all threads to copy data
	__syncthreads();

	// find parent thread id
	int parent_id = global_index;
	for (int i = block_start_index; i < global_index; i++)
	{
		if (data[i * width + group_by] == data[global_index * width + group_by]) {
			parent_id = i;
			break;
		}
	}

	// count parents
	if (parent_id == global_index) {
		atomicAdd(&parent_count[0], 1);
	}

	// add tid data to parent data
	if (parent_id != global_index) {
		// add expansion data { "sum", "count", "max", "min", "avg" } to parent row
		int a = 0;
		for (int k = 0; k < raw_width; k++)
		{
			if (k == group_by) {
				continue;
			}
			int self_expansion_start = global_index * width + raw_width;
			int parent_expansion_start = parent_id * width + raw_width;
			// sum
			int sum = data[self_expansion_start + (a * 5) + SUM];
			atomicAdd(&data[parent_expansion_start + (a * 5) + SUM], sum);
			// count
			int count = data[self_expansion_start + (a * 5) + COUNT];
			atomicAdd(&data[parent_expansion_start + (a * 5) + COUNT], count);
			// max
			int max = data[self_expansion_start + (a * 5) + MAX];
			atomicMax(&data[parent_expansion_start + (a * 5) + MAX], max);
			// min
			int min = data[self_expansion_start + (a * 5) + MIN];
			atomicMin(&data[parent_expansion_start + (a * 5) + MIN], min);

			// next expansion position
			a++;
		}

	}

	// wait for all threads to add their data to parent row
	__syncthreads();

	// calculate avg on parent rows
	if (global_index == parent_id) {
		int a = 0;
		for (int k = 0; k < raw_width; k++)
		{
			if (k == group_by) {
				continue;
			}

			int self_expansion_start = global_index * width + raw_width;
			// avg
			int avg = data[self_expansion_start + (a * 5) + SUM] / data[self_expansion_start + (a * 5) + COUNT];
			data[self_expansion_start + (a * 5) + AVG] = avg;

			// next expansion position
			a++;
		}

	}

	// zero out non parent rows
	if (global_index != parent_id) {
		for (int i = 0; i < width; i++)
		{
			data[global_index * width + i] = 0;
		}

		// mark used
		proccessed[global_index] = 1;
	}

	return;
}

cudaError_t filter(int* data, int width, int& length, int filter_col, int op, int value)
{
	int* device_table;
	int* device_matches;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMalloc((void**)&device_table, sizeof(int) * length * width);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMalloc((void**)&device_matches, sizeof(int) * length);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMemcpy(device_table, data, sizeof(int) * length * width, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	int numThreads = 256;
	int numBlocks = (length + numThreads - 1) / numThreads;

	filter_kernel<<<numBlocks, numThreads >>>(device_table, width, length, filter_col, op, value, device_matches);

	cudaDeviceSynchronize();

	int* matches = new int[length];

	cudaStatus = cudaMemcpy(matches, device_matches, sizeof(int) * length, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaFree(device_table);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaFree(device_matches);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	// fix matches order
	int m = 0;
	for (int i = 0; i < length; i++)
	{
		if (matches[i]) {
			for (int j = 0; j < width; j++)
			{
				data[m * width + j] = data[i * width + j];
			}
			m++;
		}
	}

	length = m;
	return cudaStatus;
}

cudaError_t group(int*& data, int& width, int& length, int group_by, std::string& header)
{
	// expand table width
	int expanded_width = width + ((width - 1) * 5);
	int* expanded = new int[length * expanded_width];

	// copy data
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < expanded_width; j++)
		{
			if (j < width) {
				expanded[i * expanded_width + j] = data[i * width + j];
			}
			else {
				expanded[i * expanded_width + j] = 0;
			}
		}
	}

	// add headers
	std::string operations[] = { "sum", "count", "max", "min", "avg" };
	for (int i = 0; i < width; i++)
	{
		if (i == group_by) {
			continue;
		}
		for (int j = 0; j < 5; j++) {
			std::string& op = operations[j];
			std::string& name = op + "(" + std::to_string(i) + ")";
			header += ", " + name;
		}
	}

	// allocate device data
	int* device_table;
	int* device_processed;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMalloc((void**)&device_table, sizeof(int) * length * expanded_width);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMalloc((void**)&device_processed, sizeof(int) * length);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMemcpy(device_table, expanded, sizeof(int) * length * expanded_width, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMemset(device_processed, 0, sizeof(int) * length);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	// binary tree group by
	// start with small group bys than go to larger group bys

	int threads = length > 256 ? 256 : length / 8;
	int blocks = length / threads;

	// inital run
	group_kernel << <blocks, threads >> > (device_table, width, expanded_width, length, group_by, device_processed, true);

	do {
		threads *= 2;
		blocks = (length + threads - 1) / threads;
		group_kernel << <blocks, threads >> > (device_table, width, expanded_width, length, group_by, device_processed, false);
	} while (blocks != 1);

	cudaDeviceSynchronize();

	// copy to host
	int* processed = new int[length];

	cudaStatus = cudaMemcpy(processed, device_processed, sizeof(int) * length, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMemcpy(expanded, device_table, sizeof(int) * length * expanded_width, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaFree(device_table);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaFree(device_processed);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	// sort
	int len = 0;
	for (int i = 0; i < length; i++)
	{
		if (processed[i]) {
			continue;
		}

		if (i != len) {
			for (int j = 0; j < expanded_width; j++)
			{
				expanded[len * expanded_width + j] = expanded[i * expanded_width + j];
			}
		}
		len++;
	}

	data = expanded;
	width = expanded_width;
	length = len;

	return cudaStatus;
}

cudaError_t sort(int* data, int width, int length, int order_col, int asc) {
	int* device_table;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMalloc((void**)&device_table, sizeof(int) * length * width);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMemcpy(device_table, data, sizeof(int) * length * width, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	const int base_threads = 512;
	int threads = length > base_threads ? base_threads : length;
	int blocks = length / threads;

	for (int i = 0; i < length; i++)
	{
		odd_sort_kernel << <blocks, threads >> > (device_table, width, length, order_col, asc);
		even_sort_kernel << <blocks, threads >> > (device_table, width, length, order_col, asc);
	}

	cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(data, device_table, sizeof(int) * length * width, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaFree(device_table);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	return cudaStatus;
}

///////////////////////////////////////////////////////////
//// CPU Utilites
///////////////////////////////////////////////////////////

void limit(int* table, int& length, int new_length) {
	length = new_length > length ? length : new_length;
}

static inline std::vector<std::string> split(const std::string& str, const std::string& delim)
{
	int pos_start = 0, pos_end, delim_len = delim.length();
	std::string token;
	std::vector<std::string> tokens;

	while ((pos_end = str.find(delim, pos_start)) != std::string::npos)
	{
		token = str.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		tokens.push_back(token);
	}

	tokens.push_back(str.substr(pos_start));
	return tokens;
}

int main()
{
	std::ifstream file("grades64k.csv");
	std::string header;
	std::vector<std::vector<int>> data;

	if (file.is_open()) {
		std::string line;
		std::getline(file, header);
		std::getline(file, line);
		while (std::getline(file, line))
		{
			auto row_str = split(line, ",");
			std::vector<int> row;
			for (auto& s : row_str) row.push_back(std::stoi(s));
			data.push_back(row);
		}
		file.close();
	}
	else {
		std::cout << "Could not open file.";
		return 255;
	}

	auto begin = std::chrono::high_resolution_clock::now();

	int length = data.size();
	int width = data.front().size();

	int* table = (int*)malloc(sizeof(int) * width * length);

	// copy data
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < width; j++)
		{
			table[i * width + j] = data[i][j];
		}
	}

	// where
	if (bool where = true) {
		int where_col = 3 /*grade*/;
		int op = 1 /*>*/;
		int v = 9;
		auto result = filter(
			table,
			width,
			length,
			where_col,
			op,
			v
		);
		if (result != cudaSuccess) {
			std::cerr << "filter error: code " << result << std::endl;
			return 255;
		}
	}

	// group by
	if (bool group_by = true) {
		int group_by_col = 2 /*course_id*/;
		auto result = group(
			table,
			width,
			length,
			group_by_col,
			header
		);
		if (result != cudaSuccess) {
			std::cerr << "group error: code " << result << std::endl;
			return 255;
		}
	}


	// order by
	if (bool order_by = true) {
		int order_by_property = 4 + (2 * 5) + AVG; /* avg(grade) */
		int asc = 1 /*asc*/;
		auto result = sort(
			table,
			width,
			length,
			order_by_property,
			asc
		);
		if (result != cudaSuccess) {
			std::cerr << "sort error: code " << result << std::endl;
			return 255;
		}
	}


	// limit
	if (int lim = 10) {
		limit(table, length, lim);
	}

	auto end = std::chrono::high_resolution_clock::now();

	// print table
	std::cout << header << std::endl;
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < width; j++)
		{
			std::cout << table[i * width + j];
			if ((j + 1) != width) {
				std::cout << ",";
			}
		}
		std::cout << std::endl;
	}

	std::cout << "Query Finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

	return 0;
}