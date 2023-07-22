#include "cuda_runtime.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

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

// Odd Even Sort
__global__ void sort_kernel(int* table, int width, int length, int* sort_keys, int sort_keys_len, int asc) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int phase;
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

// copy if
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

// { "sum", "count", "max", "min", "avg" }
__global__ void group_kernel(int* data, int raw_width, int width, int length, int group_by, int worksize, int* block_output_len, int* proccessed) {
	int block_start_index = blockIdx.x * blockDim.x;
	int block_end_offset = 0;
	for (size_t i = block_start_index; i < (block_start_index + worksize) && i < length; i++)
	{
		if (proccessed[i]) {
			continue;
		}
		auto current_i = i;
		// init aggregation info
		size_t a = 0;
		for (size_t k = 0; k < raw_width; k++, a++)
		{
			if (k == group_by) {
				continue;
			}
			// sum
			data[(current_i * width) + raw_width + a + 0] = data[(current_i * width) + k];
			// count
			data[(current_i * width) + raw_width + a + 1] = 1;
			// max
			data[(current_i * width) + raw_width + a + 2] = data[(current_i * width) + k];
			// min
			data[(current_i * width) + raw_width + a + 3] = data[(current_i * width) + k];
			// avg
			data[(current_i * width) + raw_width + a + 4] = 0;
		}

		for (size_t j = i + 1; j < (block_start_index + worksize) && j < length; j++)
		{
			if (proccessed[j]) {
				continue;
			}
			auto next_i = j;
			// check if match for group by
			if (data[current_i * width + group_by] == data[next_i * width + group_by]) {
				// add the match to current record

				for (size_t k = 0; k < raw_width; k++, a++)
				{
					if (k == group_by) {
						continue;
					}
					// sum
					data[(current_i * width) + raw_width + a + 0] = data[(current_i * width) + k];
					// count
					data[(current_i * width) + raw_width + a + 1] = data[(current_i * width) + k];
					// max
					data[(current_i * width) + raw_width + a + 2] = data[(current_i * width) + raw_width + a + 2] > data[(current_i * width) + k] ? data[(current_i * width) + raw_width + a + 2] : data[(current_i * width) + k];
					// min
					data[(current_i * width) + raw_width + a + 3] = data[(current_i * width) + raw_width + a + 2] < data[(current_i * width) + k] ? data[(current_i * width) + raw_width + a + 2] : data[(current_i * width) + k];
					// avg
					data[(current_i * width) + raw_width + a + 4] = 0;
				}
				proccessed[j] = true;
			}
		}

		// align to begining
		if (i != block_end_offset) {
			// copy i -> block_end_offset
			for (size_t j = 0; j < length; j++)
			{
				data[block_end_offset * width + j] = data[i * width + j];
			}
		}
		block_end_offset++;
	}

	block_output_len[blockIdx.x] = block_end_offset;
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

	filter_kernel << <numBlocks, numThreads >> > (device_table, width, length, filter_col, op, value, device_matches);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaDeviceSynchronize();

	int* matches = new int[length];

	cudaStatus = cudaMemcpy(matches, device_table, sizeof(int) * length * width, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaFree(&device_table);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaFree(&device_matches);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	// fix matches order
	size_t m = 0;
	for (size_t i = 0; i < length; i++)
	{
		if (matches[i]) {
			for (size_t j = 0; j < width; j++)
			{
				data[m * width + j] = data[i * width + j];
				m++;
			}
		}
	}

	length = m;
	return cudaStatus;
}

cudaError_t group(int* data, int& width, int& length, int group_by)
{
	// expand table width
	int expansion = (width - 1) * 5;
	int* expanded = new int[length * (width + expansion)];

	// copy data
	for (size_t i = 0; i < length; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			expanded[i * (width + expansion) + j] = data[i * width + j];
		}
	}

	// allocate device data
	int* device_table;
	int* device_processed;
	int* device_block_output_len;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMalloc((void**)&device_table, sizeof(int) * length * (width + expansion));
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMalloc((void**)&device_processed, sizeof(int) * length);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMalloc((void**)&device_block_output_len, sizeof(int) * length);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMemcpy(device_table, expanded, sizeof(int) * length * (width + expansion), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMemset(device_processed, 0, sizeof(int) * length);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	// binary tree group by
	// start with small group bys than go to larger group bys
	int worksize = 32;
	int blocks;
	do {
		worksize *= 2;
		blocks = (length + worksize - 1) / worksize;
		group_kernel << <blocks, 1 >> > (device_table, width, width + expansion, length, group_by, worksize, device_block_output_len, device_processed);
		cudaDeviceSynchronize();
	} while (blocks != 1);

	// copy to host
	int* block_output_len = new int[0];

	cudaStatus = cudaMemcpy(expanded, device_table, sizeof(int) * length * (width + expansion), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMemcpy(block_output_len, device_block_output_len, sizeof(int) * 1, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	data = &expansion;
	width = width + expansion;
	length = block_output_len[0];

	return cudaStatus;
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

	sort_kernel << <1, length >> > (device_table, width, length, device_sort_keys, order_cols_count, asc);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaMemcpy(data, device_table, sizeof(int) * length * width, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaFree(&device_table);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	cudaStatus = cudaFree(&device_sort_keys);
	if (cudaStatus != cudaSuccess) return cudaStatus;

	return cudaStatus;
}

void limit(int* table, int& length, int new_length) {
	length = new_length > length ? length : new_length;
}

static inline std::vector<std::string> split(const std::string& str, const std::string& delim)
{
	size_t pos_start = 0, pos_end, delim_len = delim.length();
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
	std::ifstream file("memeTemplateEvent64k-int.csv");
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

	int length = data.size();
	int width = data.front().size();

	int* table = (int*)malloc(sizeof(int) * width * length);

	// copy data
	for (size_t i = 0; i < length; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			table[i * width + j] = data[i][j];
		}
	}

	// where
	if (bool where = false) {
		int where_col = 1 /*template_id*/;
		int op = 1 /*>*/;
		int v = 1000;
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
			return;
		}
	}

	// group by
	if (bool group_by = false) {
		int group_by_col = 1 /*template_id*/;
		auto result = group(
			table,
			width,
			length,
			group_by_col
		);
		if (result != cudaSuccess) {
			std::cerr << "group error: code " << result << std::endl;
			return;
		}
	}


	// order by
	if (bool order_by = true) {
		int* order_by_properties_index = new int{ 2 /* count */ };
		int order_by_properties_count = 1;
		int asc = 0 /*desc*/;
		auto result = sort(
			table,
			width,
			length,
			order_by_properties_index,
			order_by_properties_count,
			asc
		);
		if (result != cudaSuccess) {
			std::cerr << "sort error: code " << result << std::endl;
			return;
		}
	}


	// limit
	if (int lim = 10) {
		limit(table, length, lim);
	}

	// print table
	std::cout << header << std::endl;
	for (size_t i = 0; i < length; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			std::cout << table[i * width + j];
			if ((j + 1) != width) {
				std::cout << ",";
			}
		}
		std::cout << std::endl;
	}
	return 0;
}