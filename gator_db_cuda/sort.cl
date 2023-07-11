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
