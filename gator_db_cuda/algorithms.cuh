#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>

cudaError_t sort(int* data, int width, int length, int* order_cols, int order_cols_count, int asc);

void limit(int* table, int& length, int new_length);