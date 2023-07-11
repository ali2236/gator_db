
#include <string>
#include <vector>
#include <iostream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "include/strutil.h"
#include "algorithms.cuh"

/*
    Example Queries:
    - select * from memeTemplateEvent64k-int.csv 
*/

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
            auto& row_str = strutil::split(line,",");
            std::vector<int> row;
            for (auto& s : row_str) row.push_back(std::stoi(s));
            data.push_back(row);
        }
        file.close();
    }

    int length = data.size();
    int width = data.front().size();

    int* table = (int*) malloc(sizeof(int) * width * length);

    // copy data
    for (size_t i = 0; i < length; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            table[i * width + j] = data[i][j];
        }
    }

    // group by


    // order by
    int* order_by_properties_index = new int{2 /* count */};
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


    // limit
    limit(table, length, 10);

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

/*
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/