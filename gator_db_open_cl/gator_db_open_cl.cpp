#include <iostream>
#include <CL/cl.h>
#include <CL/opencl.hpp>

__kernel void vectorAdd(__global const float* a, __global const float* b, __global float* result, const int size) {
    int i = get_global_id(0);

    if (i < size) {
        result[i] = a[i] + b[i];
    }
}

int main()
{
    std::cout << "Hello World!\n";
}

