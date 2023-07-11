#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;


static auto exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const& e : e_list) {
        try {
            std::rethrow_exception(e);
        }
        catch (std::exception const& e) {
#if _DEBUG
            std::cout << "Failure" << std::endl;
#endif
            std::terminate();
        }
    }
};


void VectorAdd(queue& q, const IntVector& a_vector, const IntVector& b_vector,
    IntVector& sum_parallel) {
    // Create the range object for the vectors managed by the buffer.
    range<1> num_items{ a_vector.size() };

    // Create buffers that hold the data shared between the host and the devices.
    // The buffer destructor is responsible to copy the data back to host when it
    // goes out of scope.
    buffer a_buf(a_vector);
    buffer b_buf(b_vector);
    buffer sum_buf(sum_parallel.data(), num_items);

    for (size_t i = 0; i < num_repetitions; i++) {

        // Submit a command group to the queue by a lambda function that contains the
        // data access permission and device computation (kernel).
        q.submit([&](handler& h) {
            // Create an accessor for each buffer with access permission: read, write or
            // read/write. The accessor is a mean to access the memory in the buffer.
            accessor a(a_buf, h, read_only);
            accessor b(b_buf, h, read_only);

            // The sum_accessor is used to store (with write permission) the sum data.
            accessor sum(sum_buf, h, write_only, no_init);

            // Use parallel_for to run vector addition in parallel on device. This
            // executes the kernel.
            //    1st parameter is the number of work items.
            //    2nd parameter is the kernel, a lambda that specifies what to do per
            //    work item. The parameter of the lambda is the work item id.
            // DPC++ supports unnamed lambda kernel by default.
            h.parallel_for(num_items, [=](auto i) { sum[i] = a[i] + b[i]; });
            });
    };
    // Wait until compute tasks on GPU done
    q.wait();
}


void main() {
    const auto& selector = default_selector_v;

    try {
        queue q(selector, exception_handler);

        std::cout << "Running on device: "
            << q.get_device().get_info<info::device::name>() << "\n";
      

        // Vector addition in DPC++
        VectorAdd(q, a, b, sum_parallel);
    }
    catch (exception const& e) {
        std::cout << "An exception is caught for vector add.\n";
        std::terminate();
    }

    // Compute the sum of two vectors in sequential for validation.
    for (size_t i = 0; i < sum_sequential.size(); i++)
        sum_sequential.at(i) = a.at(i) + b.at(i);

    // Verify that the two vectors are equal.  
    for (size_t i = 0; i < sum_sequential.size(); i++) {
        if (sum_parallel.at(i) != sum_sequential.at(i)) {
            std::cout << "Vector add failed on device.\n";
            return -1;
        }
    }

    int indices[]{ 0, 1, 2, (static_cast<int>(a.size()) - 1) };
    constexpr size_t indices_size = sizeof(indices) / sizeof(int);

    // Print out the result of vector add.
    for (int i = 0; i < indices_size; i++) {
        int j = indices[i];
        if (i == indices_size - 1) std::cout << "...\n";
        std::cout << "[" << j << "]: " << a[j] << " + " << b[j] << " = "
            << sum_parallel[j] << "\n";
    }

    a.clear();
    b.clear();
    sum_sequential.clear();
    sum_parallel.clear();

    std::cout << "Vector add successfully completed on device.\n";
    return 0;
}