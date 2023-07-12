#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include "../gator_db/include/strutil.h"

void main() {
    std::ifstream file("memeTemplateEvent64k-int.csv");
    std::string header;
    std::vector<std::vector<int>> data;

    if (file.is_open()) {
        std::string line;
        std::getline(file, header);
        std::getline(file, line);
        while (std::getline(file, line))
        {
            auto& row_str = strutil::split(line, ",");
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

    // group by


    // order by
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