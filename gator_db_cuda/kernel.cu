
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
            auto row_str = strutil::split(line,",");
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