/*
 * By: Ali Ghanbari
 * Student ID: 40110524
 * Professor: Dr. Mossavi nia
 * Course: Prallel Proccessing
**/
#include <iostream>
#include "query.h"
#include "storage.h"
#include "operations.h"
#include <iomanip>
#include <fstream>
#include <chrono>

/*
    Benchmark Queries:
    - select * from grades4096k.csv where grade > 9 limit 10
    - select * from grades64k.csv order by semester asc limit 10
    - select * from grades64k.csv group by course_id limit 10
    - select * from grades2048k.csv where grade > 9 group by course_id order by avg(grade) asc limit 10
*/

int main()
{

    std::string q;
    std::vector<std::string> lines;

    std::cout << "Enter Query: ";
    std::getline(std::cin, q);
    std::cout << std::endl;

    auto query = parse_query(q);
    table t(query.from);
    parse_csv(query.from, lines);

    auto begin = std::chrono::high_resolution_clock::now();

    import_table_from_csv_lines(lines, &t);
    auto r = query.where.size() > 0 ? restrict(t, query.where[0]) : t;
    auto g = query.group_by.size() > 0 ? aggregation(r, query.group_by) : r;
    auto o = query.order_by.size() > 0 ? order(g, query.order_by, query.asc) : g;
    auto p = query.select.size() > 0 ? projection(o, query.select) : o;
    auto l = limit(p, query.limit);

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Query Finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    l.put(std::cout);

}