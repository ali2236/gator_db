#include <iostream>
#include "query.h"
#include "storage.h"
#include <iomanip>
#include <fstream>
#include <chrono>

/*
    Example Queries:
    - select host, path, count(id) from records.csv where host = adobe.com group by path order by count(id) desc limit 10000 (300ms)
    - select template_id, sum(count) from memeTemplateEvent30k.csv where type = open group by template_id order by sum(count) desc limit 10 (15580ms)
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
    auto r = restrict(t, query.where);
    auto g = aggregation(r, query.group_by);
    auto o = order(g, query.order_by, query.asc);
    auto p = projection(&o, query.select);
    auto l = limit(p, query.limit);

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Query Finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    l.put(std::cout);

}