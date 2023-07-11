#pragma once
#include "query.h"

table projection(table t, std::vector<std::string> select);

table restrict(table t, where restriction);

table aggregation(table t, std::vector<std::string> group_by);

table order(table t, std::vector<std::string> order_by, bool asc);

table limit(table table, int limit);