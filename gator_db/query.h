#pragma once

#include <string>
#include <vector>
#include <stdbool.h>
#include "storage.h"

class where {
public:
	std::string column;
	std::string op;
	std::string value;
};

class query
{
public:
	std::vector<std::string> select;
	std::string from;
	where where;
	std::vector<std::string> group_by;
	std::vector<std::string> order_by;
	bool asc = true;
	int limit = 1000;
};

query parse_query(std::string& query);