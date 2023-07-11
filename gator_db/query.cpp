#include "query.h"
#include "include/strutil.h"
#include <regex>

query parse_query(std::string& q) {
	query result;
	std::smatch matches;

	// normalize
	for (auto& ch : q) ch = tolower(ch);

	// select
	if (regex_search(q, matches, std::regex("select\\s+(([\\w\\d()]+)(\\s*,{1}\\s*[\\w\\d()]+)*)"))) {
		std::string columnsRaw = matches[1];
		result.select = strutil::split(columnsRaw, ',');
		std::for_each(result.select.begin(), result.select.end(), strutil::trim);
	}

	// from
	if (regex_search(q, matches, std::regex("from\\s([\\w\\d.]+)"))) {
		result.from = matches[1];
	}

	// where
	if (regex_search(q, matches, std::regex("where\\s+(([\\w\\d]+)\\s*(=|>|<|<>|~=|!=)\\s*([\\w\\d.]+))"))) {
		where wh;
		wh.column = matches[2];
		wh.op = matches[3];
		wh.value = matches[4];
		result.where.push_back(wh);
	}

	// group by
	if (regex_search(q, matches, std::regex("group by\\s+(([\\w\\d()]+)(\\s*,{1}\\s*[\\w\\d()]+)*)"))) {
		std::string groupBys = matches[1];
		result.group_by = strutil::split(groupBys, ',');
		std::for_each(result.group_by.begin(), result.group_by.end(), strutil::trim);
	}

	// order by
	if (regex_search(q, matches, std::regex("order by\\s+(([\\w\\d()]+)(\\s*,{1}\\s*[\\w\\d()]+)*)\\s+(desc|asc)?"))) {
		std::string orderbys = matches[1];
		std::string dir = matches[4];
		result.asc = dir == "asc";
		result.order_by = strutil::split(orderbys, ',');
		std::for_each(result.order_by.begin(), result.order_by.end(), strutil::trim);
	}

	// limit
	if (regex_search(q, matches, std::regex("limit\\s([\\d]+)"))) {
		result.limit = stoi(matches[1]);
	}

	return result;
}