#pragma once
#include <string>
#include <vector>
#include <map>
#include <variant>
#include <any>

enum fieldDataType
{
	DATA_TYPE_INT, DATA_TYPE_STRING, DATA_TYPE_NONE
};


class field {
private:
	fieldDataType _type;
	std::variant<int, std::string> _data;
public:
	field();
	field(int data);
	field(std::string data);
	inline fieldDataType get_type() { return _type; }
	inline void put(std::ostream& out);
	int i();
	inline void make_null() { _type = DATA_TYPE_NONE; }
	int compare_string(const std::string& other);
	field operator+(field other);
	inline bool operator==(const std::string& other) { return compare_string(other) == 0; };
	inline bool operator!=(const std::string& other) { return compare_string(other) != 0; };
	inline bool operator>(const std::string& other) { return compare_string(other) > 0; };
	inline bool operator>=(const std::string& other) { return compare_string(other) >= 0; };
	inline bool operator<(const std::string& other) { return compare_string(other) < 0; };
	inline bool operator<=(const std::string& other) { return compare_string(other) <= 0; };
	bool operator!=(field other);
};

class record
{
public:
	field* fields;
};

class column
{
public:
	std::string name;
	fieldDataType type;
};

class table
{
public:
	std::string name;
	int width, length;
	column* columns;
	record* records;

	table(std::string tableName);
	void put(std::ostream& out);
};

void parse_csv(std::string fileName, std::vector<std::string>& lines);
void import_table_from_csv_lines(std::vector<std::string>& lines, table* t);
