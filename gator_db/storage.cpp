#include "storage.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include "include/strutil.h"

field::field()
{
	_type = DATA_TYPE_NONE;
}

field::field(int data)
{
	_data = data;
	_type = DATA_TYPE_INT;
}

field::field(std::string data)
{
	_data = data;
	_type = DATA_TYPE_STRING;
}

inline void field::put(std::ostream& out)
{
	switch (_type)
	{
	case DATA_TYPE_INT:
		out << std::get<int>(_data);
		break;
	case DATA_TYPE_STRING:
		out << std::get<std::string>(_data);
		break;
	default:
		break;
	}
}

int field::i()
{
	return std::get<int>(_data);
}

field field::operator+(field other)
{
	if (this->_type == other._type) {
		switch (_type)
		{
		case DATA_TYPE_INT:
			return field(std::get<int>(_data) + std::get<int>(other._data));
		case DATA_TYPE_STRING:
			return field(std::get<std::string>(_data) + std::get<std::string>(other._data));
		default:
			break;
		}
	}
}

bool field::operator!=(field other)
{
	if (this->_type != other._type) {
		return true;
	}

	switch (_type)
	{
	case DATA_TYPE_INT:
		return std::get<int>(_data) != std::get<int>(other._data);
	case DATA_TYPE_STRING:
		return std::get<std::string>(_data) != std::get<std::string>(other._data);
	default:
		break;
	}

	return false;
}

int field::compare_string(const std::string& other)
{
	switch (_type)
	{
	case DATA_TYPE_INT:
		return std::get<int>(_data) - std::stoi(other);
	case DATA_TYPE_STRING:
		return std::get<std::string>(_data).compare(other);
	default:
		break;
	}
}

table::table(std::string tableName)
{
	name = tableName;
}

std::map<std::string, int> vectorToMap(const std::vector<std::string>&words) {
	std::map<std::string, int> wordMap;

	// Creating the map from each word to its index
	for (int i = 0; i < words.size(); i++) {
		wordMap[words[i]] = i;
	}

	return wordMap;
}

void parse_csv(std::string fileName, std::vector<std::string>& lines) {
	std::ifstream file(fileName);
	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line))
		{
			lines.push_back(std::string(line));
		}
		file.close();
	}
}

void import_table_from_csv_lines(std::vector<std::string>& lines, table* t)
{	
	size_t i = 0;
	std::string line;
	std::string* linesArr = &lines[0];
	// get column names
	line = linesArr[i++];
	auto colNames = strutil::split(line, ",");

	// get column types
	line = linesArr[i++];
	auto colTypes = strutil::split(line, ",");

	t->width = colNames.size();
	t->columns = new column[t->width];

	for (size_t j = 0; j < t->width; j++)
	{
		auto& name = colNames[j];
		auto& type = colTypes[j];
		strutil::trim(name);
		strutil::trim(type);

		column c;
		c.name = name;
		c.type = type == "int" ? DATA_TYPE_INT : DATA_TYPE_STRING;

		t->columns[j] = c;
	}

	size_t records_count = 0;
	int offset = i;
	t->length = lines.size() - i;
	t->records = new record[t->length+2];
	for (;i < t->length;i++){
		line = linesArr[i];
		record r;
		auto fieldsRaw = strutil::split(line, ",");
		r.fields = new field[t->width];
		for (size_t j = 0; j < fieldsRaw.size(); j++)
		{
			std::string value = fieldsRaw[j];
			strutil::trim(value);
			switch (t->columns[j].type)
			{
			case DATA_TYPE_STRING:
				r.fields[j] = field(value);
				break;
			case DATA_TYPE_INT:
				r.fields[j] = field(std::stoi(value));
				break;
			default:
				break;
			}
		}

		t->records[i-offset] = r;
		records_count++;
		if (records_count % 10000 == 0) {
			std::cout << "Loaded " << records_count << " Records" << std::endl;
		}
	}
	std::cout << "All Records loaded" << std::endl;
	std::cout << "Size: " << sizeof(t->records) * t->length << " Bytes" << std::endl;
}

void table::put(std::ostream& out) {
	// print table header
	int n = this->width;
	for (size_t i = 0; i < n; i++) {
		out << columns[i].name;
		if ((i + 1) != n) {
			out << ",";
		}
	}
	out << "\n";
	// print records
	for (size_t k = 0; k < length; k++)
	{
		for (size_t j = 0; j < width; j++) {
			auto& field = records[k].fields[j];
			field.put(out);
			if ((j + 1) != width) {
				out << ",";
			}
		}
		out << "\n";
	}
}