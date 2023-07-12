#include <sycl/sycl.hpp>
#include "parallel.h"
#include "include/strutil.h"

void parse_parallel(std::vector<std::string>& lines, table* t)
{
	////////////////////////////////
	/// serial
	////////////////////////////////

	size_t i = 0;
	std::string line;
	// get column names
	line = lines[i++];
	auto colNames = strutil::split(line, ",");

	// get column types
	line = lines[i++];
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

	////////////////////////////////
	/// parallel
	////////////////////////////////

	auto exception_handler = [](sycl::exception_list e_list) {
		for (std::exception_ptr const& e : e_list) {
			try {
				std::rethrow_exception(e);
			}
			catch (std::exception const& e) {
				std::cout << "Failure" << std::endl;
				std::terminate();
			}
		}
	};

	sycl::queue q;
	std::cout << "Running on device: "
		<< q.get_device().get_info<sycl::info::device::name>() << "\n";

	////////////////////////////////
	/// Host Data
	////////////////////////////////

	int lines_count = lines.size() - 2;
	sycl::range<1> records_size{ lines_count };
	std::vector<record> records;
	std::vector<const char*> data_lines;

	// init lines
	while (i < lines.size())
	{
		auto c = lines[i++].c_str();
		data_lines.push_back(c);
	}

	// init records
	records.resize(lines_count);

	////////////////////////////////
	/// Device Data
	////////////////////////////////

	int kernels = lines.size();
	sycl::buffer lines_buf(data_lines);
	sycl::buffer records_buf(records.data(), records_size);

	////////////////////////////////
	/// kernel
	////////////////////////////////
	q.submit([&](sycl::handler& h) {
		sycl::accessor lines_accessor(lines_buf, h, sycl::read_only);
		sycl::accessor records_accessor(records_buf, h, sycl::write_only, sycl::no_init);

		h.parallel_for(kernels, [=](auto i) {
			record r;
			r.fields = new field[t->width];
			
		});

	});
	q.wait();
}
/*
size_t records_count = 0;
int offset = i;
t->length = lines.size() - i;
t->records = new record[t->length + 2];
for (; i < t->length; i++) {
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

	t->records[i - offset] = r;
	records_count++;
	if (records_count % 50000 == 0) {
		std::cout << "Loaded " << records_count << " Records" << std::endl;
	}
}
std::cout << "All Records loaded" << std::endl;
std::cout << "Size: " << sizeof(t->records) * t->length << " Bytes" << std::endl;*/
