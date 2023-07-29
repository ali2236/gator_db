
#include "operations.h"
#include <algorithm>
#include <iterator>

table projection(table t, std::vector<std::string> select) {
	std::string* colNames = &select[0];
	std::vector<int> field_selector;
	column* columns = new column[select.size()];
	for (size_t i = 0; i < select.size(); i++)
	{
		// find first match column by name
		for (size_t j = i; j < t.width; j++)
		{
			if (colNames[i] == t.columns[j].name) {
				columns[i] = t.columns[j];
				field_selector.push_back(j);
				break;
			}
		}
	}
	t.columns = columns;
	t.width = select.size();

	record* records = new record[t.length];
	for (size_t i = 0; i < t.length - 2; i++)
	{
		int j = 0;
		records[i].fields = new field[t.width];
		for (int selector : field_selector) {
			records[i].fields[j++] = t.records[i].fields[selector];
		}
	}
	t.records = records;

	return t;
}

table restrict(table t, where restriction){

	int control_field = -1;
	for (size_t i = 0; i < t.width; i++)
	{
		if (t.columns[i].name == restriction.column) {
			control_field = i;
			break;
		}
	}

	std::vector<record> filtered;

	auto eq = [&restriction](field& f) {return (f == restriction.value); };
	auto neq = [&restriction](field& f) {return !(f == restriction.value); };
	auto lt = [&restriction](field& f) {return (f < restriction.value); };
	auto lte = [&restriction](field& f) {return (f <= restriction.value); };
	auto gt = [&restriction](field& f) {return (f > restriction.value); };
	auto gte = [&restriction](field& f) {return (f >= restriction.value); };

	if (restriction.op == "=") {
		std::copy_if(t.records, t.records + t.length - 2, std::back_inserter(filtered), [control_field,&eq](record& record) {return  eq(record.fields[control_field]); });
	}
	else if (restriction.op == "!=" || restriction.op == "<>" || restriction.op == "~=") {
		std::copy_if(t.records, t.records + t.length, std::back_inserter(filtered), [control_field, &neq](record& record) {return  neq(record.fields[control_field]); });
	}
	else if (restriction.op == "<") {
		std::copy_if(t.records, t.records + t.length - 2, std::back_inserter(filtered), [control_field, &lt](record& record) {return  lt(record.fields[control_field]); });
	}
	else if (restriction.op == "<=") {
		std::copy_if(t.records, t.records + t.length - 2, std::back_inserter(filtered), [control_field, &lte](record& record) {return  lte(record.fields[control_field]); });
	}
	else if (restriction.op == ">") {
		std::copy_if(t.records, t.records + t.length - 2, std::back_inserter(filtered), [control_field, &gt](record& record) {return  gt(record.fields[control_field]); });
	}
	else if (restriction.op == ">=") {
		std::copy_if(t.records, t.records + t.length - 2, std::back_inserter(filtered), [control_field, &gte](record& record) {return  gte(record.fields[control_field]); });
	}

	t.length = filtered.size();
	t.records = new record[t.length];
	for (size_t i = 0; i < t.length; i++)
	{
		t.records[i] = filtered[i];
	}

	return t;
}

table aggregation(table t, std::vector<std::string> group_by) {
	std::string operations[] = { "sum", "count", "max", "min", "avg" };
	int old_width = t.width;

	std::vector<std::string> int_cols;
	std::vector<int> int_cols_indexes;
	std::vector<int> group_indexes;
	for (int j = 0; j < old_width; j++) {
		if (std::find(group_by.begin(), group_by.end(), t.columns[j].name) != group_by.end()) {
			group_indexes.push_back(j);
		}
		if (t.columns[j].type == DATA_TYPE_INT) {
			int_cols.push_back(t.columns[j].name);
			int_cols_indexes.push_back(j);
		}
	}

	t.width += int_cols.size() * 5;
	column* cols = new column[t.width];
	for (int j = 0; j < old_width; j++) {
		cols[j] = t.columns[j];
	}
	t.columns = cols;

	int i = old_width;
	for (auto& ic : int_cols)
	{
		for (int j = 0; j < 5; j++) {
			std::string& op = operations[j];
			column c;
			c.name = op + "(" + ic + ")";
			c.type = DATA_TYPE_INT;
			t.columns[i++] = c;
		}
	}

	// group by similar field
	int len = 0;
	record* records = new record[t.length];

	for (size_t i = 0; i < t.length - 2; i++) {
		auto current = t.records[i];
		if (current.fields[group_indexes[0]].get_type() == DATA_TYPE_NONE) {
			continue;
		}

		record merged;
		merged.fields = new field[t.width];

		for (size_t j = 0; j < old_width; j++) {
			merged.fields[j] = current.fields[j];
		}

		int* intColIndexArr = &int_cols_indexes[0];
		for (int j = 0; j < int_cols.size(); j++) {
			merged.fields[old_width + (j * 5) + 0] = current.fields[intColIndexArr[j]]; // sum
			merged.fields[old_width + (j * 5) + 1] = field(1); // count
			merged.fields[old_width + (j * 5) + 2] = current.fields[intColIndexArr[j]]; // max
			merged.fields[old_width + (j * 5) + 3] = current.fields[intColIndexArr[j]]; // min
			merged.fields[old_width + (j * 5) + 4] = 0; // avg
		}

		for (size_t j = i + 1; j < t.length - 2; j++)
		{
			auto next = t.records[j];
			int match = 1;
			for (int gi : group_indexes) {
				if (current.fields[gi] != next.fields[gi]) {
					match = 0;
					break;
				}
			}

			if (match) {
				for (size_t k = 0; k < int_cols.size(); k++)
				{
					field& f = next.fields[intColIndexArr[k]];
					merged.fields[old_width + (k * 5) + 0] = merged.fields[old_width + (k * 5) + 0] + f; // sum
					merged.fields[old_width + (k * 5) + 1] = merged.fields[old_width + (k * 5) + 1] + field(1); // count
					merged.fields[old_width + (k * 5) + 2] = f.i() > merged.fields[old_width + (k * 5) + 2].i() ? f : merged.fields[old_width + (k * 5) + 2]; // max
					merged.fields[old_width + (k * 5) + 3] = f.i() < merged.fields[old_width + (k * 5) + 2].i() ? f : merged.fields[old_width + (k * 5) + 2]; // min
					merged.fields[old_width + (k * 5) + 4] = field(merged.fields[old_width + (k * 5) + 0].i() / merged.fields[old_width + (k * 5) + 1].i()); // avg
				}
				next.fields[group_indexes[0]].make_null();
			}
		}

		records[len++] = merged;
		current.fields[group_indexes[0]].make_null();
	}

	t.length = len;
	t.records = records;

	return t;
}

table order(table t, std::vector<std::string> order_by, bool asc)
{
	std::vector<int> compare_fields;
	for (auto& order : order_by) {
		for (int i = 0; i < t.width; i++) {
			if (t.columns[i].name == order && t.columns[i].type == DATA_TYPE_INT) {
				compare_fields.push_back(i);
			}
		}
	}

	auto cmpr = [&compare_fields, asc](record a, record b) -> bool {
		int acc = 0;
		for (auto c : compare_fields) {
			acc += a.fields[c].i() - b.fields[c].i();
		}
		if (asc) {
			return acc < 0;
		}
		else {
			return acc > 0;
		}
	};

	int i, j, k;

	for (i = 0; i < t.length - 2; i++) {
		if (i % 2 == 0) {
			// Even phase
			for (j = 2; j < t.length - 2; j += 2) {
				if (cmpr(t.records[j], t.records[j-1])) {
					record temp = t.records[j];
					t.records[j] = t.records[j - 1];
					t.records[j - 1] = temp;
				}
			}
		}
			else {
				// Odd phase
				for (j = 1; j < t.length - 2; j += 2) {
					if (cmpr(t.records[j], t.records[j - 1])) {
						record temp = t.records[j];
						t.records[j] = t.records[j - 1];
						t.records[j - 1] = temp;
					}
				}
			}
	}

	// std::sort(t.records, t.records + t.length - 2, cmpr);

	return t;
}

table limit(table t, int limit) {
	t.length = limit > t.length ? t.length : limit;
	memcpy(t.records, t.records, sizeof(record) * t.length);
	return t;
}