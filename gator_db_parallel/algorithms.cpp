#include "algorithms.h"
#include <sycl/sycl.hpp>

using namespace sycl;

bool compare_rows(row a, row b, vector<int> compare_fields, int asc) {
	int acc = 0;
	for (size_t i = 0; i < compare_fields.size(); i++)
	{
		int c = compare_fields[i];
		acc += a[c] - b[c];
	}
	if (asc) {
		return acc < 0;
	}
	else {
		return acc > 0;
	}
};

table sort(table t, vector<int> order_cols, int asc) {
	queue q;
	int work_size = t.size();


	q.submit([&](sycl::handler& h) {
		h.parallel_for(work_size, [=](auto& tid) {
			int phase, i;
	vector<int> temp;
	for (phase = 0; phase < t.size(); phase++) {
		if (phase % 2 == 0) {
			if (tid % 2 == 0) {
				if (tid < t.size() - 1 && compare_rows(t[tid], t[tid + 1], order_cols, asc)) {
					temp = t[tid];
					t[tid] = t[tid + 1];
					t[tid + 1] = temp;
				}
			}
		}
		else {
			if (tid % 2 != 0) {
				if (tid < t.size() - 1 && compare_rows(t[tid], t[tid + 1], order_cols, asc)) {
					temp = t[tid];
					t[tid] = t[tid + 1];
					t[tid + 1] = temp;
				}
			}
		}
		q.
	});
		});
	});
}

table limit(table t, int new_length) {
	int length = new_length > t.size() ? t.size() : new_length;
	t.resize(length);
	return t;
}