#pragma once
#include "storage.h"

table sort(table data, vector<int> order_cols, int asc);

table limit(table t, int new_length);