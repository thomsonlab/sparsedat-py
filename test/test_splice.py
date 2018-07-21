import sparsedat

# Manually constructing the following matrix:
# 1 0 0 1 0
# 5 7 0 0 2

TARGET_INDEX = (1, 1)
EXPECTED_VALUE = 7

sdt = sparsedat.Sparse_Data_Table()
sdt._data_type = sparsedat.Data_Type.INT
sdt._data_size = 4
sdt._num_rows = 2
sdt._num_columns = 5
sdt._metadata = {
    sparsedat.Metadata_Type.ROW_NAMES: [
        "A", "B"
    ],
    sparsedat.Metadata_Type.COLUMN_NAMES: [
        "AA", "BB", "CC", "DD", "EE"
    ]
}
sdt._row_start_indices = [0, 2]
sdt._row_lengths = [2, 3]
sdt._row_column_indices = [0, 3, 0, 1, 4]
sdt._row_data = [1, 1, 5, 7, 2]
sdt._default_value = 0
sdt._column_start_indices = [0, 2, 3, 3, 4]
sdt._column_lengths = [2, 1, 0, 1, 1]
sdt._column_row_indices = [0, 1, 1, 0, 1]
sdt._column_data = [1, 5, 7, 1, 2]
sdt._num_entries = 5
sdt._is_data_on_buffer = False
sdt._load_scipy_matrix()


print(sdt[TARGET_INDEX])
