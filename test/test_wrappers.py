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

sdt.save("test.sdt")

sdt = sparsedat.Sparse_Data_Table("test.sdt")
sdt.load()

python_list = sdt.to_list()
python_list_value = python_list[TARGET_INDEX[0]][TARGET_INDEX[1]]

if python_list_value != EXPECTED_VALUE:
    raise ValueError("Value after Python list conversion not as expected")

numpy_array = sparsedat.wrappers.to_numpy(sdt)
numpy_array_value = numpy_array[TARGET_INDEX]

if numpy_array_value != EXPECTED_VALUE:
    raise ValueError("Value after numpy array conversion not as expected")

scipy_csr_matrix = sparsedat.wrappers.to_csr(sdt)
scipy_csr_value = scipy_csr_matrix[TARGET_INDEX]

if scipy_csr_value != EXPECTED_VALUE:
    raise ValueError("Value after scipy CSR matrix conversion not as expected")

scipy_csc_matrix = sparsedat.wrappers.to_csc(sdt)
scipy_csc_value = scipy_csc_matrix[TARGET_INDEX]

if scipy_csc_value != EXPECTED_VALUE:
    raise ValueError("Value after scipy CSC matrix conversion not as expected")

print("Test successful")
