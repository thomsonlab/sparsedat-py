import sparsedat
import os

test_data_directory = "data"

EXPECTED_VALUE = 2

#TARGET_INDEX = (13697, 46)
TARGET_INDEX = (16596, 123)

sdt = sparsedat.wrappers.load_mtx(
    os.path.join(test_data_directory, "genes.tsv"),
    os.path.join(test_data_directory, "barcodes.tsv"),
    os.path.join(test_data_directory, "matrix.mtx"))

# original_value = sdt[TARGET_INDEX]
# if original_value != EXPECTED_VALUE:
#     raise ValueError("Value from importing mtx is not correct")

sdt.save(os.path.join(test_data_directory, "big.sdt"))

scipy_sparse_matrix = sparsedat.wrappers.to_csr(sdt)
value_in_scipy = scipy_sparse_matrix[TARGET_INDEX]

if value_in_scipy != EXPECTED_VALUE:
    raise ValueError("Value from loading all data and converstion to scipy "
        "is not correct")


sdt2 = sparsedat.Sparse_Data_Table(
    os.path.join(test_data_directory, "from_mtx.sdt"))

reloaded_value = sdt2[TARGET_INDEX]

if reloaded_value != EXPECTED_VALUE:
    raise ValueError("Value from writing/reading after mtx is not correct")

print("Test successful")
