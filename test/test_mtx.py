import sparsedat

EXPECTED_VALUE = 3

sdt = sparsedat.wrappers.load_mtx("test/genes.tsv", "test/barcodes.tsv", "test/matrix.mtx")

original_value = sdt[20306, 117]
if original_value != EXPECTED_VALUE:
    raise ValueError("Value from importing mtx is not correct")

sdt.save("from_mtx.sdt")

sdt2 = sparsedat.Sparse_Data_Table("from_mtx.sdt")
sdt2.load()

reloaded_value = sdt2[20306, 117]

if reloaded_value != EXPECTED_VALUE:
    raise ValueError("Value from writing/reading after mtx is not correct")

print("Test successful")
