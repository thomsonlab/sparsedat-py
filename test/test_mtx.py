import sparsedat

sdt = sparsedat.wrappers.load_mtx("test/genes.tsv", "test/barcodes.tsv", "test/matrix.mtx")

sdt.save("from_mtx.sdt")

sdt2 = sparsedat.Sparse_Data_Table("from_mtx.sdt")
sdt2.load()

print(sdt2[20306, 117])
