import sparsedat

sdt = sparsedat.Sparse_Data_Table("data/test.sdt")

print(sdt.row_names)
print(sdt.column_names)

my_metadata = {
    "Some thing": "blah",
    "Some other important thing": "blahblah"
}

sdt.metadata = my_metadata

sdt.save("data/test_metadata.sdt")

sdt_loaded = sparsedat.Sparse_Data_Table("data/test_metadata.sdt")

print(sdt_loaded.metadata)

if sdt_loaded.metadata != my_metadata:
    raise ValueError("Loaded metadata values do not match expected")

print("Test successful")
