import time
import sparsedat
from scipy import sparse
import os

test_data_directory = "test"

start_time = time.time()
sdt = sparsedat.Sparse_Data_Table(
    os.path.join(test_data_directory, "from_mtx.sdt"))
duration = time.time() - start_time
print("Reading %i entries from file: %.2fs" % (sdt.num_entries, duration))

start_time = time.time()
sdt.save(os.path.join(test_data_directory, "out.sdt"))
duration = time.time() - start_time
print("Writing %i entries to file: %.2fs" % (sdt.num_entries, duration))

start_time = time.time()
python_list = sdt.to_list()
duration = time.time() - start_time
print("Converting to Python list: %.2fs" % duration)

start_time = time.time()
numpy_array = sparsedat.wrappers.to_numpy(sdt)
duration = time.time() - start_time
print("Converting to numpy array: %.2fs" % duration)

start_time = time.time()
scipy_coo_matrix = sparsedat.wrappers.to_coo(sdt)
duration = time.time() - start_time
print("Converting to scipy COO array: %.2fs" % duration)

start_time = time.time()
scipy_coo_file = open(os.path.join(test_data_directory, "scipy_coo.npz"), "wb")
sparse.save_npz(scipy_coo_file, scipy_coo_matrix, compressed=False)
scipy_coo_file.close()
duration = time.time() - start_time
print("Saving scipy COO array: %.2fs" % duration)

start_time = time.time()
scipy_coo_file = open(os.path.join(test_data_directory, "scipy_coo.npz"), "rb")
scipy_coo_matrix = sparse.load_npz(scipy_coo_file)
scipy_coo_file.close()
duration = time.time() - start_time
print("Loading scipy COO array: %.2fs" % duration)

start_time = time.time()
scipy_csr_matrix = sparsedat.wrappers.to_csr(sdt)
duration = time.time() - start_time
print("Converting to scipy sparse row matrix: %.2fs" % duration)

start_time = time.time()
scipy_csr_file = open(os.path.join(test_data_directory, "scipy_csr.npz"), "wb")
sparse.save_npz(scipy_csr_file, scipy_csr_matrix, compressed=False)
scipy_csr_file.close()
duration = time.time() - start_time
print("Saving scipy CSR array: %.2fs" % duration)

start_time = time.time()
scipy_csr_file = open(os.path.join(test_data_directory, "scipy_csr.npz"), "rb")
scipy_csr_matrix = sparse.load_npz(scipy_csr_file)
scipy_csr_file.close()
duration = time.time() - start_time
print("Loading scipy CSR array: %.2fs" % duration)

start_time = time.time()
scipy_csc_matrix = sparsedat.wrappers.to_csc(sdt)
duration = time.time() - start_time
print("Converting to scipy sparse column matrix: %.2fs" % duration)

start_time = time.time()
scipy_csc_file = open(os.path.join(test_data_directory, "scipy_csc.npz"), "wb")
sparse.save_npz(scipy_csc_file, scipy_csc_matrix, compressed=False)
scipy_csc_file.close()
duration = time.time() - start_time
print("Saving scipy CSC array: %.2fs" % duration)

start_time = time.time()
scipy_csc_file = open(os.path.join(test_data_directory, "scipy_csc.npz"), "rb")
scipy_csc_matrix = sparse.load_npz(scipy_csc_file)
scipy_csc_file.close()
duration = time.time() - start_time
print("Loading scipy CSC array: %.2fs" % duration)
