import time
import sparsedat
from scipy import sparse
import os
import numpy

import timeit

test_data_directory = "data"
scipy_coo_file_path = os.path.join(test_data_directory, "big_scipy_coo.npz")
scipy_csr_file_path = os.path.join(test_data_directory, "big_scipy_csr.npz")
scipy_csc_file_path = os.path.join(test_data_directory, "big_scipy_csc.npz")

if not os.path.exists(scipy_coo_file_path) or \
        not os.path.exists(scipy_csr_file_path) or \
        not os.path.exists(scipy_csc_file_path):

    start_time = time.time()
    sdt = sparsedat.Sparse_Data_Table(
        os.path.join(test_data_directory, "big.sdt"))
    duration = time.time() - start_time
    print("Preloaded (%i, %i) table from file: %.4fs" %
          (sdt.num_rows, sdt.num_columns, duration))

    start_time = time.time()
    sdt.load_all_data()
    duration = time.time() - start_time
    print("Fully loaded %i entries into memory: %.4fs" %
          (sdt.num_entries, duration))

    start_time = time.time()
    scipy_coo_matrix = sparsedat.wrappers.to_coo(sdt)
    duration = time.time() - start_time
    print("Converted to scipy COO array: %.4fs" % duration)

    start_time = time.time()
    scipy_coo_file = open(scipy_coo_file_path, "wb")
    sparse.save_npz(scipy_coo_file, scipy_coo_matrix, compressed=False)
    scipy_coo_file.close()
    duration = time.time() - start_time
    print("Saved scipy COO array: %.4fs" % duration)

    start_time = time.time()
    scipy_csr_matrix = sparsedat.wrappers.to_csr(sdt)
    duration = time.time() - start_time
    print("Converted to scipy CSR matrix: %.4fs" % duration)

    start_time = time.time()
    scipy_csr_file = open(scipy_csr_file_path, "wb")
    sparse.save_npz(scipy_csr_file, scipy_csr_matrix, compressed=False)
    scipy_csr_file.close()
    duration = time.time() - start_time
    print("Saved scipy CSR array: %.4fs" % duration)

    start_time = time.time()
    scipy_csc_matrix = sparsedat.wrappers.to_csc(sdt)
    duration = time.time() - start_time
    print("Converted to scipy CSC matrix: %.4fs" % duration)

    start_time = time.time()
    scipy_csc_file = open(scipy_csc_file_path, "wb")
    sparse.save_npz(scipy_csc_file, scipy_csc_matrix, compressed=False)
    scipy_csc_file.close()
    duration = time.time() - start_time
    print("Saved scipy CSC array: %.4fs" % duration)

    del sdt


# Slicing sequential full rows

NUM_REPETITIONS = 100
NUM_ROWS_TO_SLICE = 100
START_ROW = 1000

end_row = START_ROW + NUM_ROWS_TO_SLICE


def load_indices():

    sdt = sparsedat.Sparse_Data_Table(
        os.path.join(test_data_directory, "big.sdt"))

    del sdt


duration = timeit.timeit(load_indices, number=NUM_REPETITIONS)
print("SDT indices loaded from file: %.4fs" % duration)


def splice_CSR():
    scipy_csr_file = open(scipy_csr_file_path, "rb")
    scipy_csr_matrix = sparse.load_npz(scipy_csr_file)
    for _ in range(NUM_ROWS_TO_SLICE):
        row = numpy.random.randint(0, scipy_csr_matrix.shape[0])
        sliced = scipy_csr_matrix[row, 100]
    scipy_csr_file.close()


duration = timeit.timeit(splice_CSR, number=NUM_REPETITIONS)
print("Scipy CSR Loaded and sliced %i row(s) from file: %.4fs" %
      (NUM_ROWS_TO_SLICE, duration))


def splice_CSC():
    scipy_csc_file = open(scipy_csr_file_path, "rb")
    scipy_csc_matrix = sparse.load_npz(scipy_csc_file)

    for _ in range(NUM_ROWS_TO_SLICE):
        row = numpy.random.randint(0, scipy_csc_matrix.shape[0])
        sliced = scipy_csc_matrix[row, 100]
    scipy_csc_file.close()


duration = timeit.timeit(splice_CSC, number=NUM_REPETITIONS)
print("Scipy CSC Loaded and sliced %i row(s) from file: %.4fs" %
      (NUM_ROWS_TO_SLICE, duration))


def splice_SDT():
    sdt = sparsedat.Sparse_Data_Table(
        os.path.join(test_data_directory, "big.sdt"))

    for _ in range(NUM_ROWS_TO_SLICE):
        row = numpy.random.randint(0, sdt.shape[0])
        sliced = sdt[row, 100]

    del sdt


duration = timeit.timeit(splice_SDT, number=NUM_REPETITIONS)
print("SDT loaded and sliced %i row(s) from file: %.4fs" %
      (NUM_ROWS_TO_SLICE, duration))
