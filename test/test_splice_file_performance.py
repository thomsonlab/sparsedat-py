import time
import sparsedat
from scipy import sparse
from scipy import io as sio
import os
import numpy

import timeit

test_data_directory = "data"
scipy_coo_file_path = os.path.join(test_data_directory, "big_scipy_coo.npz")
scipy_csr_file_path = os.path.join(test_data_directory, "big_scipy_csr.npz")
scipy_csc_file_path = os.path.join(test_data_directory, "big_scipy_csc.npz")
mtx_file_path = os.path.join(test_data_directory, "matrix.mtx")

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

NUM_REPETITIONS = 1
ROWS_TO_SLICE = slice(1000, 1500, None)
COLUMNS_TO_SLICE = slice(400, 401, None)

num_rows = 32738 #ROWS_TO_SLICE.stop - ROWS_TO_SLICE.start


def load_indices():

    sdt = sparsedat.Sparse_Data_Table(
        os.path.join(test_data_directory, "big.sdt"))

    del sdt


# duration = timeit.timeit(load_indices, number=NUM_REPETITIONS) / NUM_REPETITIONS
# print("SDT indices loaded from file: %.4fs" % duration)


def splice_MTX():

    mtx_matrix = sio.mmread(mtx_file_path).tocsc()
    sliced = mtx_matrix[ROWS_TO_SLICE, COLUMNS_TO_SLICE].todense()


# duration = timeit.timeit(splice_MTX, number=NUM_REPETITIONS) / NUM_REPETITIONS
# print("Scipy loaded mtx and sliced %i row(s) from file: %.4fs" %
#       (num_rows, duration))


def splice_COO():
    scipy_coo_file = open(scipy_coo_file_path, "rb")
    scipy_coo_matrix = sparse.load_npz(scipy_coo_file)
    scipy_coo_matrix = scipy_coo_matrix.tocsr()
    sliced = scipy_coo_matrix[ROWS_TO_SLICE, COLUMNS_TO_SLICE].todense()
    scipy_coo_file.close()


# duration = timeit.timeit(splice_COO, number=NUM_REPETITIONS) / NUM_REPETITIONS
# print("Scipy COO Loaded and sliced %i row(s) from file: %.4fs" %
#       (num_rows, duration))


def splice_CSC():
    scipy_csc_file = open(scipy_csr_file_path, "rb")
    scipy_csc_matrix = sparse.load_npz(scipy_csc_file)
    sliced = scipy_csc_matrix[ROWS_TO_SLICE, COLUMNS_TO_SLICE].todense()
    # print(sliced)
    scipy_csc_file.close()

duration = timeit.timeit(splice_CSC, number=NUM_REPETITIONS) / NUM_REPETITIONS
print("Scipy CSC Loaded and sliced %i row(s) from file: %.4fs" %
      (num_rows, duration))

def splice_SDT():
    sdt = sparsedat.Sparse_Data_Table(
        os.path.join(test_data_directory, "big.sdt"),
        load_on_demand=True)
    sliced = sdt[ROWS_TO_SLICE, COLUMNS_TO_SLICE]
    # print(sliced)

    del sdt

duration = timeit.timeit(splice_SDT, number=NUM_REPETITIONS) / NUM_REPETITIONS
print("SDT loaded and sliced %i row(s) from file: %.4fs" %
      (num_rows, duration))


def splice_CSR():
    scipy_csr_file = open(scipy_csr_file_path, "rb")
    scipy_csr_matrix = sparse.load_npz(scipy_csr_file)
    sliced = scipy_csr_matrix[ROWS_TO_SLICE, COLUMNS_TO_SLICE].todense()
    # print(sliced)
    scipy_csr_file.close()


duration = timeit.timeit(splice_CSR, number=NUM_REPETITIONS) / NUM_REPETITIONS
print("Scipy CSR Loaded and sliced %i row(s) from file: %.4fs" %
      (num_rows, duration))
