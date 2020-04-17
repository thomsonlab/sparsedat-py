import numpy
from scipy import sparse

from .Sparse_Data_Table import Sparse_Data_Table


def load_mtx(
        mtx_file_path,
        row_names_file_path=None,
        column_names_file_path=None
):

    if row_names_file_path is not None or column_names_file_path is not None:
        raise DeprecationWarning("Row and column names no longer part of MTX \
                                 loading -  set row and column names manually")

    sdt = Sparse_Data_Table()

    with open(mtx_file_path, "r") as matrix_file:

        row_column_values = []

        read_header = False

        while not read_header:

            line = matrix_file.readline().strip()

            if len(line) < 1 or line[0] == "#" or line[0] == "%":
                continue

            header_entries = line.split(" ")
            num_rows = int(header_entries[0])
            num_columns = int(header_entries[1])
            num_entries = int(header_entries[2])

            read_header = True

        for entry_index in range(num_entries):

            entry_line = matrix_file.readline().strip().split(" ")

            row_index = int(entry_line[0]) - 1
            column_index = int(entry_line[1]) - 1
            data_value = int(entry_line[2])

            row_column_values.append((row_index, column_index, data_value))

    sdt.from_row_column_values(row_column_values, num_rows, num_columns)

    return sdt


def to_mtx(sdt, row_names_file_path, column_names_file_path, matrix_file_path):

    with open(row_names_file_path, "w") as row_names_file:
        for row_name in sdt.row_names[::-1]:
            row_names_file.write("%s\t%s\t%s\n" %
                                 (row_name, row_name, row_name))

    with open(column_names_file_path, "w") as column_names_file:
        for column_name in sdt.column_names:
            column_names_file.write("%s\n" % column_name)

    with open(matrix_file_path, "w") as matrix_file:

        matrix_file.write(
            "%%MatrixMarket matrix coordinate integer general\n")
        matrix_file.write(
            "%metadata_json: {\"format_version\": 2, "
            "\"software_version\": \"3.0.0\"}\n")

        matrix_file.write("%i %i %i\n" %
                          (sdt.num_rows, sdt.num_columns, sdt.num_entries))

        column_index = 0

        for entry_index in range(sdt.num_entries):

            while entry_index >= sdt.column_start_indices[column_index] + \
                    sdt.column_lengths[column_index]:
                column_index += 1

            matrix_file.write("%i %i %i\n" % (
                sdt.num_rows - sdt.column_row_indices[entry_index],
                column_index + 1,
                sdt.column_data[entry_index]
            ))


def to_numpy(sdt):

    numpy_array = numpy.full(sdt.shape, fill_value=sdt.default_value)

    for row_index, row_start_index in enumerate(sdt._row_start_indices):

        num_row_entries = sdt._row_lengths[row_index]

        data_indices = range(row_start_index,
                             row_start_index + num_row_entries)

        for data_index in data_indices:
            column_index = sdt._row_column_indices[data_index]
            numpy_array[row_index, column_index] = sdt._row_data[data_index]

    return numpy_array


def to_coo(sdt):

    row_indices = [
        [row_index for _ in range(row_length)]
        for row_index, row_length in enumerate(sdt._row_lengths)]

    row_indices = [y for x in row_indices for y in x]

    scipy_array = sparse.coo_matrix(
        (sdt._row_data, (row_indices, sdt._row_column_indices)),
        shape=sdt.shape)

    return scipy_array


def to_csr(sdt):

    row_start_indices = sdt.row_start_indices
    row_start_indices = numpy.append(row_start_indices, sdt.num_entries - 1)

    scipy_array = sparse.csr_matrix(
        (sdt.row_data, sdt.row_column_indices, row_start_indices),
        shape=sdt.shape)

    return scipy_array


def to_csc(sdt):

    column_start_indices = sdt._column_start_indices
    column_start_indices = numpy.append(column_start_indices, sdt.num_entries - 1)

    scipy_array = sparse.csc_matrix(
        (sdt._column_data, sdt._column_row_indices, column_start_indices),
        shape=sdt.shape)

    return scipy_array

