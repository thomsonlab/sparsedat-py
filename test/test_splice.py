import os
import random
import time
import struct
from scipy import io
import sys

bytes_per_value = 4
bytes_per_index = 8


def get_cell_names(path):

    barcodes_file_path = os.path.join(path, "barcodes.tsv")

    with open(barcodes_file_path, "r") as barcodes_file:
        cell_names = [line[:-1] for line in barcodes_file]

    return cell_names


def get_cell_file_size(path):

    index_file_path = os.path.join(path, "index.idx")

    total_bytes = os.path.getsize(index_file_path)

    return total_bytes


def get_cell_start_bytes(path):

    index_file_path = os.path.join(path, "index.idx")

    total_bytes = get_cell_file_size(path)
    cell_start_bytes = []

    index_file = open(index_file_path, "rb")

    current_byte = 0

    while current_byte < total_bytes:
        cell_start_bytes.append(struct.unpack("q", index_file.read(bytes_per_index))[0])
        current_byte += bytes_per_index

    index_file.close()

    return cell_start_bytes

def get_sparse_matrix_components(
        row_names_file_path, column_names_file_path, data_file_path):

    with open(row_names_file_path, "r") as row_names_file:
        # Read until tab to ignore user-friendly name for now
        row_names = [line[:line.find("\t")] for line in row_names_file]

    with open(column_names_file_path, "r") as column_names_file:
        column_names = [line[:-1] for line in column_names_file]

    with open(data_file_path, "r") as data_file:

        row_column_values = []

        line_index = 0
        for line in data_file:

            if line_index < 3:
                line_index += 1
                continue

            row_column_value = line[:-1].split(" ")
            row_column_value = [int(value) for value in row_column_value]

            row_column_values.append(row_column_value)

    return row_names, column_names, row_column_values


def write_binary_sparse_column_row_matrix(
        row_names, column_names, data, output_dir_path):

    data_file_path = os.path.join(output_dir_path, "data.db")
    index_file_path = os.path.join(output_dir_path, "index.idx")

    # Put data in column-accessible format
    column_row_dictionaries = [dict() for x in range(len(column_names))]
    for cell in data:
        row_index = cell[0] - 1
        column_index = cell[1] - 1
        value = cell[2]

        column_row_dictionaries[column_index][row_index] = value

    data_file = open(data_file_path, "wb")
    index_file = open(index_file_path, "wb")

    current_byte_index = 0

    for column_index, column_name in enumerate(column_names):

        # Get all the row values for this column
        row_values = column_row_dictionaries[column_index]

        if column_index == 0:
            print("Writing %i for cell %s" % (len(row_values), column_name))

        num_values = len(row_values)
        byte_size = num_values * (2 * bytes_per_value)

        index_file.write(struct.pack("q", current_byte_index))

        current_byte_index += byte_size

        # Iterate through the row values sorted:
        for row_index, value in sorted(row_values.items()):
            data_file.write(struct.pack("i", row_index))
            data_file.write(struct.pack("i", value))

    data_file.close()
    index_file.close()

directory_path = "."

genes_file_path = os.path.join(directory_path, "genes.tsv")
barcodes_file_path = os.path.join(directory_path, "barcodes.tsv")
mtx_file_path = os.path.join(directory_path, "matrix.mtx")

print("Reading .mtx")
start_time = time.time()
genes, barcodes, gene_counts = get_sparse_matrix_components(
    genes_file_path, barcodes_file_path, mtx_file_path)
end_time = time.time()

print(end_time-start_time)

print("Writing binary format")
start_time = time.time()
write_binary_sparse_column_row_matrix(
    genes, barcodes, gene_counts, directory_path)
end_time = time.time()

print(end_time-start_time)

cell_names = get_cell_names(directory_path)

some_random_cells = random.sample(list(enumerate(cell_names)), 100)

print("Getting gene counts for 3000 random cells")

print("Getting start indices for all cells")

start_time = time.time()

cell_start_bytes = get_cell_start_bytes(directory_path)

cell_indices_bytes = []

for cell_index, cell in some_random_cells:
    start_byte = cell_start_bytes[cell_index]
    if cell_index + 1 < len(cell_start_bytes):
        end_byte = cell_start_bytes[cell_index + 1] - 1
    else:
        end_byte = get_cell_file_size(directory_path)

    cell_indices_bytes.append((cell_index, start_byte, end_byte))

cell_indices_bytes = sorted(cell_indices_bytes, key=lambda x: x[1])

end_time = time.time()

print(end_time - start_time)

print("Getting data from binary file")

start_time = time.time()

data_file_path = os.path.join(directory_path, "data.db")

cell_data_file = open(data_file_path, "rb")


for cell_index, start_byte, end_byte in cell_indices_bytes:

    cell_data_file.seek(start_byte)
    num_bytes_to_read = end_byte-start_byte+1

    num_genes = int(num_bytes_to_read / (2*bytes_per_value))

    for gene_index in range(num_genes):

        gene_id = struct.unpack("i", cell_data_file.read(bytes_per_value))[0]
        gene_count = struct.unpack("i", cell_data_file.read(bytes_per_value))[0]

end_time = time.time()

print(end_time-start_time)

print("Reading sparse matrix")

start_time = time.time()
sparsey = io.mmread(mtx_file_path)

end_time = time.time()
print(end_time - start_time)

cell_data_file.close()
