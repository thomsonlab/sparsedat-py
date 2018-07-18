from .Sparse_Data_Table import Sparse_Data_Table


def load_mtx(row_names_file_path, column_names_file_path, matrix_file_path):

    sdt = Sparse_Data_Table()

    with open(row_names_file_path, "r") as row_names_file:
        row_names = [line[:-1] for line in row_names_file]

    with open(column_names_file_path, "r") as column_names_file:
        column_names = [line[:-1] for line in column_names_file]

    with open(matrix_file_path, "r") as matrix_file:

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

    sdt.set_column_names(column_names)
    sdt.set_row_names(row_names)

    return sdt
