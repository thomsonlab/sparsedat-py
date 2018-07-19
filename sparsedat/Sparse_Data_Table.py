import struct
import zlib
import pickle
import numpy
import io
import time

from .Data_Type import Data_Type
from .Metadata_Type import Metadata_Type


CURRENT_VERSION = 1
USE_COMPRESSION = False
HEADER_SIZE = 45


class Sparse_Data_Table:

    @staticmethod
    def get_num_bytes(data_type, max_value, min_value=None):

        if data_type == Data_Type.UINT:
            if max_value <= 65535:
                return 2
            elif max_value <= 4294967295:
                return 4
            elif max_value <= 18446744073709551615:
                return 8
            else:
                raise ValueError("Max integer value greater than 8 bytes \
                        not currently supported")

        elif data_type == Data_Type.INT:

            if max_value <= 32767 and min_value >= -32768:
                return 2
            elif max_value <= 2147483647 and min_value >= -2147483648:
                return 4
            elif max_value <= 9223372036854775807 \
                    and min_value >= -9223372036854775808:
                return 8

        elif data_type == Data_Type.FLOAT:

            return 8
        else:
            raise NotImplementedError("Only support int, uint, and float")

    @staticmethod
    def get_pack_format(data_type, data_size):

        if data_type == Data_Type.UINT:
            if data_size == 2:
                return "H"
            elif data_size == 4:
                return "I"
            elif data_size == 8:
                return "Q"
            else:
                raise NotImplementedError(
                    "Do not support packing data more than 8 bytes")
        elif data_type == Data_Type.INT:
            if data_size == 2:
                return "h"
            elif data_size == 4:
                return "i"
            elif data_size == 8:
                return "q"
            else:
                raise NotImplementedError(
                    "Do not support packing data more than 8 bytes")
        elif data_type == Data_Type.FLOAT:
            if data_size == 4:
                return "f"
            elif data_size == 8:
                return "d"
            else:
                raise NotImplementedError(
                    "Do not support packing data more than 8 bytes")

    def __init__(
            self,
            file_path=None,
            load_on_demand=True):

        self._file_path = None
        self._data_buffer = None
        self._version = CURRENT_VERSION
        self._metadata = {}
        self._data_type = None
        self._data_size = None
        self._num_rows = None
        self._num_columns = None
        self._num_entries = None
        self._row_start_indices = None
        self._row_column_indices = None
        self._row_lengths = None
        self._column_start_indices = None
        self._column_row_indices = None
        self._column_lengths = None
        self._default_value = None
        self._row_data = None
        self._column_data = None

        self._num_bytes_row_index = None
        self._num_bytes_column_index = None
        self._num_bytes_row_byte = None
        self._num_bytes_column_byte = None
        self._max_row_byte = None
        self._max_column_byte = None
        self._pack_format_row_byte = None
        self._pack_format_column_byte = None
        self._pack_format_row_index = None
        self._pack_format_column_index = None
        self._pack_format_data = None

        # Data relevant for real time file I/O
        self._metadata_size = None
        self._is_metadata_loaded = False
        self._row_data_start_byte = None
        self._column_data_start_byte = None
        self._is_data_loaded = False
        self._load_on_demand = load_on_demand

        self._unpack_data = None
        self._unpack_column_index = None
        self._unpack_row_index = None

        if file_path is not None:
            self.set_file_path(file_path)

    def __del__(self):

        if self._data_buffer is not None:
            self._data_buffer.close()

    def __setitem__(self, index, value):
        raise NotImplementedError("Sparse_Data_Table is read-only for now")

    def __getitem__(self, index):

        if isinstance(index, int):
            raise NotImplementedError(
                "Haven't implemented just row slicing - use a full index spec")

        row_index, column_index = index

        if isinstance(row_index, slice) or isinstance(column_index, slice):

            if isinstance(row_index, slice):
                if row_index.step is not None:
                    raise NotImplementedError(
                        "Haven't implemented step slicing")

                if row_index.start is None:
                    start_row = 0
                else:
                    start_row = row_index.start
                if row_index.stop is None:
                    stop_row = self._num_rows
                else:
                    stop_row = row_index.stop
            else:
                start_row = row_index
                stop_row = row_index + 1

            if isinstance(column_index, slice):
                if column_index.step is not None:
                    raise NotImplementedError(
                        "Haven't implemented step slicing")

                if column_index.start is None:
                    start_column = 0
                else:
                    start_column = column_index.start
                if column_index.stop is None:
                    stop_column = self._num_columns
                else:
                    stop_column = column_index.stop
            else:
                start_column = column_index
                stop_column = column_index + 1

            num_rows = stop_row - start_row
            num_columns = stop_column - start_column

            subarray = numpy.full(
                (num_rows, num_columns),
                fill_value=self._default_value)

            subarray_row_index = 0
            subarray_column_index = 0

            # If we have less rows, then we should slice along rows first
            if num_rows <= num_columns:

                # byte_calculation_time = 0
                # unpack_time = 0
                # array_write_time = 0
                # seek_time = 0
                # indexing_time = 0
                # array_copy_time = 0

                bytes_per_entry = self._num_bytes_column_index + self._data_size

                for target_row_index in range(start_row, stop_row):

                    # start_time = time.time()
                    num_row_entries = self._row_lengths[target_row_index]

                    if num_row_entries == 0:
                        continue

                    row_entry_start_index = self._row_start_indices[target_row_index]
                    # indexing_time += time.time() - start_time

                    row_entry_end_index = row_entry_start_index + num_row_entries

                    byte_index = self._row_data_start_byte + \
                                 row_entry_start_index * bytes_per_entry

                    self._data_buffer.seek(byte_index)

                    # entry_column_index_bytes = bytearray()
                    # data_value_bytes = bytearray()
                    #
                    # for row_entry_index in range(row_entry_start_index,
                    #                              row_entry_end_index):
                    #
                    #     # start_time = time.time()
                    #
                    #     # byte_calculation_time += time.time() - start_time
                    #
                    #     # start_time = time.time()
                    #     # seek_time += time.time() - start_time
                    #     #
                    #     # start_time = time.time()
                    #
                    #     entry_column_index_bytes += self._data_buffer.read(
                    #             self._num_bytes_column_index)
                    #
                    #     # entry_column_index = numpy.ndarray(
                    #     #     (1,),
                    #     #     self._pack_format_column_index,
                    #     #     some_bytes
                    #     # )[0]
                    #
                    #     # entry_column_index = self._unpack_column_index(
                    #     #     self._data_buffer.read(
                    #     #         self._num_bytes_column_index))[0]
                    #     # unpack_time += time.time() - start_time
                    #
                    #     # if entry_column_index < start_column:
                    #     #     row_entry_index += 1
                    #     #     continue
                    #     # if entry_column_index >= stop_column:
                    #     #     break
                    #
                    #     # start_time = time.time()
                    #     data_value_bytes += self._data_buffer.read(
                    #         self._data_size)
                    #     # unpack_time += time.time() - start_time
                    #
                    #     # start_time = time.time()
                    #     # subarray[target_row_index - start_row,
                    #     #     entry_column_index - start_column] = data_value
                    #     # array_write_time += time.time() - start_time

                    self._data_buffer.seek(byte_index)

                    # start_time = time.time()
                    column_indices = numpy.ndarray(
                        (num_row_entries,),
                        self._pack_format_column_index,
                        self._data_buffer.read((self._num_bytes_column_index + self._data_size) * num_row_entries),
                        strides=self._data_size
                    )

                    self._data_buffer.seek(byte_index + self._num_bytes_column_index)

                    read_length = (self._num_bytes_column_index + self._data_size) * num_row_entries - self._num_bytes_column_index

                    row_data_array = numpy.ndarray(
                        (num_row_entries,),
                        self._pack_format_data,
                        self._data_buffer.read(read_length),
                        strides=self._num_bytes_column_index
                    )

                    # array_copy_time += time.time() - start_time

                    # start_time = time.time()
                    target_row_index = target_row_index - start_row

                    start_row_entry_index = 0
                    end_row_entry_index = num_row_entries - 1

                    while column_indices[start_row_entry_index] < start_column:
                        start_row_entry_index += 1

                    while column_indices[end_row_entry_index] >= stop_column:
                        end_row_entry_index -= 1

                    subarray_column_indices = numpy.subtract(column_indices, start_column)

                    for row_entry_index in range(start_row_entry_index, end_row_entry_index):

                        subarray[target_row_index, subarray_column_indices[row_entry_index]] = \
                            row_data_array[row_entry_index]

                # print("Byte calculation time: %.4fs" % byte_calculation_time)
                # print("Unpack time: %.4fs" % unpack_time)
                # print("Array write time: %.4fs" % array_write_time)
                # print("Seek time: %.4fs" % seek_time)
                # print("Indexing time: %.4fs" % indexing_time)
                # print("Array copy time: %.4fs" % array_copy_time)
            else:
                for target_column_index in range(start_column, stop_column):
                    subarray[:, subarray_column_index] = \
                        self.get_column(target_column_index, row_index)
                    subarray_column_index += 1

            return subarray

        num_row_entries = self._row_lengths[row_index]
        num_column_entries = self._column_lengths[column_index]

        if num_row_entries < num_column_entries:

            row_start_index = self._row_start_indices[row_index]
            row_end_index = row_start_index + num_row_entries

            if self._load_on_demand and not self._is_data_loaded:

                row_entry_index = row_start_index

                while row_entry_index <= row_end_index:

                    byte_index = self._row_data_start_byte + \
                                     row_entry_index * (
                                         self._num_bytes_column_index +
                                         self._data_size)

                    self._data_buffer.seek(byte_index)
                    entry_column_index = self._unpack_column_index(
                        self._data_buffer.read(self._num_bytes_column_index))[0]

                    if column_index == entry_column_index:
                        data_value = self._unpack_data(
                            self._data_buffer.read(self._data_size))[0]
                        return data_value

                    row_entry_index += 1

                return self._default_value
            else:
                try:
                    column_data_index = \
                        self._row_column_indices[
                            row_start_index:row_end_index]\
                        .index(column_index)
                    return self._row_data[row_start_index + column_data_index]
                except ValueError:
                    return self._default_value
        else:
            column_start_index = self._column_start_indices[column_index]
            column_end_index = column_start_index + num_column_entries

            if self._load_on_demand and not self._is_data_loaded:

                column_index = column_start_index

                while column_index <= column_end_index:

                    byte_index = self._column_data_start_byte + \
                                    column_index * (
                                         self._num_bytes_row_index +
                                         self._data_size)

                    self._data_buffer.seek(byte_index)
                    entry_row_index = self._unpack_row_index(
                        self._data_buffer.read(self._num_bytes_row_index))[0]

                    if row_index == entry_row_index:
                        data_value = self._unpack_data(
                            self._data_buffer.read(self._data_size))[0]
                        return data_value

                    column_index += 1

                return self._default_value
            else:
                try:
                    row_data_index = \
                        self._column_row_indices[
                            column_start_index:column_end_index]\
                        .index(row_index)
                    return self._column_data[
                        column_start_index + row_data_index]
                except ValueError:
                    return self._default_value

    def get_row(self, row_index, column_slice=None):

        if column_slice is None:
            column_start_index = 0
            column_stop_index = self._num_columns
        else:
            if column_slice.start is None:
                column_start_index = 0
            else:
                column_start_index = column_slice.start
            if column_slice.stop is None:
                column_stop_index = self._num_columns
            else:
                column_stop_index = column_slice.stop

        num_columns = column_stop_index - column_start_index

        row = numpy.full((num_columns,), fill_value=self._default_value)

        num_row_entries = self._row_lengths[row_index]
        row_entry_start_index = self._row_start_indices[row_index]
        row_entry_end_index = row_entry_start_index + num_row_entries

        row_entry_index = row_entry_start_index

        if self._load_on_demand and not self._is_data_loaded:

            while row_entry_index < row_entry_end_index:

                byte_index = self._row_data_start_byte + \
                                row_entry_index * (
                                     self._num_bytes_column_index +
                                     self._data_size)

                self._data_buffer.seek(byte_index)
                entry_column_index = self._unpack_column_index(
                    self._data_buffer.read(self._num_bytes_column_index))[0]

                if entry_column_index < column_start_index:
                    row_entry_index += 1
                    continue
                if entry_column_index >= column_stop_index:
                    break

                data_value = self._unpack_data(
                    self._data_buffer.read(self._data_size))[0]
                row[entry_column_index - column_start_index] = data_value

                row_entry_index += 1
        else:

            while row_entry_index < row_entry_end_index:

                column_index = self._row_column_indices[row_entry_index]

                if column_index < column_start_index:
                    row_entry_index += 1
                    continue

                if column_index >= column_stop_index:
                    break

                data_value = self._row_data[row_entry_index]

                row[column_index - column_start_index] = data_value

                row_entry_index += 1

        return row

    def get_column(self, column_index, row_slice=None):

        if row_slice is None:
            row_start_index = 0
            row_stop_index = self._num_rows
        else:
            if row_slice.start is None:
                row_start_index = 0
            else:
                row_start_index = row_slice.start
            if row_slice.stop is None:
                row_stop_index = self._num_columns
            else:
                row_stop_index = row_slice.stop

        num_rows = row_stop_index - row_start_index

        column = numpy.full((num_rows,), fill_value=self._default_value)

        num_column_entries = self._column_lengths[column_index]
        column_entry_start_index = self._row_start_indices[column_index]
        column_entry_end_index = column_entry_start_index + num_column_entries

        column_entry_index = column_entry_start_index

        if self._load_on_demand and not self._is_data_loaded:

            while column_entry_index < column_entry_end_index:

                byte_index = self._column_data_start_byte + \
                                column_entry_index * (
                                     self._num_bytes_row_index +
                                     self._data_size)

                self._data_buffer.seek(byte_index)
                entry_row_index = self._unpack_row_index(
                    self._data_buffer.read(self._num_bytes_row_index))[0]

                if entry_row_index < row_start_index:
                    column_entry_index += 1
                    continue
                if entry_row_index >= row_stop_index:
                    break

                data_value = self._unpack_data(
                    self._data_buffer.read(self._data_size))[0]
                column[entry_row_index - row_start_index] = data_value

                column_entry_index += 1
        else:

            while column_entry_index < column_entry_end_index:

                row_index = self._column_row_indices[column_entry_index]

                if row_index < row_start_index:
                    column_entry_index += 1
                    continue

                if row_index >= row_stop_index:
                    break

                data_value = self._column_data[column_entry_index]

                column[row_index - row_start_index] = data_value

                column_entry_index += 1

        return column

    def to_list(self):

        array = [
            [self._default_value for _ in range(self._num_columns)]
            for _ in range(self._num_rows)]

        for row_index, row_start_index in enumerate(self._row_start_indices):

            num_row_entries = self._row_lengths[row_index]

            data_indices = range(row_start_index,
                                 row_start_index + num_row_entries)

            for data_index in data_indices:
                column_index = self._row_column_indices[data_index]
                array[row_index][column_index] = self._row_data[data_index]

        return array

    def from_row_column_values(self,
                               row_column_values,
                               num_rows=None,
                               num_columns=None,
                               default_value=0):

        self._version = CURRENT_VERSION

        first_entry = row_column_values[0]
        first_value = first_entry[2]

        max_column_index = first_entry[0]
        max_row_index = first_entry[1]
        max_value = first_value
        has_negative_values = False
        min_value = first_value

        row_column_value_map = {}
        column_row_value_map = {}

        for row_index, column_index, data_value in row_column_values:

            if row_index > max_row_index:
                if num_rows is not None and row_index >= num_rows:
                    raise ValueError("Row index %i is more than the number \
                        of rows" % row_index)
                max_row_index = row_index
            if column_index > max_column_index:
                if num_columns is not None and column_index >= num_columns:
                    raise ValueError("Column index %i is more than the number \
                        of columns" % column_index)
                max_column_index = column_index
            if data_value > max_value:
                max_value = data_value
            if data_value < min_value:
                min_value = data_value
            if not has_negative_values and data_value < 0:
                has_negative_values = True

            if row_index not in row_column_value_map:
                row_column_value_map[row_index] = {}

            if column_index not in column_row_value_map:
                column_row_value_map[column_index] = {}

            row_column_value_map[row_index][column_index] = data_value
            column_row_value_map[column_index][row_index] = data_value

        if num_rows is None:
            self._num_rows = max_row_index + 1
        else:
            self._num_rows = num_rows

        if num_columns is None:
            self._num_columns = max_column_index + 1
        else:
            self._num_columns = num_columns

        self._num_entries = len(row_column_values)

        entry_index = 0

        self._row_start_indices = []
        self._row_lengths = []
        self._row_column_indices = []
        self._row_data = []

        for row_index in range(self._num_rows):

            self._row_start_indices.append(entry_index)

            if row_index not in row_column_value_map:
                self._row_lengths.append(0)
                continue

            column_value_map = row_column_value_map[row_index]

            num_entries = len(column_value_map)

            entry_index += num_entries

            self._row_lengths.append(num_entries)

            for column_index in sorted(column_value_map.keys()):
                self._row_column_indices.append(column_index)
                self._row_data.append(column_value_map[column_index])

        entry_index = 0

        self._column_start_indices = []
        self._column_lengths = []
        self._column_row_indices = []
        self._column_data = []

        for column_index in range(self._num_columns):

            self._column_start_indices.append(entry_index)

            if column_index not in column_row_value_map:
                self._column_lengths.append(0)
                continue

            row_value_map = column_row_value_map[column_index]

            num_entries = len(row_value_map)

            entry_index += num_entries

            self._column_lengths.append(num_entries)

            for row_index in sorted(row_value_map.keys()):
                self._column_row_indices.append(row_index)
                self._column_data.append(row_value_map[row_index])

        if isinstance(first_value, int):
            if not has_negative_values:
                self._data_type = Data_Type.UINT
            else:

                self._data_type = Data_Type.INT
        elif isinstance(first_value, float):
            self._data_type = Data_Type.FLOAT

        self._data_size = Sparse_Data_Table.get_num_bytes(
            self._data_type, max_value, min_value=min_value)

        self._default_value = default_value

    @property
    def row_names(self):

        if self._load_on_demand and not self._is_metadata_loaded:
            self.load_all_metadata()

        return self._metadata[Metadata_Type.ROW_NAMES]

    @property
    def column_names(self):

        if not self._is_metadata_loaded:
            self.load_all_metadata()

        return self._metadata[Metadata_Type.COLUMN_NAMES]

    @row_names.setter
    def row_names(self, new_row_names):

        if self._num_rows is not None and len(new_row_names) != self._num_rows:
            raise ValueError("Row names must match number of rows!")

        self._metadata[Metadata_Type.ROW_NAMES] = new_row_names

    @column_names.setter
    def column_names(self, new_column_names):

        if self._num_columns is not None and \
                len(new_column_names) != self._num_columns:
            raise ValueError("Column names must match number of columns!")

        self._metadata[Metadata_Type.COLUMN_NAMES] = new_column_names

    @property
    def num_rows(self):
        return self._num_rows

    @property
    def num_columns(self):
        return self._num_columns

    @property
    def shape(self):
        return self._num_rows, self._num_columns

    @property
    def default_value(self):
        return self._default_value

    @property
    def num_entries(self):
        return self._num_entries

    @property
    def metadata(self):

        if self._load_on_demand and not self._is_metadata_loaded:
            self.load_all_metadata()

        return self._metadata[Metadata_Type.USER_METADATA]

    @metadata.setter
    def metadata(self, user_metadata):

        if self._load_on_demand and not self._is_metadata_loaded:
            self.load_all_metadata()

        self._metadata[Metadata_Type.USER_METADATA] = user_metadata

    def _calculate_formats(self):

        self._num_bytes_row_index = self.get_num_bytes(
            Data_Type.UINT, self._num_rows)
        self._num_bytes_column_index = self.get_num_bytes(
            Data_Type.UINT, self._num_columns)

        self._max_row_byte = (self._num_bytes_column_index +
                              self._data_size) * self._num_entries
        self._max_column_byte = (self._num_bytes_row_index +
                                 self._data_size) * self._num_entries

        self._num_bytes_row_byte = self.get_num_bytes(
            Data_Type.UINT, self._max_row_byte)
        self._num_bytes_column_byte = self.get_num_bytes(
            Data_Type.UINT, self._max_column_byte)

        self._pack_format_row_byte = Sparse_Data_Table.get_pack_format(
            Data_Type.UINT, self._num_bytes_row_byte)
        self._pack_format_column_byte = Sparse_Data_Table.get_pack_format(
            Data_Type.UINT, self._num_bytes_column_byte)

        self._pack_format_row_index = Sparse_Data_Table.get_pack_format(
            Data_Type.UINT, self._num_bytes_row_index)
        self._pack_format_column_index = Sparse_Data_Table.get_pack_format(
            Data_Type.UINT, self._num_bytes_column_index)

        self._pack_format_data = Sparse_Data_Table.get_pack_format(
            self._data_type, self._data_size)

    def save(self, file_path=None):

        if file_path is None:
            file_path = self._file_path

        if self._load_on_demand:
            if not self._is_metadata_loaded:
                self.load_all_metadata()
            if not self._is_data_loaded:
                self.load_all_data()

        data_buffer = io.BytesIO()

        version_string = bytes("SDTv%04d" % CURRENT_VERSION, "UTF-8")

        data_buffer.write(version_string)

        data_buffer.write(struct.pack("B", self._data_type.value))

        data_buffer.write(struct.pack("I", self._data_size))

        data_buffer.write(struct.pack("Q", self._num_rows))

        data_buffer.write(struct.pack("Q", self._num_columns))

        data_buffer.write(struct.pack("Q", self._num_entries))

        metadata_bytes = self.get_encoded_metadata()

        data_buffer.write(struct.pack("Q", len(metadata_bytes)))

        data_buffer.write(metadata_bytes)

        self._calculate_formats()

        for row_index in range(self._num_rows):

            row_start_byte = self._row_start_indices[row_index] * \
                (self._num_bytes_column_index + self._data_size)

            data_buffer.write(
                struct.pack(self._pack_format_row_byte, row_start_byte))

        for column_index in range(self._num_columns):

            column_start_byte = self._column_start_indices[column_index] * \
                (self._num_bytes_row_index + self._data_size)

            data_buffer.write(
                struct.pack(self._pack_format_column_byte, column_start_byte))

        data_buffer.write(struct.pack(self._pack_format_data,
                                      self._default_value))

        for row_index, row_start_index in enumerate(self._row_start_indices):

            num_row_entries = self._row_lengths[row_index]

            data_indices = range(row_start_index,
                                 row_start_index + num_row_entries)

            for entry_index in data_indices:

                column_index = self._row_column_indices[entry_index]
                data_value = self._row_data[entry_index]

                data_buffer.write(
                    struct.pack(self._pack_format_column_index, column_index))
                data_buffer.write(struct.pack(self._pack_format_data, data_value))

        for column_index, column_start_index in enumerate(
                self._column_start_indices):

            num_column_entries = self._column_lengths[column_index]

            data_indices = range(column_start_index,
                                 column_start_index + num_column_entries)

            for entry_index in data_indices:

                column_index = self._column_row_indices[entry_index]
                data_value = self._column_data[entry_index]

                data_buffer.write(
                    struct.pack(self._pack_format_row_index, column_index))
                data_buffer.write(struct.pack(self._pack_format_data, data_value))

        data_buffer.seek(0)
        with open(file_path, "wb") as data_file:
            if USE_COMPRESSION:
                data_file.write(zlib.compress(data_buffer.read()))
            else:
                data_file.write(data_buffer.read())

    def get_encoded_metadata(self):

        metadata_bytes = bytearray()

        # The metadata array starts with the number of metadata entries
        metadata_bytes += struct.pack("I", len(self._metadata))

        # The content of the metadata, excludes index
        metadata_content_bytes = bytearray()

        byte_index = 0

        # Now we index the metadata - list each metadata id, followed by its
        # starting index
        for metadata_type, metadata in self._metadata.items():

            metadata_bytes += struct.pack("I", metadata_type.value)
            metadata_bytes += struct.pack("I", byte_index)

            if metadata_type == Metadata_Type.ROW_NAMES or \
                    metadata_type == Metadata_Type.COLUMN_NAMES:

                for name in metadata:

                    name_bytes = name.encode("utf-8")
                    name_length = len(name_bytes)

                    metadata_content_bytes.extend(
                        struct.pack("I", name_length))
                    metadata_content_bytes.extend(name_bytes)

                    byte_index += 4 + name_length
            elif metadata_type == Metadata_Type.USER_METADATA:

                user_metadata_bytes = pickle.dumps(metadata)
                user_metadata_length = len(user_metadata_bytes)

                metadata_content_bytes.extend(user_metadata_bytes)

                byte_index += 4 + user_metadata_length
            else:
                raise NotImplementedError("Metadata type %i not implemented." %
                                          metadata_type.value)

        metadata_bytes += metadata_content_bytes

        return metadata_bytes

    def load_all_metadata(self):

        if self._is_metadata_loaded:
            return

        self._data_buffer.seek(HEADER_SIZE)

        metadata_bytes = self._data_buffer.read(self._metadata_size)

        self._load_metadata_from_bytes(metadata_bytes)

        self._is_metadata_loaded = True

    def _load_metadata_from_bytes(self, metadata_bytes):

        self._metadata = {}

        metadata_type_list = []
        metadata_type_start_bytes = []
        metadata_lengths = []

        num_metadata_entries = struct.unpack("I", metadata_bytes[0:4])[0]

        byte_index = 4

        for metadata_entry_index in range(num_metadata_entries):

            metadata_type_id = \
                struct.unpack("I", metadata_bytes[byte_index:byte_index+4])[0]

            byte_index += 4

            metadata_type_list.append(Metadata_Type(metadata_type_id))
            metadata_type_start_bytes.append(
                struct.unpack("I", metadata_bytes[byte_index:byte_index+4])[0])

            byte_index += 4

        metadata_content_bytes = metadata_bytes[byte_index:]
        metadata_content_length = len(metadata_content_bytes)

        for metadata_entry_index in range(0, num_metadata_entries - 1):

            metadata_lengths.append(
                metadata_type_start_bytes[metadata_entry_index + 1] -
                metadata_type_start_bytes[metadata_entry_index])

        metadata_lengths.append(metadata_content_length -
                                metadata_type_start_bytes[-1])

        for metadata_index, metadata_type in enumerate(metadata_type_list):

            byte_index = metadata_type_start_bytes[metadata_index]
            metadata_length = metadata_lengths[metadata_index]

            if metadata_type == Metadata_Type.ROW_NAMES or \
                    metadata_type == Metadata_Type.COLUMN_NAMES:

                metadata = []

                while byte_index < metadata_length +\
                        metadata_type_start_bytes[metadata_index]:

                    name_length = struct.unpack(
                        "I", metadata_content_bytes[
                             byte_index:byte_index + 4])[0]

                    byte_index += 4

                    name = metadata_content_bytes[
                           byte_index:byte_index + name_length].decode("utf-8")

                    byte_index += name_length

                    metadata.append(name)

                self._metadata[metadata_type] = metadata
            elif metadata_type == Metadata_Type.USER_METADATA:

                end_byte = byte_index + metadata_length

                metadata = pickle.loads(
                    metadata_content_bytes[byte_index:end_byte])

                self._metadata[metadata_type] = metadata
            else:
                raise NotImplementedError("Metadata type %i not implemented." %
                                          metadata_type.value)

    def set_file_path(self, file_path):

        file_buffer = open(file_path, "rb")

        if not self._load_on_demand:
            self._data_buffer = io.BytesIO()
            self._data_buffer.write(file_buffer.read())

            self._data_buffer.seek(0)
        else:
            self._data_buffer = file_buffer

        version_string = self._data_buffer.read(8).decode("UTF-8")
        self._version = int(version_string[4:])

        data_type_id = struct.unpack("B", self._data_buffer.read(1))[0]

        self._data_type = Data_Type(data_type_id)
        self._data_size = struct.unpack("I", self._data_buffer.read(4))[0]
        self._num_rows = struct.unpack("Q", self._data_buffer.read(8))[0]
        self._num_columns = struct.unpack("Q", self._data_buffer.read(8))[0]
        self._num_entries = struct.unpack("Q", self._data_buffer.read(8))[0]
        self._metadata_size = struct.unpack("Q", self._data_buffer.read(8))[0]

        self._is_metadata_loaded = False
        self._metadata = None

        self._calculate_formats()

        self._load_indices()

        # Load the default value
        self._default_value = struct.unpack(
            self._pack_format_data, self._data_buffer.read(self._data_size))[0]

        self._row_data_start_byte = self._data_buffer.tell()
        self._column_data_start_byte = \
            self._row_data_start_byte + self._num_entries * \
            (self._num_bytes_column_index + self._data_size)

        self._unpack_data = struct.Struct(self._pack_format_data).unpack
        self._unpack_column_index = struct.Struct(
            self._pack_format_column_index).unpack
        self._unpack_row_index = struct.Struct(
            self._pack_format_row_index).unpack

        # We don't load the data until it is requested
        self._is_data_loaded = False
        self._row_column_indices = None
        self._row_data = None
        self._column_row_indices = None
        self._column_data = None

    def _load_indices(self):

        # We skip the metadata until it is requested
        self._data_buffer.seek(HEADER_SIZE + self._metadata_size)

        # Load the row and column indices

        row_start_indices_bytes = \
            self._data_buffer.read(self._num_bytes_row_byte * self._num_rows)

        self._row_start_indices = numpy.ndarray(
            (self._num_rows,),
            self._pack_format_row_byte,
            row_start_indices_bytes
        )

        self._row_start_indices = numpy.floor_divide(
            self._row_start_indices,
            self._num_bytes_column_index + self._data_size)

        row_start_indices_plus_one = \
            numpy.append(self._row_start_indices,
                         self._num_bytes_row_byte * self._num_rows)

        self._row_lengths = numpy.subtract(row_start_indices_plus_one[1:],
                                           self._row_start_indices)

        column_start_indices_bytes = \
            self._data_buffer.read(
                self._num_bytes_column_byte * self._num_columns)

        self._column_start_indices = numpy.ndarray(
            (self._num_columns,),
            self._pack_format_column_byte,
            column_start_indices_bytes
        )

        self._column_start_indices = numpy.floor_divide(
            self._column_start_indices,
            self._num_bytes_row_index + self._data_size)

        self._column_lengths = numpy.zeros(
            (self._num_columns,), dtype=self._pack_format_column_byte)

        for column_index, column_start_index in enumerate(
                self._column_start_indices):

            if column_index == self._num_columns - 1:
                column_end_index = self._num_entries
            else:
                column_end_index = self._column_start_indices[column_index + 1]

            num_column_entries = column_end_index - column_start_index

            self._column_lengths[column_index] = num_column_entries

    def load_all_data(self):

        if self._is_data_loaded:
            return

        self._data_buffer.seek(self._row_data_start_byte)

        self._row_column_indices = numpy.zeros(
            (self._num_entries,), dtype=numpy.uint32)
        self._row_data = numpy.zeros(
            (self._num_entries,), dtype=numpy.uint32)

        data_index = 0

        for row_index, row_start_index in enumerate(self._row_start_indices):

            for entry_index in range(self._row_lengths[row_index]):

                column_index = self._unpack_column_index(
                    self._data_buffer.read(self._num_bytes_column_index))[0]
                data_value = self._unpack_data(
                    self._data_buffer.read(self._data_size))[0]

                self._row_column_indices[data_index] = column_index
                self._row_data[data_index] = data_value
                data_index += 1

        self._column_row_indices = numpy.zeros(
            (self._num_entries,), dtype=numpy.uint32)
        self._column_data = numpy.zeros(
            (self._num_entries,), dtype=numpy.uint32)

        data_index = 0

        for column_index, column_start_index in enumerate(
                self._column_start_indices):

            for entry_index in range(self._column_lengths[column_index]):

                row_index = self._unpack_row_index(
                    self._data_buffer.read(self._num_bytes_row_index))[0]

                data_value = self._unpack_data(
                    self._data_buffer.read(self._data_size))[0]

                self._column_row_indices[data_index] = row_index
                self._column_data[data_index] = data_value

                data_index += 1

        self._is_data_loaded = True
