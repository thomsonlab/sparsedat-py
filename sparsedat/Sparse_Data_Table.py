import struct
import zlib
import numpy

from .Data_Type import Data_Type
from .Metadata_Type import Metadata_Type

import io


CURRENT_VERSION = 1
USE_COMPRESSION = False


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

    def __init__(self, file_path=None):

        self._file_path = file_path
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

    def __setitem__(self, index, value):
        raise NotImplementedError("Sparse_Data_Table is read-only for now")

    def __getitem__(self, index):

        row_index, column_index = index

        num_row_entries = self._row_lengths[row_index]
        num_column_entries = self._column_lengths[column_index]

        if num_row_entries < num_column_entries:
            row_start_index = self._row_start_indices[row_index]
            row_end_index = row_start_index + num_row_entries
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
            try:
                row_data_index = \
                    self._column_row_indices[
                        column_start_index:column_end_index]\
                    .index(row_index)
                return self._column_data[column_start_index + row_data_index]
            except ValueError:
                return self._default_value

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
        return self._metadata[Metadata_Type.ROW_NAMES]

    @property
    def column_names(self):
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

        metadata_bytes += metadata_content_bytes

        return metadata_bytes

    def load_metadata_from_bytes(self, metadata_bytes):

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

    def load(self, file_path=None):

        if file_path is None:
            file_path = self._file_path

        with open(file_path, "rb") as data_file:
            data_buffer = io.BytesIO()
            data_buffer.write(data_file.read())

        data_buffer.seek(0)

        version_string = data_buffer.read(8).decode("UTF-8")
        self._version = int(version_string[4:])

        data_type_id = struct.unpack("B", data_buffer.read(1))[0]

        self._data_type = Data_Type(data_type_id)

        self._data_size = struct.unpack("I", data_buffer.read(4))[0]

        self._num_rows = struct.unpack("Q", data_buffer.read(8))[0]

        self._num_columns = struct.unpack("Q", data_buffer.read(8))[0]

        self._num_entries = struct.unpack("Q", data_buffer.read(8))[0]

        metadata_size = struct.unpack("Q", data_buffer.read(8))[0]

        metadata_bytes = data_buffer.read(metadata_size)

        self.load_metadata_from_bytes(metadata_bytes)

        self._calculate_formats()

        self._row_start_indices = numpy.zeros(
            (self._num_rows,), dtype=numpy.uint32)

        # Load the row and column indices

        for row_index in range(self._num_rows):

            row_start_byte = struct.unpack(
                self._pack_format_row_byte,
                data_buffer.read(self._num_bytes_row_byte))[0]

            self._row_start_indices[row_index] = \
                int(row_start_byte /
                    (self._num_bytes_column_index + self._data_size))

        self._column_start_indices = numpy.zeros(
            (self._num_columns,), dtype=numpy.uint32)

        for column_index in range(self._num_columns):

            column_start_byte = struct.unpack(
                self._pack_format_column_byte,
                data_buffer.read(self._num_bytes_column_byte))[0]

            self._column_start_indices[column_index] = \
                int(column_start_byte /
                    (self._num_bytes_row_index + self._data_size))

        # Load the data
        self._default_value = struct.unpack(
            self._pack_format_data, data_buffer.read(self._data_size))[0]

        self._row_lengths = numpy.zeros(
            (self._num_rows,), dtype=numpy.uint32)
        self._row_column_indices = numpy.zeros(
            (self._num_entries,), dtype=numpy.uint32)
        self._row_data = numpy.zeros(
            (self._num_entries,), dtype=numpy.uint32)

        data_index = 0

        unpack_data = struct.Struct(self._pack_format_data).unpack
        unpack_column_index = struct.Struct(self._pack_format_column_index).unpack
        unpack_row_index = struct.Struct(self._pack_format_row_index).unpack

        for row_index, row_start_index in enumerate(self._row_start_indices):

            if row_index == self._num_rows - 1:
                row_end_index = self._num_entries
            else:
                row_end_index = self._row_start_indices[row_index + 1]

            num_row_entries = row_end_index - row_start_index

            self._row_lengths[row_index] = num_row_entries

            for entry_index in range(num_row_entries):

                column_index = unpack_column_index(
                    data_buffer.read(self._num_bytes_column_index))[0]
                data_value = unpack_data(data_buffer.read(self._data_size))[0]

                self._row_column_indices[data_index] = column_index
                self._row_data[data_index] = data_value
                data_index += 1

        self._column_lengths = numpy.zeros(
            (self._num_columns,), dtype=numpy.uint32)
        self._column_row_indices = numpy.zeros(
            (self._num_entries,), dtype=numpy.uint32)
        self._column_data = numpy.zeros(
            (self._num_entries,), dtype=numpy.uint32)

        data_index = 0

        for column_index, column_start_index in enumerate(
                self._column_start_indices):

            if column_index == self._num_columns - 1:
                column_end_index = self._num_entries
            else:
                column_end_index = self._column_start_indices[column_index + 1]

            num_column_entries = column_end_index - column_start_index

            self._column_lengths[column_index] = num_column_entries

            for entry_index in range(num_column_entries):

                row_index = unpack_row_index(
                    data_buffer.read(self._num_bytes_row_index))[0]

                data_value = unpack_data(data_buffer.read(self._data_size))[0]

                self._column_row_indices[data_index] = row_index
                self._column_data[data_index] = data_value

                data_index += 1
