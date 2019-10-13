import struct
import zlib
import pickle
import numpy
import io
from scipy import sparse
from enum import Enum
import pandas

from .Data_Type import Data_Type
from .Metadata_Type import Metadata_Type


CURRENT_VERSION = 1
HEADER_SIZE = 45


class Output_Mode(Enum):

    SDT = 0
    NUMPY = 1
    PANDAS = 2


def if_none(a, b):
    return b if a is None else a


class Sparse_Data_Table:

    OUTPUT_MODE = Output_Mode.SDT

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
            load_on_demand=False):

        # If this SDT has a location on disk, this is it
        self._file_path = None

        # If this SDT is stored on a buffer
        self._data_buffer = None

        # The version of the file. Defaults to current version
        self._version = CURRENT_VERSION

        # Any metadata associated with the file
        self._metadata = {}

        # The data needed to traverse and query the SDT
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

        self._row_name_index_map = None
        self._column_name_index_map = None

        # Information about how to store data when writing it out
        self._data_type = None
        self._data_size = None
        self._num_bytes_row_index = None
        self._num_bytes_column_index = None
        self._num_bytes_row_byte = None
        self._num_bytes_column_byte = None
        self._num_bytes_row_entry = None
        self._num_bytes_column_entry = None
        self._max_row_byte = None
        self._max_column_byte = None
        self._pack_format_row_byte = None
        self._pack_format_column_byte = None
        self._pack_format_row_index = None
        self._pack_format_column_index = None
        self._pack_format_data = None

        # Data relevant for real time file I/O
        self._is_data_on_buffer = False
        self._is_metadata_on_buffer = False

        self._metadata_size = None
        self._row_data_start_byte = None
        self._column_data_start_byte = None

        self._load_on_demand = load_on_demand

        if file_path is not None:
            self.set_file_path(file_path)

            if file_path.endswith(".gz"):
                self.load_all_data()

        if not self._load_on_demand:
            self.load_all_metadata()
            self.load_all_data()

    def __del__(self):

        if self._data_buffer is not None:
            self._data_buffer.close()

    def __setitem__(self, index, value):
        raise NotImplementedError("Sparse_Data_Table is read-only for now")

    def get_indices_from_slices(self, index):

        if isinstance(index, tuple):

            if len(index) != 2:
                raise ValueError("Only support 2D indexing")

            row_index = index[0]
            column_index = index[1]

        else:
            row_index = index
            column_index = list(range(self._num_columns))

        if isinstance(row_index, slice):
            row_indices = list(range(
                if_none(row_index.start, 0),
                if_none(row_index.stop, self._num_rows),
                if_none(row_index.step, 1)
            ))
        elif isinstance(row_index, str) or not hasattr(row_index, "__len__"):
            row_indices = [row_index]
        else:
            row_indices = row_index

        if isinstance(column_index, slice):

            column_indices = list(range(
                if_none(column_index.start, 0),
                if_none(column_index.stop, self._num_columns),
                if_none(column_index.step, 1)
            ))
        elif isinstance(column_index, str) or \
                not hasattr(column_index, "__len__"):
            column_indices = [column_index]
        else:
            column_indices = column_index

        if isinstance(row_indices[0], str):

            row_indices = [
                self._row_name_index_map[i] for i in row_indices
            ]
        elif isinstance(row_indices[0], bool) or \
                isinstance(row_indices[0], numpy.bool_):
            row_indices = [i for i, x in enumerate(row_indices) if x]

        if isinstance(column_indices[0], str):

            column_indices = [
                self._column_name_index_map[i] for i in column_indices
            ]
        elif isinstance(column_indices[0], bool) or \
                isinstance(column_indices[0], numpy.bool_):
            column_indices = [i for i, x in enumerate(column_indices) if x]

        return row_indices, column_indices

    def __repr__(self):

        if Sparse_Data_Table.OUTPUT_MODE == Output_Mode.PANDAS:
            return self.to_pandas().__repr__()

        return self.to_array().__repr__()

    def __str__(self):

        if Sparse_Data_Table.OUTPUT_MODE == Output_Mode.PANDAS:
            return self.to_pandas().__str__()

        return self.to_array().__str__()

    def _repr_html_(self):
        return self.to_pandas()._repr_html_()

    def get_sub_SDT(self, row_indices, column_indices):

        num_rows = len(row_indices)
        num_columns = len(column_indices)

        new_SDT = Sparse_Data_Table()
        new_SDT._default_value = self._default_value
        new_SDT._num_rows = num_rows
        new_SDT._num_columns = num_columns

        num_row_entries = sum([self._row_lengths[i] for i in row_indices])
        num_column_entries = sum([self._column_lengths[i]
                                  for i in column_indices])

        if num_row_entries < num_column_entries:

            # We're going to create a new SDT with only the requested rows,
            # but without filtering any columns (to avoid searching for the
            # existence of each column in each row)

            row_data = numpy.ndarray(
                (num_row_entries,),
                dtype=self._pack_format_data
            )
            row_column_indices = numpy.ndarray(
                (num_row_entries,),
                dtype=self._pack_format_column_index
            )
            row_start_indices = numpy.ndarray(
                (num_rows,),
                dtype=self._pack_format_row_byte
            )

            entry_index = 0

            for new_row_index, row_index in enumerate(row_indices):

                row_length = self._row_lengths[row_index]
                row_start_index = self._row_start_indices[row_index]

                row_start_indices[new_row_index] = entry_index

                if row_length == 0:
                    continue

                row_data[entry_index:entry_index+row_length] = \
                    self._row_data[row_start_index:row_start_index+row_length]

                row_column_indices[entry_index:entry_index+row_length] = \
                    self._row_column_indices[
                        row_start_index:row_start_index+row_length]

                entry_index += row_length

            row_start_indices = numpy.append(row_start_indices, num_row_entries)

            new_SDT.from_sparse_row_entries(
                (row_data, row_column_indices, row_start_indices),
                num_rows,
                self._num_columns,
                self._default_value
            )

            if Metadata_Type.ROW_NAMES in self._metadata:
                new_SDT._metadata[Metadata_Type.ROW_NAMES] = []
                new_SDT._row_name_index_map = {}

                for new_row_index, row_index in enumerate(row_indices):
                    row_name = self._metadata[Metadata_Type.ROW_NAMES][
                        row_index]
                    new_SDT._metadata[Metadata_Type.ROW_NAMES].append(row_name)
                    new_SDT._row_name_index_map[row_name] = new_row_index

            if Metadata_Type.COLUMN_NAMES in self._metadata:
                new_SDT._metadata[Metadata_Type.COLUMN_NAMES] = \
                    self._metadata[Metadata_Type.COLUMN_NAMES].copy()
                new_SDT._column_name_index_map = \
                    self._column_name_index_map.copy()

            # Now we have an SDT with all the requested rows; if we need to
            # filter out columns, we repeat the process
            if num_columns < self._num_columns:
                return new_SDT.get_sub_SDT(
                    list(range(num_rows)), column_indices
                )

        else:

            column_data = numpy.ndarray(
                (num_column_entries,),
                dtype=self._pack_format_data
            )
            column_row_indices = numpy.ndarray(
                (num_column_entries,),
                dtype=self._pack_format_row_index
            )
            column_start_indices = numpy.ndarray(
                (num_columns,),
                dtype=self._pack_format_column_byte
            )

            entry_index = 0

            for new_column_index, column_index in enumerate(column_indices):

                column_length = self._column_lengths[column_index]
                column_start_index = self._column_start_indices[column_index]

                column_start_indices[new_column_index] = entry_index

                if column_length == 0:
                    continue

                column_data[entry_index:entry_index+column_length] = \
                    self._column_data[
                        column_start_index:column_start_index+column_length]

                column_row_indices[entry_index:entry_index+column_length] = \
                    self._column_row_indices[
                        column_start_index:column_start_index+column_length]

                entry_index += column_length

            column_start_indices = numpy.append(
                column_start_indices, num_column_entries)

            new_SDT.from_sparse_column_entries(
                (column_data, column_row_indices, column_start_indices),
                self._num_rows,
                num_columns,
                self._default_value
            )

            if Metadata_Type.ROW_NAMES in self._metadata:
                new_SDT._metadata[Metadata_Type.ROW_NAMES] = \
                    self._metadata[Metadata_Type.ROW_NAMES].copy()
                new_SDT._row_name_index_map = \
                    self._row_name_index_map.copy()

            if Metadata_Type.COLUMN_NAMES in self._metadata:
                new_SDT._metadata[Metadata_Type.COLUMN_NAMES] = []
                new_SDT._column_name_index_map = {}

                for new_column_index, column_index in enumerate(column_indices):
                    column_name = self._metadata[Metadata_Type.COLUMN_NAMES][
                        column_index]
                    new_SDT._metadata[Metadata_Type.COLUMN_NAMES].append(
                        column_name
                    )
                    new_SDT._column_name_index_map[
                        column_name] = new_column_index

            # Now we have an SDT with all the requested columns; if we need to
            # filter out rows, we repeat the process
            if num_rows < self._num_rows:
                return new_SDT.get_sub_SDT(
                    row_indices,
                    list(range(num_columns))
                )

        return new_SDT

    def __getitem__(self, index):

        # Special case for all item getter
        if len(index) == 2:

            row_index = index[0]
            column_index = index[1]

            if isinstance(row_index, slice) and isinstance(column_index, slice):
                if row_index.start is None and row_index.stop is None and \
                        column_index.start is None and column_index.stop is None:
                    if Sparse_Data_Table.OUTPUT_MODE == Output_Mode.SDT or \
                            Sparse_Data_Table.OUTPUT_MODE == Output_Mode.NUMPY:
                        return self.to_array()
                    elif Sparse_Data_Table.OUTPUT_MODE == Output_Mode.PANDAS:
                        return self.to_pandas()
                    else:
                        raise NotImplementedError("Invalid output mode")

        row_indices, column_indices = self.get_indices_from_slices(index)

        # Special case for single item getter
        if len(row_indices) == 1 and len(column_indices) == 1:
            row_index = row_indices[0]
            column_index = column_indices[0]
            row_length = self._row_lengths[row_index]
            row_start_index = self._row_start_indices[row_index]
            row_column_indices = self._row_column_indices[
                row_start_index:row_start_index+row_length
            ]

            if len(row_column_indices) == 0:
                return self._default_value

            relative_column_index = numpy.searchsorted(
                row_column_indices, column_index)

            if relative_column_index >= len(row_column_indices):
                return self._default_value

            if row_column_indices[relative_column_index] != column_index:
                return self._default_value

            return self._row_data[row_start_index+relative_column_index]

        if Sparse_Data_Table.OUTPUT_MODE == Output_Mode.SDT:
            return self.get_sub_SDT(row_indices, column_indices)

        num_rows = len(row_indices)
        num_columns = len(column_indices)

        num_row_entries = sum([self._row_lengths[i] for i in row_indices])
        num_column_entries = sum([self._column_lengths[i]
                                  for i in column_indices])

        if self._default_value == 0:
            sliced_array = numpy.zeros(
                (num_rows, num_columns),
                dtype=self._pack_format_data
            )
        else:
            sliced_array = numpy.full(
                (num_rows, num_columns),
                fill_value=self._default_value,
                dtype=self._pack_format_data)

        # If we have less rows, then we should slice along rows first
        if num_row_entries <= num_column_entries:

            target_column_indices = column_indices

            # Loop through each row we want to grab

            for relative_row_index, target_row_index in enumerate(row_indices):

                num_row_entries = self._row_lengths[target_row_index]

                # If this is an empty row, skip it
                if num_row_entries == 0:
                    continue

                # This is where this row's data entries start
                row_entry_start_index = \
                    self._row_start_indices[target_row_index]

                # And this is where it ends
                row_entry_end_index = row_entry_start_index + \
                    num_row_entries

                # If the data is on a buffer, we look at the buffer
                if self._is_data_on_buffer:

                    byte_index = self._row_data_start_byte + \
                        row_entry_start_index * self._num_bytes_row_entry

                    self._data_buffer.seek(byte_index)

                    column_indices_read_length = self._num_bytes_row_entry *\
                        num_row_entries

                    column_indices = numpy.ndarray(
                        (num_row_entries,),
                        self._pack_format_column_index,
                        self._data_buffer.read(column_indices_read_length),
                        strides=self._num_bytes_row_entry
                    )

                    self._data_buffer.seek(byte_index +
                                           self._num_bytes_column_index)

                    read_length = self._num_bytes_row_entry * \
                        num_row_entries - self._num_bytes_column_index

                    row_data = numpy.ndarray(
                        (num_row_entries,),
                        self._pack_format_data,
                        self._data_buffer.read(read_length),
                        strides=self._num_bytes_row_entry
                    )

                # If the data is not on a buffer, we can just grab it directly
                # from our stored arrays
                else:
                    column_indices = self._row_column_indices[
                        row_entry_start_index:row_entry_end_index]
                    row_data = self._row_data[
                        row_entry_start_index:row_entry_end_index]

                first_found_index = numpy.searchsorted(
                    column_indices, target_column_indices[0], side="left")

                last_found_index = numpy.searchsorted(
                    column_indices, target_column_indices[-1], side="right")

                # This means the right-most column of this row is already
                # before the bounds of the desired slice; we can stop
                if first_found_index >= num_row_entries:
                    continue

                if last_found_index == 0:
                    if column_indices[0] != target_column_indices[-1]:
                        continue

                # This means the left-most column of this row is at or beyond
                # the left bound of the slice. If so, this is where we start
                if first_found_index == 0:
                    # If the value is the same, it means the left-most column
                    # of this row is exactly the start of the slice bound
                    if column_indices[0] < target_column_indices[0]:
                        first_found_index = 1

                if last_found_index == num_row_entries:
                    if column_indices[last_found_index - 1] > \
                            target_column_indices[-1]:
                        last_found_index = num_row_entries - 1

                if first_found_index >= last_found_index:
                    continue

                found_indices = numpy.arange(
                    first_found_index, last_found_index)

                relative_column_indices = \
                    column_indices[found_indices] - column_indices[0]
                sliced_array[relative_row_index, relative_column_indices] = \
                    row_data[found_indices]
        else:

            target_row_indices = row_indices

            # Loop through each column we want to grab

            for relative_column_index, target_column_index in \
                    enumerate(column_indices):

                num_column_entries = self._column_lengths[target_column_index]

                # If this is an empty column, skip it
                if num_column_entries == 0:
                    continue

                # This is where this column's data entries start
                column_entry_start_index = \
                    self._column_start_indices[target_column_index]

                # And this is where it ends
                column_entry_end_index = column_entry_start_index + \
                    num_column_entries

                # If the data is on a buffer, we look at the buffer
                if self._is_data_on_buffer:

                    byte_index = self._column_data_start_byte + \
                        column_entry_start_index * self._num_bytes_column_entry

                    self._data_buffer.seek(byte_index)

                    row_indices_read_length = self._num_bytes_column_entry *\
                        num_column_entries

                    row_indices = numpy.ndarray(
                        (num_column_entries,),
                        self._pack_format_row_index,
                        self._data_buffer.read(row_indices_read_length),
                        strides=self._num_bytes_column_entry
                    )

                    self._data_buffer.seek(byte_index +
                                           self._num_bytes_row_index)

                    read_length = self._num_bytes_column_entry * \
                        num_column_entries - self._num_bytes_row_index

                    column_data = numpy.ndarray(
                        (num_column_entries,),
                        self._pack_format_data,
                        self._data_buffer.read(read_length),
                        strides=self._num_bytes_column_entry
                    )
                # If the data is not on a buffer, we can just grab it directly
                # from our stored arrays
                else:
                    row_indices = self._column_row_indices[
                        column_entry_start_index:column_entry_end_index]
                    column_data = self._column_data[
                        column_entry_start_index:column_entry_end_index]

                first_found_index = numpy.searchsorted(
                    row_indices, target_row_indices[0], side="left")

                last_found_index = numpy.searchsorted(
                    row_indices, target_row_indices[-1], side="right")

                # This means the right-most row of this column is already
                # before the bounds of the desired slice; we can stop
                if first_found_index >= num_column_entries:
                    continue

                if last_found_index == 0:
                    if row_indices[0] != target_row_indices[-1]:
                        continue

                # This means the left-most row of this column is at or beyond
                # the left bound of the slice. If so, this is where we start
                if first_found_index == 0:
                    # If the value is the same, it means the left-most row
                    # of this column is exactly the start of the slice bound
                    if row_indices[0] < target_row_indices[0]:
                        first_found_index = 1

                if last_found_index == num_column_entries:
                    if row_indices[last_found_index - 1] > \
                            target_row_indices[-1]:
                        last_found_index = num_column_entries - 1

                if first_found_index >= last_found_index:
                    continue

                found_indices = numpy.arange(
                    first_found_index, last_found_index)

                relative_row_indices = \
                    row_indices[found_indices] - row_indices[0]

                sliced_array[relative_row_indices, relative_column_index] = \
                    column_data[found_indices]

        if sliced_array.shape == (1, 1):
            return sliced_array[0][0]

        elif Sparse_Data_Table.OUTPUT_MODE == Output_Mode.PANDAS:
            return pandas.DataFrame(
                sliced_array,
                index=[self.row_names[i] for i in target_row_indices],
                columns=[self.column_names[i] for i in target_column_indices]
            )
        elif Sparse_Data_Table.OUTPUT_MODE == Output_Mode.NUMPY:
            return sliced_array
        else:
            raise ValueError("Invalid Output Mode")

    def get_row_index(self, index):

        if isinstance(index, int):
            return index

        if index not in self._row_name_index_map:
            raise ValueError("Row %s does not exist" % index)

        return self._row_name_index_map[index]

    def get_column_index(self, index):

        if isinstance(index, int):
            return index

        if index not in self._column_name_index_map:
            raise ValueError("Column %s does not exist" % index)

        return self._column_name_index_map[index]

    def add_row(self, index_values, row_name=None):
        """
        Add a row to this sparse data table with the given index values.
        :param index_values: An iterable of tuples of indices and the value at
            that index. Indices can be either numerical or column names. All
            unspecified values are assumed to be self._default_value
        :param row_name: The name of the row
        :return: None
        """

        if self._is_data_on_buffer:
            raise NotImplementedError("Adding rows to on-disk sdt is not "
                                      "supported")

        if row_name is not None:
            self.load_all_metadata()

            if Metadata_Type.ROW_NAMES not in self._metadata:
                raise ValueError("Can't add named row to unnamed SDT")
        elif row_name is None:

            if Metadata_Type.ROW_NAMES in self._metadata:
                raise ValueError("Must name new row in named SDT")

        row_data_to_add = []
        row_column_indices_to_add = []

        for index, value in index_values:

            try:
                column_index = self.get_column_index(index)
            except ValueError:
                continue

            if value == self._default_value:
                continue

            row_data_to_add.append(value)
            row_column_indices_to_add.append(column_index)

        sorted_column_indices = numpy.argsort(row_column_indices_to_add)
        row_column_indices_to_add = numpy.array(row_column_indices_to_add)[
            sorted_column_indices]
        row_data_to_add = numpy.array(row_data_to_add)[sorted_column_indices]

        self._row_start_indices = numpy.append(
            self._row_start_indices, len(self._row_data))
        self._row_lengths = numpy.append(
            self._row_lengths, len(row_data_to_add))

        self._row_data = numpy.resize(
            self._row_data,
            (
                self._row_data.shape[0] + len(row_data_to_add),
            )
        )

        self._row_column_indices = numpy.resize(
            self._row_column_indices,
            (
                self._row_column_indices.shape[0] +
                len(row_column_indices_to_add),
            )
        )

        self._row_data[-len(row_data_to_add):] = row_data_to_add
        self._row_column_indices[-len(row_column_indices_to_add):] = \
            row_column_indices_to_add

        self._num_rows += 1

        new_column_data = numpy.ndarray(
            self._row_data.shape,
            dtype=self._column_data.dtype
        )

        new_column_row_indices = numpy.ndarray(
            self._row_column_indices.shape,
            dtype=self._column_row_indices.dtype
        )

        previous_entry_index = 0
        num_entries_added = 0
        previous_column_index = -1

        new_row_index = self._num_rows - 1

        # Loop through each column that had a value added
        for value, column_index in \
                zip(row_data_to_add, row_column_indices_to_add):
            # The new entry is added as the last entry to the column
            entry_index = previous_entry_index + \
                          self._column_lengths[previous_column_index + 1:
                                               column_index + 1].sum()

            # It is offset in the new data table by the number of entries that
            # have been added
            new_entry_index = entry_index + num_entries_added

            previous_new_entry_index = previous_entry_index + num_entries_added

            # Copy over all the data from the last column through the current
            # column
            new_column_data[previous_new_entry_index:new_entry_index] = \
                self._column_data[previous_entry_index:entry_index]

            new_column_row_indices[previous_new_entry_index:new_entry_index] \
                = self._column_row_indices[previous_entry_index:entry_index]

            new_column_data[new_entry_index] = value
            new_column_row_indices[new_entry_index] = new_row_index

            self._column_lengths[column_index] += 1

            if column_index + 1 < self._num_columns:
                self._column_start_indices[column_index + 1:] += 1

            previous_column_index = column_index
            previous_entry_index = entry_index
            num_entries_added += 1

        self._column_data = new_column_data
        self._column_row_indices = new_column_row_indices

        if row_name is not None:
            self._metadata[Metadata_Type.ROW_NAMES].append(row_name)
            self._row_name_index_map[row_name] = new_row_index

    def add_column(self, index_values, column_name=None):
        """
        Add a column to this sparse data table with the given index values.
        :param index_values: An iterable of tuples of indices and the value at
            that index. Indices can be either numerical or column names. All
            unspecified values are assumed to be self._default_value
        :param column_name: The name of the column
        :return: None
        """

        if self._is_data_on_buffer:
            raise NotImplementedError("Adding columns to on-disk sdt is not "
                                      "supported")

        if column_name is not None:
            self.load_all_metadata()

            if Metadata_Type.COLUMN_NAMES not in self._metadata:
                raise ValueError("Can't add named column to unnamed SDT")
        elif column_name is None:

            if Metadata_Type.ROW_NAMES in self._metadata:
                raise ValueError("Must name new column in named SDT")

        column_data_to_add = []
        column_row_indices_to_add = []

        for index, value in index_values:

            try:
                row_index = self.get_row_index(index)
            except ValueError:
                continue

            if value == self._default_value:
                continue

            column_data_to_add.append(value)
            column_row_indices_to_add.append(row_index)

        sorted_row_indices = numpy.argsort(column_row_indices_to_add)
        column_row_indices_to_add = numpy.array(column_row_indices_to_add)[
            sorted_row_indices]
        column_data_to_add = numpy.array(column_data_to_add)[sorted_row_indices]

        self._column_start_indices = numpy.append(
            self._column_start_indices, len(self._column_data))
        self._column_lengths = numpy.append(
            self._column_lengths, len(column_data_to_add))

        self._column_data = numpy.resize(
            self._column_data,
            (
                self._column_data.shape[0] + len(column_data_to_add),
            )
        )

        self._column_row_indices = numpy.resize(
            self._column_row_indices,
            (
                self._column_row_indices.shape[0] +
                len(column_row_indices_to_add),
            )
        )

        self._column_data[-len(column_data_to_add):] = column_data_to_add
        self._column_row_indices[-len(column_row_indices_to_add):] = \
            column_row_indices_to_add

        self._num_columns += 1

        new_row_data = numpy.ndarray(
            self._column_data.shape,
            dtype=self._row_data.dtype
        )

        new_row_column_indices = numpy.ndarray(
            self._column_row_indices.shape,
            dtype=self._row_column_indices.dtype
        )

        previous_entry_index = 0
        num_entries_added = 0
        previous_row_index = -1

        new_column_index = self._num_columns - 1

        # Loop through each row that had a value added
        for value, row_index in \
                zip(column_data_to_add, column_row_indices_to_add):
            # The new entry is added as the last entry to the row
            entry_index = previous_entry_index + \
                          self._row_lengths[previous_row_index + 1:
                                            row_index + 1].sum()

            # It is offset in the new data table by the number of entries that
            # have been added
            new_entry_index = entry_index + num_entries_added

            previous_new_entry_index = previous_entry_index + num_entries_added

            # Copy over all the data from the last row through the current
            # row
            new_row_data[previous_new_entry_index:new_entry_index] = \
                self._row_data[previous_entry_index:entry_index]

            new_row_column_indices[previous_new_entry_index:new_entry_index] \
                = self._row_column_indices[previous_entry_index:entry_index]

            new_row_data[new_entry_index] = value
            new_row_column_indices[new_entry_index] = new_column_index

            self._row_lengths[row_index] += 1

            if row_index + 1 < self._num_rows:
                self._row_start_indices[row_index + 1:] += 1

            previous_row_index = row_index
            previous_entry_index = entry_index
            num_entries_added += 1

        self._row_data = new_row_data
        self._row_column_indices = new_row_column_indices

        if column_name is not None:
            self._metadata[Metadata_Type.COLUMN_NAMES].append(column_name)
            self._column_name_index_map[column_name] = new_column_index

    def to_array(self):

        if self._is_data_on_buffer:
            self.load_all_data()

        full_array = numpy.full(
            (self._num_rows, self._num_columns),
            fill_value=self._default_value,
            dtype=self._pack_format_data)

        if self._num_rows >= self._num_columns:
            for row_index in range(self._num_rows):

                # This is where this row's data entries start
                row_entry_start_index = self._row_start_indices[row_index]

                # And this is where it ends
                row_entry_end_index = row_entry_start_index + \
                    self._row_lengths[row_index]

                full_array[
                    row_index,
                    self._row_column_indices[
                        row_entry_start_index:row_entry_end_index]] = \
                    self._row_data[row_entry_start_index:row_entry_end_index]
        else:
            for column_index in range(self._num_columns):

                # This is where this row's data entries start
                column_entry_start_index = \
                    self._column_start_indices[column_index]

                # And this is where it ends
                column_entry_end_index = column_entry_start_index + \
                    self._column_lengths[column_index]

                full_array[self._column_row_indices[
                           column_entry_start_index:column_entry_end_index],
                           column_index] = \
                    self._column_data[
                        column_entry_start_index:column_entry_end_index]
        return full_array

    def to_pandas(self):

        return pandas.DataFrame(
            self.to_array(),
            index=self.row_names,
            columns=self.column_names
        )

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

            # In case we're at the end, we fill up the rest with a pointer
            # to the last element
            if entry_index >= self._num_entries:
                entry_index = self._num_entries - 1

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

            # In case we're at the end, we fill up the rest with a pointer
            # to the last element
            if entry_index >= self._num_entries:
                entry_index = self._num_entries - 1

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

    def from_sparse_column_entries(
            self,
            data_rows_columns,
            num_rows,
            num_columns,
            default_value=0):

        self._num_rows = num_rows
        self._num_columns = num_columns
        self._num_entries = len(data_rows_columns[0])
        self._default_value = default_value

        if len(data_rows_columns[0]) == 0:
            self._data_type = Data_Type.FLOAT
            self._data_size = 8
        else:
            min_value = data_rows_columns[0].min()
            max_value = data_rows_columns[0].max()
            first_value = data_rows_columns[0][0]

            if isinstance(first_value, int) or issubclass(
                    data_rows_columns[0].dtype.type, numpy.integer):
                if min_value >= 0:
                    self._data_type = Data_Type.UINT
                else:
                    self._data_type = Data_Type.INT
            else:
                self._data_type = Data_Type.FLOAT

            self._data_size = Sparse_Data_Table.get_num_bytes(
                self._data_type, max_value, min_value=min_value
            )

        self._calculate_formats()
        self._column_start_indices = numpy.array(data_rows_columns[2])

        column_start_indices_plus_one = \
            numpy.append(self._column_start_indices, self._num_entries)

        column_start_indices_plus_one = column_start_indices_plus_one[1:]
        self._column_lengths = numpy.subtract(
            column_start_indices_plus_one,
            self._column_start_indices
        )

        self._column_data = numpy.array(data_rows_columns[0])
        self._column_row_indices = numpy.array(data_rows_columns[1])

        scipy_sparse_csc = sparse.csc_matrix(data_rows_columns,
                                             (num_rows, num_columns))
        scipy_sparse_csr = scipy_sparse_csc.tocsr(True)

        self._row_start_indices = numpy.array(scipy_sparse_csr.indptr)

        row_start_indices_plus_one = \
            numpy.append(self._row_start_indices, self._num_entries)

        row_start_indices_plus_one = row_start_indices_plus_one[1:]
        self._row_lengths = numpy.subtract(
            row_start_indices_plus_one,
            self._row_start_indices
        )

        self._row_data = numpy.array(scipy_sparse_csr.data)
        self._row_column_indices = numpy.array(scipy_sparse_csr.indices)

    def from_sparse_row_entries(
            self,
            data_columns_rows,
            num_rows,
            num_columns,
            default_value=0):

        self._num_rows = num_rows
        self._num_columns = num_columns
        self._num_entries = len(data_columns_rows[0])
        self._default_value = default_value

        if len(data_columns_rows[0]) == 0:
            self._data_type = Data_Type.FLOAT
            self._data_size = 8
        else:
            min_value = data_columns_rows[0].min()
            max_value = data_columns_rows[0].max()
            first_value = data_columns_rows[0][0]

            if isinstance(first_value, int) or issubclass(
                    data_columns_rows[0].dtype.type, numpy.integer):
                if min_value >= 0:
                    self._data_type = Data_Type.UINT
                else:
                    self._data_type = Data_Type.INT
            else:
                self._data_type = Data_Type.FLOAT

            self._data_size = Sparse_Data_Table.get_num_bytes(
                self._data_type, max_value, min_value=min_value
            )

        self._calculate_formats()
        self._row_start_indices = numpy.array(data_columns_rows[2])

        row_start_indices_plus_one = \
            numpy.append(self._row_start_indices, self._num_entries)

        row_start_indices_plus_one = row_start_indices_plus_one[1:]
        self._row_lengths = numpy.subtract(
            row_start_indices_plus_one,
            self._row_start_indices
        )

        self._row_data = numpy.array(data_columns_rows[0])
        self._row_column_indices = numpy.array(data_columns_rows[1])

        scipy_sparse_csr = sparse.csr_matrix(data_columns_rows,
                                             (num_rows, num_columns))
        scipy_sparse_csc = scipy_sparse_csr.tocsc(True)

        self._column_start_indices = numpy.array(scipy_sparse_csc.indptr)

        column_start_indices_plus_one = \
            numpy.append(self._column_start_indices, self._num_entries)

        column_start_indices_plus_one = column_start_indices_plus_one[1:]
        self._column_lengths = numpy.subtract(
            column_start_indices_plus_one,
            self._column_start_indices
        )

        self._column_data = numpy.array(scipy_sparse_csc.data)
        self._column_row_indices = numpy.array(scipy_sparse_csc.indices)

    @property
    def row_names(self):

        if self._is_metadata_on_buffer:
            self.load_all_metadata()

        if Metadata_Type.ROW_NAMES not in self._metadata:
            return [str[i] for i in range(self._num_rows)]

        return self._metadata[Metadata_Type.ROW_NAMES]

    @property
    def column_names(self):

        if self._is_metadata_on_buffer:
            self.load_all_metadata()

        if Metadata_Type.COLUMN_NAMES not in self._metadata:
            return [str[i] for i in range(self._num_columns)]


        return self._metadata[Metadata_Type.COLUMN_NAMES]

    @row_names.setter
    def row_names(self, new_row_names):

        if self._num_rows is not None and len(new_row_names) != self._num_rows:
            raise ValueError("Row names must match number of rows!")

        self._metadata[Metadata_Type.ROW_NAMES] = new_row_names

        self._row_name_index_map = {
            row_name: index for index, row_name in
            enumerate(self._metadata[Metadata_Type.ROW_NAMES])
        }

    @column_names.setter
    def column_names(self, new_column_names):

        if self._num_columns is not None and \
                len(new_column_names) != self._num_columns:
            raise ValueError("Column names must match number of columns!")

        self._metadata[Metadata_Type.COLUMN_NAMES] = new_column_names

        self._column_name_index_map = {
            column_name: index for index, column_name in
            enumerate(self._metadata[Metadata_Type.COLUMN_NAMES])
        }

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

        if self._is_metadata_on_buffer:
            self.load_all_metadata()

        return self._metadata[Metadata_Type.USER_METADATA]

    @metadata.setter
    def metadata(self, user_metadata):

        if self._is_metadata_on_buffer:
            self.load_all_metadata()

        self._metadata[Metadata_Type.USER_METADATA] = user_metadata

    @property
    def row_data(self):

        if self._is_data_on_buffer:
            self.load_all_data()

        return self._row_data

    @property
    def row_column_indices(self):

        if self._is_data_on_buffer:
            self.load_all_data()

        return self._row_column_indices

    @property
    def row_start_indices(self):
        return self._row_start_indices

    def _calculate_formats(self):

        self._num_bytes_row_index = self.get_num_bytes(
            Data_Type.UINT, self._num_rows)
        self._num_bytes_column_index = self.get_num_bytes(
            Data_Type.UINT, self._num_columns)

        self._num_bytes_row_entry = \
            self._num_bytes_column_index + self._data_size
        self._num_bytes_column_entry = \
            self._num_bytes_row_index + self._data_size

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

        if self._is_data_on_buffer:
            self.load_all_data()
            if self._is_metadata_on_buffer:
                self.load_all_metadata()

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
                data_buffer.write(
                    struct.pack(self._pack_format_data, data_value))

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
                data_buffer.write(
                    struct.pack(self._pack_format_data, data_value))

        data_buffer.seek(0)
        with open(file_path, "wb") as data_file:
            if file_path.endswith(".gz"):
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

        if not self._is_metadata_on_buffer:
            return

        if not self._data_buffer:
            self._is_metadata_on_buffer = False
            return

        self._data_buffer.seek(HEADER_SIZE)

        metadata_bytes = self._data_buffer.read(self._metadata_size)

        self._load_metadata_from_bytes(metadata_bytes)

        self._is_metadata_on_buffer = False

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

        if Metadata_Type.COLUMN_NAMES in self._metadata:
            self._column_name_index_map = {
                column_name: index for index, column_name in
                enumerate(self._metadata[Metadata_Type.COLUMN_NAMES])
            }

        if Metadata_Type.ROW_NAMES in self._metadata:
            self._row_name_index_map = {
                row_name: index for index, row_name in
                enumerate(self._metadata[Metadata_Type.ROW_NAMES])
            }

    def set_file_path(self, file_path):

        file_buffer = open(file_path, "rb")
        if file_path.endswith(".gz"):
            self._data_buffer = io.BytesIO(zlib.decompress(file_buffer.read()))
            file_buffer.close()
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

        self._is_metadata_on_buffer = True
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

        # We don't load the data until it is requested
        self._is_data_on_buffer = True
        
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
            numpy.append(self._row_start_indices, self._num_entries)

        row_start_indices_plus_one = row_start_indices_plus_one[1:]
        self._row_lengths = numpy.subtract(row_start_indices_plus_one,
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

        column_start_indices_plus_one = \
            numpy.append(self._column_start_indices, self._num_entries)

        column_start_indices_plus_one = column_start_indices_plus_one[1:]
        self._column_lengths = numpy.subtract(
            column_start_indices_plus_one,
            self._column_start_indices)

    def load_all_data(self):

        if not self._is_data_on_buffer:
            return

        if not self._data_buffer:
            self._is_data_on_buffer = False
            return

        self._data_buffer.seek(self._row_data_start_byte)

        row_columns_read_length = self._num_bytes_row_entry * self._num_entries

        self._row_column_indices = numpy.ndarray(
            (self._num_entries,),
            self._pack_format_column_index,
            self._data_buffer.read(row_columns_read_length),
            strides=self._num_bytes_row_entry
        )

        self._data_buffer.seek(self._row_data_start_byte +
                               self._num_bytes_column_index)

        row_data_read_length = \
            (self._num_bytes_column_index + self._data_size) \
            * self._num_entries - self._num_bytes_column_index

        self._row_data = numpy.ndarray(
            (self._num_entries,),
            self._pack_format_data,
            self._data_buffer.read(row_data_read_length),
            strides=self._data_size+self._num_bytes_column_index
        )

        column_start_byte = self._data_buffer.tell()

        column_rows_read_length = self._num_bytes_column_entry * \
            self._num_entries

        self._column_row_indices = numpy.ndarray(
            (self._num_entries,),
            self._pack_format_row_index,
            self._data_buffer.read(column_rows_read_length),
            strides=self._num_bytes_column_entry
        )

        self._data_buffer.seek(column_start_byte + self._num_bytes_row_index)

        column_data_read_length = \
            (self._num_bytes_row_index + self._data_size) \
            * self._num_entries - self._num_bytes_row_index

        self._column_data = numpy.ndarray(
            (self._num_entries,),
            self._pack_format_data,
            self._data_buffer.read(column_data_read_length),
            strides=self._num_bytes_column_entry
        )

        self._is_data_on_buffer = False
