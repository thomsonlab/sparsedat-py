# sparsedat-py

sparsedat-py is a Python package to access and manipulate a Sparse Data
Table (SDT) file, as described by the specifications here:
[sparsedat GitHub](https://github.com/thomsonlab/sparsedat)

## Requirements
sparsedat requires the following Python packages be installed:
```
scipy
numpy
pandas
```

## Installation
sparsedat can be installed via Pip, e.g.:
```
pip install --upgrade https://github.com/thomsonlab/sparsedat-py
```

## Usage

### Importing

To use the Sparse_Data_Table object, import via:
```
from sparsedat import Sparse_Data_Table
```

### Workflow
In general, the workflow of sparsedat proceeds like this:

1. Create an SDT file from existing data/file
2. Initialize a Sparse_Data_Table object from an SDT file path
(optionally without loading the whole file from disk)
3. View and manipulate the object
4. If changes want to be saved, save the object

### Key Considerations
* sparsedat will not save any changes to files without you explicitly
calling the save function!
* sparsedat will by default save back to the same file path as created,
so be careful to change file paths if you don't want to overwrite!
* Indexing a sparsedat object returns a new sparsedat object, without
a file path specified. You must specify a file path to save subsampled
data
* There are no operations built in - you must convert a sparsedat
object to numpy or pandas in order to do calculations

### Creating an SDT file

There are currently 3 ways to create an SDT file. The first two, from
row-column values and sparse representation, do not consider row and
column names. If you want to add row and column names, you can do that
separately.

#### From row-column values

If you have a list of row and column indices and their values, you can
use ```Sparse_Data_Table.from_row_column_values()```. Example usage:
```
from sparsedat import Sparse_Data_Table

row_column_values = [
    (0, 1, 5),
    (1, 2, 15),
    (5, 4, 2)
]

sdt = Sparse_Data_Table()
sdt.from_row_column_values(
    row_column_values,
    num_rows=8,
    num_columns=8,
    default_value=0
)

sdt.save(file_path="test.sdt")
```

#### From sparse row or sparse column representation

This operates the same as scipy's csr and csc initialization functions,
as in [scipy.sparse.csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)
```
from sparsedat import Sparse_Data_Table

# Specify the starting index of each row you have data for
row_start_indices = [0, 2, 3]

# Specify the column index of each row entry
row_column_indices = [1, 5, 1, 1, 2, 4]

# The data values
values = [10, 10, 2, 1, 1, 8]

sdt = Sparse_Data_Table()

sdt.from_sparse_row_entries(
    (
        values,
        row_column_indices,
        row_start_indices
    ),
    num_rows=len(row_start_indices),
    num_columns=8,
    default_value=0
)

sdt.save(file_path="test.sdt")
```

The same can be done in sparse column format with
```Sparse_Data_Table.from_sparse_column_entries```

#### Adding row/column names

To add row and/or column names to a loaded file, before saving:

```
...

sdt.row_names = ["Row 1", "Row 2", "Row 3"]
sdt.column_names = ["Col %i" % (i + 1) for i in range(8)]

sdt.save(file_path="test.sdt")
```

#### From mtx

Finally, there is a wrapper function for creating an SDT file from
[MTX format](https://math.nist.gov/MatrixMarket/formats.html#MMformat)

Example:
```
import os
test_data_directory = os.path.join("test", "data")

from sparsedat import wrappers as sparsedat_wrappers

sdt = sparsedat_wrappers.load_mtx(
    os.path.join(test_data_directory, "features.tsv"),
    os.path.join(test_data_directory, "barcodes.tsv"),
    os.path.join(test_data_directory, "matrix.mtx")
)

sdt.save("test_mtx.sdt")

```

### Using an SDT file

#### Loading

To load an SDT file:
```
from sparsedat import Sparse_Data_Table

sdt = Sparse_Data_Table("test_mtx.sdt")
```

Optionally, you can load an SDT file without loading it all to memory.
This will reduce the memory footprint, but will require reading from
disk each time you access it.
```
from sparsedat import Sparse_Data_Table

sdt = Sparse_Data_Table("test.sdt", load_on_demand=True)
```

#### Indexing

Sparsedat-py supports several types of indexing: location, boolean, or
name-based. Values can be either slices, lists, or individual values.
Using the test_mtx.sdt from above, here are some examples:

##### 
```
from sparsedat import Sparse_Data_Table

sdt = Sparse_Data_Table("test_mtx.sdt")

# Get value by direct location
sdt[0, 0]

# Get an entire row
sdt[42, :]

# Boolean indexing
sdt[:, [True, True, True, False, False, False, False, True, True, False]]

# Named indexing
sdt["ENSG00000187608\tISG15\tGene Expression", :]
```

#### Conversion

By default, indexing a Sparse_Data_Table returns another
Sparse_Data_Table. However, you may want to do arithmetic or other
actions using numpy or pandas objects. A Sparse_Data_Table object can
be converted as follows:
```
from sparsedat import Sparse_Data_Table

sdt = Sparse_Data_Table("test_mtx.sdt")

# To a numpy array
sdt.to_array()

# To a pandas object
sdt.to_pandas()
```

#### Adding rows and columns

Currently, directly updating values is not supported. However, adding
rows and columns is supported by providing all non-default values for
the new row/column, either by name or index. Using the test.sdt, with
column and row names added from above:

```
from sparsedat import Sparse_Data_Table

sdt = Sparse_Data_Table("test.sdt")

new_values = [
    (0, 5),
    (3, 5),
    (5, 2)
]

sdt.add_row(new_values, row_name="New Row")

# Or by name

new_values = {
    "Col 1": 2,
    "Col 2": 3,
    "Col 6": 1
}

sdt.add_row(new_values.items(), row_name="New Row by Name")

# Let's see what we added
sdt[["New Row", "New Row by Name"], :]
```