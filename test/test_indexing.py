import pandas
import numpy

import pepars
from pepars.plotting import plotting
plotting.init_notebook_mode()

import sparsedat
from sparsedat import Sparse_Data_Table
from sparsedat import Output_Mode

DATASET_NAME = "20190711_TC4_B"

original_gene_counts = Sparse_Data_Table(
    "/media/dibidave/mobinoodle/AAVomics/data/" + DATASET_NAME + "/transcriptome/transcripts/" + DATASET_NAME + "_transcript_counts.sdt"
)

markers = ["S100b", "Rbfox3", "Olig2", "Cldn5"]
viruses = ["AAV.CAP-B10"]

UMI_COUNT_THRESHOLD = None
VIRUS_TRANSCRIPT_COUNT_THRESHOLD = 0
MARKER_TRANSCRIPT_COUNT_THRESHOLD = 0

cell_type_barcode_filter = original_gene_counts[markers[0], :].to_array() > MARKER_TRANSCRIPT_COUNT_THRESHOLD

print(original_gene_counts["Rbfox3", "AATGAAGGTCTGTAAC-1"])

