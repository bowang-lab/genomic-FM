import os
import subprocess
import kipoiseq
from src.datasets.gene_ko.load_gene_position_gtf import CustomAnchoredGTFDl


dl = CustomAnchoredGTFDl(num_upstream=1, num_downstream=1)
dl.filter_by_gene("SHOC2")
print(dl[0])
# dl.filter_by_gene('"SHOC2"')
# print(dl._gtf_anchor)
# print(f" len = {len(dl)}")
