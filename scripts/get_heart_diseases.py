from src.dataloader.data_wrapper import ClinVarDataWrapperPrintPercent, ClinVarDataWrapper
import json
ALL_RECORDS = False
NUM_RECORDS = 300
SEQ_LEN = 10


data_loader = ClinVarDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(len(data))

data_loader = ClinVarDataWrapperPrintPercent( all_records=True)
data,bp = data_loader.get_data(Seq_length=SEQ_LEN,target='DISEASE_PATHOGENICITY',disease_subset=True)
print(len(data))
print("done")
