from src.dataloader.data_wrapper import ClinVarDataWrapper
import json
ALL_RECORDS = True
NUM_RECORDS = 30000
SEQ_LEN = 10


data_loader = ClinVarDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN,target='CLNDN',disease_subset=True)
