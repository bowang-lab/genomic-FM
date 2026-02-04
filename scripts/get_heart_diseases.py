from src.dataloader.data_wrapper import ClinVarDataWrapperPrintPercent
import json
ALL_RECORDS = True
# NUM_RECORDS = 30000
SEQ_LEN = 10


# data_loader = ClinVarDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data_loader = ClinVarDataWrapperPrintPercent( all_records=ALL_RECORDS)
data,bp = data_loader.get_data(Seq_length=SEQ_LEN,target='DISEASE_PATHOGENICITY',disease_subset=True)
