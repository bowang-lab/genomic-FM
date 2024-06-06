from src.dataloader.data_wrapper import RealClinVar, OligogenicDataWrapper, MAVEDataWrapper
from src.dataloader.data_wrapper import GWASDataWrapper, ClinVarDataWrapper, GeneKoDataWrapper
from src.dataloader.data_wrapper import CellPassportDataWrapper, eQTLDataWrapper,sQTLDataWrapper


NUM_RECORDS = 1000
ALL_RECORDS = False
SEQ_LEN = 20

# load clinician verified clinvar data
data_loader = RealClinVar(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# load oligogenic data
data_loader = OligogenicDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# load ClinVar data
data_loader = ClinVarDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# load GeneKo data
data_loader = GeneKoDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# load CellPassport data
data_loader = CellPassportDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)


# load eQTL data
data_loader = eQTLDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# load sQTL data
data_loader = sQTLDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# load MAVE data
data_loader = MAVEDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# load GWAS data
data_loader = GWASDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)
