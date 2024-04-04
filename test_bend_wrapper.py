from src.model_wrapper.base_model import BaseModel


model = BaseModel(model_initiator_name='hyenadna-tiny-1k')
sequence = ['TGGGCCGCTCGCCCCGTATC','TGGGCCGCTCGCCCCGTATC']
output = model(sequence)
for i in output:
    print(i.shape)
