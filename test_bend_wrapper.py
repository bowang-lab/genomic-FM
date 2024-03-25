from src.model_wrapper.base_model import BaseModel


model = BaseModel(model_initiator_name='hyenadna-tiny-1k')
sequence = 'ATGATATAAG'
output = model(sequence)
print(output.shape)
