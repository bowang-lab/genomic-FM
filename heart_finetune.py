# from src.tunable_model.funetune import main
# data,model = main()
# x,y = data[0]
# model(x)

# from src.tunable_model.base_model import BaseModel

# model = BaseModel('InstaDeepAI/nucleotide-transformer-v2-500m-multi-species')
from src.tunable_model.hf_trainer import main
main()
