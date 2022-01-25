import sys
sys.path.insert(0, 'efficientdet-pytorch')

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain

model_name = 'efficientdet_d0'

config = get_efficientdet_config(model_name)
print(config)