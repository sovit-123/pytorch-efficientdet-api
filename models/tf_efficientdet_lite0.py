import sys
sys.path.insert(0, 'efficientdet-pytorch')

from effdet import create_model

def create_effdet_model(num_classes=None, pretrained=True):
    model = create_model(
        'tf_efficientdet_lite0', 
        bench_task='train', 
        num_classes=num_classes , 
        # image_size=(IMAGE_SIZE,IMAGE_SIZE),
        bench_labeler=True,
        pretrained=pretrained
    )
    return model