import sys
sys.path.insert(0, 'efficientdet-pytorch')

from effdet import create_model

def create_effdet_model(
    model_name=None,
    num_classes=None, 
    pretrained=True,
    task='train',
    image_size=None,
    checkpoint_path=None,
    bench_labeler=True
):
    model = create_model(
        model_name, 
        bench_task=task, 
        num_classes=num_classes, 
        image_size=image_size,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        bench_labeler=bench_labeler
    )
    return model