import sys
sys.path.insert(0, 'efficientdet-pytorch')

import cv2
import torch
import numpy as np
import albumentations as A

from effdet import (
    get_efficientdet_config, 
    EfficientDet, 
    DetBenchPredict,
    DetBenchTrain, 
    create_model
)
from effdet.efficientdet import HeadNet
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from albumentations.pytorch import ToTensorV2


np.set_printoptions(threshold=sys.maxsize)


model = create_model(
        'efficientdet_d0', 
        bench_task='predict', 
        num_classes=90 , 
        # image_size=(IMAGE_SIZE,IMAGE_SIZE),
        pretrained=True
    )

image = cv2.imread('data/image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgb = cv2.resize(image, (512, 512))


transform = A.Compose([
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(p=1.0),
    ])

image_rgb = transform(image=image_rgb)['image']
# image_rgb = np.transpose(image_rgb, (2, 0, 1))
image_rgb = np.expand_dims(image_rgb, 0)
tensor = torch.tensor(image_rgb, dtype=torch.float32)

model.eval()
with torch.no_grad():
    outputs = model(tensor)

classes = {
    16: 'bird',
    17: 'bird',
    82: 'bird',
    86: 'bird'
}

colors = {
    'bird': (0, 255, 0)
}

num_detected = 0
threshold=0.5

def draw_bboxes(image, outputs, w, h):

    global num_detected

    orig_h, orig_w = image.shape[0], image.shape[1]

    scores = outputs[:, 4].cpu()
    # threshold_indices = [scores.index(i) for i in scores if i > 0.2]
    labels = outputs[:, 5]
    
    bboxes = outputs[:, :4].detach().cpu().numpy()

    boxes = bboxes[scores >= threshold]

    # print(boxes)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i, box in enumerate(boxes):
        box_0 = ((box[0]/w)*orig_w)
        box_1 = ((box[1]/h)*orig_h)
        box_2 = ((box[2]/w)*orig_w)
        box_3 = ((box[3]/h)*orig_h)
        cv2.rectangle(
            image,
            (int(box_0), int(box_1)),
            (int(box_2), int(box_3)),
            colors[classes[int(labels[i])]], 2
        )
        cv2.putText(image, classes[int(labels[i])], (int(box_0), int(box_1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[classes[int(labels[i])]], 2, 
                    lineType=cv2.LINE_AA)
        if labels[i] == 1:
            num_detected += 1
    return image

result = draw_bboxes(image, outputs[0], 512, 512)
cv2.imshow('Result', result)
cv2.waitKey(0)