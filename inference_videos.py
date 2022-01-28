import sys
sys.path.insert(0, 'efficientdet-pytorch')

from effdet import create_model
from config import NUM_CLASSES, CLASSES, RESIZE_TO
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from albumentations.pytorch import ToTensorV2

import numpy as np
import cv2
import os
import torch
import time
import albumentations as A
import argparse
import pathlib

np.random.seed(42)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='data/uno_custom_test_data/video_1.mp4'
)
args = vars(parser.parse_args())

# This will help us create a different color for each class.
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Define the detection threshold.
detection_threshold = 0.3

def draw_bboxes(image, outputs, w, h):
    orig_h, orig_w = image.shape[0], image.shape[1]

    scores = outputs[:, 4].cpu()
    labels = outputs[:, 5]
    
    bboxes = outputs[:, :4].detach().cpu().numpy()

    boxes = bboxes[scores >= detection_threshold]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Notice the -1 in the color and class indices in the
    # following annotations. The model predicts from [1, NUM_CLASSES],
    # but class indices start from 0. So, to manage that we have to -1
    # from the predicted label number.
    for i, box in enumerate(boxes):
        box_0 = ((box[0]/w)*orig_w)
        box_1 = ((box[1]/h)*orig_h)
        box_2 = ((box[2]/w)*orig_w)
        box_3 = ((box[3]/h)*orig_h)
        cv2.rectangle(
            image,
            (int(box_0), int(box_1)),
            (int(box_2), int(box_3)),
            COLORS[int(labels[i])-1], 2
        )
        cv2.putText(
            image, 
            CLASSES[int(labels[i])-1], 
            (int(box_0), int(box_1-10)),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            COLORS[int(labels[i])-1], 2, 
            lineType=cv2.LINE_AA
        )
    return image

transform = A.Compose([
    A.Resize(RESIZE_TO, RESIZE_TO),
    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ToTensorV2(p=1.0),
])

model = create_model(
    'tf_efficientdet_lite0', 
    bench_task='predict', 
    num_classes=NUM_CLASSES, 
    pretrained=False,
)
checkpoint = torch.load('outputs/training/last_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

device = 'cuda:0'

model.to(device).eval()

cap = cv2.VideoCapture(args['input'])

if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = str(pathlib.Path(args['input'])).split(os.path.sep)[-1].split('.')[0]
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"outputs/inference/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

# To count the total number of images iterated through
frame_count = 0
# To keep adding the FPS for each image
total_fps = 0 

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret:
        image = frame.copy()
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = transform(image=image)['image']
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image.to(device))
        end_time = time.time()

        time_taken = (end_time-start_time)
        # get the current fps
        fps = 1 / (time_taken)
        # add `fps` to `total_fps`
        total_fps += fps
        # increment frame count
        frame_count += 1
    
        result = draw_bboxes(frame, outputs[0], RESIZE_TO, RESIZE_TO)
        cv2.putText(
            result, 
            f"{fps:.1f} FPS", 
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2, 
            lineType=cv2.LINE_AA
        )
        cv2.imshow('Prediction', result)
        out.write(result)
        print(f"Frame {frame_count}, time taken: {(time_taken):.3f}")
        # press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")