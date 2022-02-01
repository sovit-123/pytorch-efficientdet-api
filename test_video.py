"""
Script to run inference on any given video using COCO pretrained model

USAGE:
python test_image.py --input <path/to/input/image> --model <model_name>

`model_name` can be:
efficientdet_d0
efficientdet_d1
tf_efficientdet_lite0
...
"""

import cv2
import torch
import numpy as np
import albumentations as A
import argparse
import yaml
import os
import time

from models.efficientdet_model import create_effdet_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from albumentations.pytorch import ToTensorV2
from custom_utils import set_infer_dir, draw_bboxes

np.random.seed(42)

def read_return_video_data(video_path):
    cap = cv2.VideoCapture(video_path)
    # Get the video's frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    assert (frame_width != 0 and frame_height !=0), 'Please check video path...'
    return cap, frame_width, frame_height

def preprocess(image):
    transform = A.Compose([
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
            ToTensorV2(p=1.0),
        ])
    transformed_image = transform(image=image)['image']
    return transformed_image

if __name__ == '__main__':
    # Construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', default='efficientdet_d0',
        help='name of the model'
    )
    parser.add_argument(
        '-c', '--config', 
        default='data_configs/test_video_config.yaml',
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-w', '--weights', default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-i', '--input', default=None,
        help='path to the input video'
    )
    parser.add_argument(
        '-th', '--threshold', default=0.3, type=float,
        help='detection threshold'
    )
    args = vars(parser.parse_args())

    # Load the model configurations
    with open('model_configs/model_config.yaml') as file:
        model_configs = yaml.safe_load(file)
    # Load the data configurations
    with open(args['config']) as file:
        data_configs = yaml.safe_load(file)

    # Inference settings and constants
    IMAGE_WIDTH = int(model_configs[args['model']][0]['image_width'])
    IMAGE_HEIGHT = int(model_configs[args['model']][1]['image_height'])
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    OUT_DIR = set_infer_dir()
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    VIDEO_PATH = None
    if args['input'] == None:
        VIDEO_PATH = data_configs['video_path']
    else:
        VIDEO_PATH = args['input']
    assert VIDEO_PATH is not None, 'Please provide path to an input video...'

    cap, frame_width, frame_height = read_return_video_data(VIDEO_PATH)

    save_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0]
    # Define codec and create VideoWriter object.
    out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4", 
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (frame_width, frame_height))

    # Load the pretrained model
    model = create_effdet_model(
        model_name=args['model'],
        num_classes=NUM_CLASSES,
        pretrained=True,
        task='predict',
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        bench_labeler=False
    )
    if args['weights'] is not None:
        checkpoint = torch.load(args['weights'])
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    # To count the total number of images iterated through
    frame_count = 0
    # To keep adding the FPS for each image
    total_fps = 0 

    # Read until end of video.
    while (cap.isOpened()):
        # Capture each frame of video.
        ret, frame = cap.read()
        if ret:
            image = frame.copy()
            # Convert BGR to RGB.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Preprocessing and transform.
            image_tensor = preprocess(image_rgb)
            image_tensor = torch.unsqueeze(image_tensor, 0)

            start_time = time.time()
            # Forward pass.
            with torch.no_grad():
                outputs = model(image_tensor.to(DEVICE))
            end_time = time.time()

            time_taken = (end_time-start_time)
            # Get the current fps.
            fps = 1 / (time_taken)
            # Add `fps` to `total_fps`.
            total_fps += fps
            # Increment frame count.
            frame_count += 1

            result = draw_bboxes(
                image, outputs[0], 
                IMAGE_WIDTH, IMAGE_HEIGHT,
                args['threshold'],
                COLORS, CLASSES
            )
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
            # Press `q` to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cv2.destroyAllWindows()
    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")