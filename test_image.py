"""
Script to run inference on any given image using COCO pretrained model

USAGE:
python test_image.py --input <path/to/input/image> --model <model_name>

`model_name` can be:
efficientdet_d0
efficientdet_d1
...
"""

import cv2
import torch
import numpy as np
import albumentations as A
import argparse
import yaml
import os

from models.efficientdet_model import create_effdet_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from albumentations.pytorch import ToTensorV2
from custom_utils import set_infer_dir, draw_bboxes

np.random.seed(42)

def read_and_return_image(image_path):
    image = cv2.imread(image_path)
    assert image is not None, 'Please provde a correct input image path...'
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return image, image_rgb

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
        '-i', '--input', default=None,
        help='path to the input image'
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
    with open('data_configs/test_image_config.yaml') as file:
        data_configs = yaml.safe_load(file)

    # Inference settings and constants
    IMAGE_WIDTH = int(model_configs[args['model']][0]['image_width'])
    IMAGE_HEIGHT = int(model_configs[args['model']][1]['image_height'])
    NUM_CLASSES = data_configs['nc']
    CLASSES = data_configs['classes']
    OUT_DIR = set_infer_dir()
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    IMAGE_PATH = None
    if args['input'] == None:
        IMAGE_PATH = data_configs['image_path']
    else:
        IMAGE_PATH = args['input']
    assert IMAGE_PATH is not None, 'Please provide path to an input image...'

    image, image_rgb = read_and_return_image(IMAGE_PATH)
    image_tensor = preprocess(image_rgb)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    # Load the pretrained model
    model = create_effdet_model(
        model_name=args['model'],
        num_classes=NUM_CLASSES,
        pretrained=True,
        task='predict',
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
    
    result = draw_bboxes(
        image, outputs[0], 
        IMAGE_WIDTH, IMAGE_HEIGHT,
        args['threshold'],
        COLORS, CLASSES
    )
    save_name = IMAGE_PATH.split(os.path.sep)[-1].split('.')[0]
    cv2.imshow('Prediction', result)
    cv2.waitKey(0)
    cv2.imwrite(f"{OUT_DIR}/{save_name}.jpg", result)
    cv2.destroyAllWindows()