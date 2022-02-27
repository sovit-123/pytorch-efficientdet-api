"""
Script to run inference on any given image using COCO pretrained model

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
import matplotlib.pyplot as plt

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
        '-c', '--config', 
        default='data_configs/test_image_config.yaml',
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-w', '--weights', default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-i', '--input', default=None,
        help='path to the input image'
    )
    parser.add_argument(
        '-th', '--threshold', default=0.3, type=float,
        help='detection threshold'
    )
    parser.add_argument(
        '-si', '--show-image', dest='show_image', action='store_true',
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-mpl', '--mpl-show', dest='mpl_show', action='store_true',
        help='visualize using matplotlib, helpful in notebooks'
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
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        bench_labeler=False
    )
    if args['weights'] is not None:
        checkpoint = torch.load(args['weights'])
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor.to(DEVICE))
    forward_end_time = time.time()
    
    result = draw_bboxes(
        image, outputs[0], 
        IMAGE_WIDTH, IMAGE_HEIGHT,
        args['threshold'],
        COLORS, CLASSES
    )
    final_end_time = time.time()

    forward_pass_time = forward_end_time - start_time
    forward_and_annot_time = final_end_time - start_time
    print(f"Forward pass time: {forward_pass_time:.3f} seconds")
    print(f"Forward pass + annotation time: {forward_and_annot_time:.3f} seconds")
    
    save_name = IMAGE_PATH.split(os.path.sep)[-1].split('.')[0]
    if args['show_image']:
        cv2.imshow('Prediction', result)
        cv2.waitKey(0)
    if args['mpl_show']:
        plt.imshow(result[:, :, ::-1])
        plt.axis('off')
        plt.show()
    cv2.imwrite(f"{OUT_DIR}/{save_name}.jpg", result)
    cv2.destroyAllWindows()