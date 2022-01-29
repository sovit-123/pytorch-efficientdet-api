import albumentations as A
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

plt.style.use('ggplot')

# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'outputs/best_model.pth')

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

# define the training tranforms
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# define the validation transforms
def get_valid_transform():
    return A.Compose([
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })


def show_transformed_image(train_loader, DEVICE, CLASSES):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.

    :param train_loader: Training data loader.  
    """
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box_num, box in enumerate(boxes):
                cv2.rectangle(sample,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 0, 255), 2)
                cv2.putText(sample, CLASSES[labels[box_num]], 
                            (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (0, 0, 255), 2)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def save_model(epoch, model, optimizer, OUT_DIR):
    """
    Function to save the trained model till current epoch, or whenever called.

    :param epoch: The epoch number.
    :param model: The neural network model.
    :param optimizer: The optimizer.
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'{OUT_DIR}/last_model.pth')

def save_loss_plot(OUT_DIR, train_loss_list, val_loss_list):
    """
    Function to save both train and validation loss graphs.
    
    :param OUT_DIR: Path to save the graphs.
    :param train_loss_list: List containing the training loss values.
    :param val_loss_list: List containing the validation loss values.
    """
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(val_loss_list, color='tab:red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')
    figure_1.savefig(f"{OUT_DIR}/train_loss.png")
    figure_2.savefig(f"{OUT_DIR}/valid_loss.png")
    print('SAVING PLOTS COMPLETE...')
    plt.close('all')

def save_train_loss_plot(OUT_DIR, train_loss_list):
    """
    Function to save both train loss graph.
    
    :param OUT_DIR: Path to save the graphs.
    :param train_loss_list: List containing the training loss values.
    """
    figure_1, train_ax = plt.subplots()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    figure_1.savefig(f"{OUT_DIR}/train_loss.png")
    print('SAVING PLOTS COMPLETE...')
    plt.close('all')

def save_validation_results(images, detections, counter, OUT_DIR):
    """
    Function to save validation results if provided in `config.py`.

    :param images: All the images from the current batch.
    :param detections: All the detection results.
    :param counter: Step counter for saving with unique ID.
    """
    for i, detection in enumerate(detections):
        image_c = images[i].clone()
        image_c = image_c.detach().cpu().numpy().astype(np.float32)
        image = np.transpose(image_c, (1, 2, 0))
        image = image / 2 + 0.5
        image = np.ascontiguousarray(image, dtype=np.float32)

        scores = detection[:, 4].cpu()
        labels = detection[:, 5]
        bboxes = detection[:, :4].detach().cpu().numpy()
        boxes = bboxes[scores >= 0.3]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for j, box in enumerate(boxes):
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255, 255, 255), 2
            )
        cv2.imwrite(f"{OUT_DIR}/image_{i}_{counter}.jpg", image*255.)

def set_infer_dir():
    """
    This functions counts the number of inference directories already present
    and creates a new one in `outputs/inference/`. 
    And returns the directory path.
    """
    if not os.path.exists('outputs/inference'):
        os.makedirs('outputs/inference')
    num_infer_dirs_present = len(os.listdir('outputs/inference/'))
    next_dir_num = num_infer_dirs_present + 1
    new_dir_name = f"outputs/inference/res_{next_dir_num}"
    os.makedirs(new_dir_name, exist_ok=True)
    return new_dir_name

def set_training_dir():
    """
    This functions counts the number of training directories already present
    and creates a new one in `outputs/training/`. 
    And returns the directory path.
    """
    if not os.path.exists('outputs/training'):
        os.makedirs('outputs/training')
    num_train_dirs_present = len(os.listdir('outputs/training/'))
    next_dir_num = num_train_dirs_present + 1
    new_dir_name = f"outputs/training/res_{next_dir_num}"
    os.makedirs(new_dir_name, exist_ok=True)
    return new_dir_name

def draw_bboxes(
    image, 
    outputs, 
    w, h, 
    detection_threshold, 
    colors, 
    classes
):
    """
    Function draws bounding boxes around the `image` and returns
    the result.
    """
    orig_h, orig_w = image.shape[0], image.shape[1]
    scores = outputs[:, 4].cpu()
    labels = outputs[:, 5]
    bboxes = outputs[:, :4].detach().cpu().numpy()
    boxes = bboxes[scores >= detection_threshold]

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
            colors[int(labels[i])-1], 2
        )
        cv2.putText(
            image, 
            classes[int(labels[i])-1], 
            (int(box_0), int(box_1-10)),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            colors[int(labels[i])-1], 2, 
            lineType=cv2.LINE_AA
        )
    return image