import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn

from torch_utils import utils
from torch_utils.coco_eval import CocoEvaluator
from torch_utils.coco_utils import get_coco_api_from_dataset
from custom_utils import save_validation_results

def train_one_epoch(
    model, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    train_loss_hist,
    print_freq, 
    scaler=None,
    scheduler=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # List to store batch losses.
    batch_loss_list = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    step_counter = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        step_counter += 1
        images = torch.stack(images)
        images = images.to(device)
        boxes = [target['boxes'].to(device).float() for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]
        
        target_res = {}
        target_res['bbox'] = boxes
        target_res['cls'] = labels

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, target_res)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()

        # The model outputs a loss dictionary and it already contains
        # the cumulative loss in the `loss` key. It should be backpropagated,
        # and not backpropagate `losses` which is the sum of all the losses.
        loss_to_backprop = loss_dict['loss']
        if scaler is not None:
            scaler.scale(loss_to_backprop).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_to_backprop.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        metric_logger.update(loss=loss_to_backprop)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        batch_loss_list.append(loss_to_backprop.detach().cpu())
        train_loss_hist.send(loss_to_backprop.detach().cpu())

        if scheduler is not None:
            scheduler.step(epoch + (step_counter/len(data_loader)))

    return metric_logger, batch_loss_list


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(
    model, 
    data_loader, 
    device,
    save_valid_preds=False
):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    counter = 0
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        counter += 1
        images = torch.stack(images)
        images = images.to(device)
        batch_size = images.shape[0]
        boxes = [target['boxes'].to(device).float() for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]
        
        target_res = {}
        target_res['bbox'] = boxes
        target_res['cls'] = labels
        target_res['img_scale'] = torch.tensor([1.0] * batch_size, dtype=torch.float).to(device)
        target_res['img_size'] = torch.tensor([images[0].shape[-2:]] * batch_size, dtype=torch.float).to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images, target_res)
        
        detections = outputs['detections']
 
        order = [1, 0, 3, 2]
        final_output = []
        for i, detection in enumerate(detections):
            detection_dict = {}
            detection_dict['boxes'] = detection[:, :4]
            # Executing the following line will do:
            # XYXY to YXYX if it is YXYX originally.
            detection_dict['boxes'] = detection_dict['boxes'][:, order] # Needed as the original targets are in YXYX format.
            detection_dict['labels'] = detection[:, 5]
            detection_dict['scores'] = detection[:, 4]
            final_output.append(detection_dict)

        final_output = [{k: v.to(cpu_device) for k, v in t.items()} for t in final_output]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, final_output)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        
        if save_valid_preds:
            save_validation_results(images, detections, counter)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator