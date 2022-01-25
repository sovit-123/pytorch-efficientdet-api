import sys
sys.path.insert(0, 'efficientdet-pytorch')

from torch_utils.engine import (
    train_one_epoch, evaluate
)
from config import (
    DEVICE, NUM_CLASSES,
    NUM_EPOCHS, NUM_WORKERS,
    OUT_DIR
)
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)

from custom_utils import (
    save_model, 
    save_train_loss_plot,
    Averager
)
from effdet import (
    get_efficientdet_config, 
    EfficientDet, 
    DetBenchPredict,
    DetBenchTrain,
    create_model
)
from effdet.efficientdet import HeadNet
from tqdm.auto import tqdm

import torch

def get_net(model_name):
    config = get_efficientdet_config(model_name)
    config.norm_kwargs=dict(eps=.001, momentum=.01)
    config.num_classes = NUM_CLASSES
    # config.image_size = [640, 640]
    net = EfficientDet(config, pretrained_backbone=True)
    # checkpoint = torch.load('weights/efficientdet_d0-f3276ba8.pth')
    # net.load_state_dict(checkpoint)
    
    net.class_net = HeadNet(config, num_outputs=NUM_CLASSES)
    return DetBenchTrain(net, config)

if __name__ == '__main__':
    device = 'cuda:0'
    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    #### FROM HERE
    # model_name = 'efficientdet_d0'
    # model = get_net(model_name)

    # model = model.to(DEVICE)
    # # Get the model parameters.
    # params = [p for p in model.parameters() if p.requires_grad]
    # # Define the optimizer.
    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # for epoch in range(NUM_EPOCHS):
    #     print(f"EPOCH: {epoch}")
    #     for step, (images, targets) in tqdm(
    #         enumerate(train_loader), 
    #         total=len(train_loader)
    #     ):
    #         images = torch.stack(images)
    #         images = images.to(DEVICE).float()
    #         batch_size = images.shape[0]
    #         boxes = [target['boxes'].to(DEVICE).float() for target in targets]
    #         labels = [target['labels'].to(DEVICE).float() for target in targets]
            
    #         target_res = {}
    #         target_res['bbox'] = boxes
    #         target_res['cls'] = labels

    #         optimizer.zero_grad()

    #         outputs = model(images, target_res)
    #         loss = outputs['loss']
    #         class_loss = outputs['class_loss']
    #         box_loss = outputs['box_loss']

    #         loss.backward()

    #         optimizer.step()
    #     print(loss)
    #### TO HERE
    

    # Initialize the Averager class.
    train_loss_hist = Averager()
    # Train and validation loss lists to store loss values of all
    # iterations till ena and plot graphs for all iterations.
    train_loss_list = []

    # Initialize the model and move to the computation device.
    # model_name = 'tf_efficientdet_d0'
    # model = get_net(model_name)

    model = create_model(
        'efficientdet_d0', 
        bench_task='train', 
        num_classes=NUM_CLASSES , 
        # image_size=(IMAGE_SIZE,IMAGE_SIZE),
        bench_labeler=True,
        pretrained=True
    )

    model = model.to(DEVICE)
    # Get the model parameters.
    params = [p for p in model.parameters() if p.requires_grad]
    # Define the optimizer.
    # optimizer = torch.optim.AdamW(params, lr=0.001)
    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.005)
    optimizer = torch.optim.AdamW(params, lr= 2e-4, weight_decay = 1e-6)

    for epoch in range(NUM_EPOCHS):
        train_loss_hist.reset()

        _, batch_loss_list = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            DEVICE, 
            epoch, 
            train_loss_hist,
            print_freq=100
        )

        evaluate(model, valid_loader, device=DEVICE)

        # Add the current epoch's batch-wise lossed to the `train_loss_list`.
        train_loss_list.extend(batch_loss_list)

        # Save the current epoch model.
        save_model(epoch, model, optimizer)

        # Save loss plot.
        save_train_loss_plot(OUT_DIR, train_loss_list)