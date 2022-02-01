from torch_utils.engine import (
    train_one_epoch, evaluate
)
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
from custom_utils import (
    save_model_state,
    save_train_loss_plot,
    Averager
)
from models.efficientdet_model import create_effdet_model
from custom_utils import set_training_dir

import torch
import argparse
import yaml

if __name__ == '__main__':
    # Construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', default='efficientdet_d0',
        help='name of the model'
    )
    parser.add_argument(
        '-c', '--config', default=None,
        help='path to the data config file'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-e', '--epochs', default=5, type=int,
        help='number of epochs to train for'
    )
    args = vars(parser.parse_args())

    # Load the model configurations
    with open('model_configs/model_config.yaml') as file:
        model_configs = yaml.safe_load(file)
    # Load the data configurations
    with open(args['config']) as file:
        data_configs = yaml.safe_load(file)
    
    # Settings/parameters/constants.
    TRAIN_DIR = data_configs['TRAIN_DIR']
    VALID_DIR = data_configs['VALID_DIR']
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']
    NUM_WORKERS = data_configs['NUM_WORKERS']
    DEVICE = args['device']
    NUM_EPOCHS = args['epochs']
    SAVE_VALID_PREDICTIONS = data_configs['SAVE_VALID_PREDICTION_IMAGES']
    BATCH_SIZE = data_configs['BATCH_SIZE']
    OUT_DIR = set_training_dir()

    # Model configurations
    IMAGE_WIDTH = int(model_configs[args['model']][0]['image_width'])
    IMAGE_HEIGHT = int(model_configs[args['model']][1]['image_height'])
    device = 'cuda:0'
    train_dataset = create_train_dataset(
        TRAIN_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES
    )
    valid_dataset = create_valid_dataset(
        VALID_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES
    )
    train_loader = create_train_loader(train_dataset, BATCH_SIZE, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Initialize the Averager class.
    train_loss_hist = Averager()
    # Train and validation loss lists to store loss values of all
    # iterations till ena and plot graphs for all iterations.
    train_loss_list = []

    model = create_effdet_model(
        model_name=args['model'],
        num_classes=NUM_CLASSES,
        pretrained=True,
        task='train',
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    model = model.to(DEVICE)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Get the model parameters.
    params = [p for p in model.parameters() if p.requires_grad]
    # Define the optimizer.
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.005)
    # optimizer = torch.optim.AdamW(params, lr= 2e-4, weight_decay = 1e-6)
    # LR will be zero as we approach `steps` number of epochs each time.
    # If `steps = 5`, LR will slowly reduce to zero every 5 epochs.
    steps = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=steps,
        T_mult=1,
        verbose=True
    )

    for epoch in range(NUM_EPOCHS):
        train_loss_hist.reset()

        _, batch_loss_list = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            DEVICE, 
            epoch, 
            train_loss_hist,
            print_freq=100,
            scheduler=None
        )

        evaluate(
            model, 
            valid_loader, 
            device=DEVICE,
            save_valid_preds=SAVE_VALID_PREDICTIONS,
            out_dir=OUT_DIR
        )

        # Add the current epoch's batch-wise lossed to the `train_loss_list`.
        train_loss_list.extend(batch_loss_list)

        # Save the current epoch model state. This can be used 
        # to resume training. It saves model state dict, number of
        # epochs trained for, optimizer state dict, and loss function.
        save_model_state(epoch, model, optimizer, OUT_DIR)

        # Save loss plot.
        save_train_loss_plot(OUT_DIR, train_loss_list)