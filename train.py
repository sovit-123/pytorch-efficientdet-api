import sys
sys.path.insert(0, 'efficientdet-pytorch')

from torch_utils.engine import (
    train_one_epoch, evaluate
)
from config import (
    DEVICE, NUM_CLASSES,
    NUM_EPOCHS, NUM_WORKERS,
    OUT_DIR, SAVE_VALID_PREDICTIONS
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
    create_model
)

import torch

if __name__ == '__main__':
    device = 'cuda:0'
    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Initialize the Averager class.
    train_loss_hist = Averager()
    # Train and validation loss lists to store loss values of all
    # iterations till ena and plot graphs for all iterations.
    train_loss_list = []

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
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.005)
    # optimizer = torch.optim.AdamW(params, lr= 2e-4, weight_decay = 1e-6)
    steps = 1 # LR will be zero as we approach `steps`.
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
            save_valid_preds=SAVE_VALID_PREDICTIONS
        )

        # Add the current epoch's batch-wise lossed to the `train_loss_list`.
        train_loss_list.extend(batch_loss_list)

        # Save the current epoch model.
        save_model(epoch, model, optimizer)

        # Save loss plot.
        save_train_loss_plot(OUT_DIR, train_loss_list)