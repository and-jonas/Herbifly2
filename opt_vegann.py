
import os

import torchmetrics

from torch.utils.data.sampler import RandomSampler

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from flash.image import SemanticSegmentation, SemanticSegmentationData

from flash import Trainer

from transforms import set_input_transform_options
from BatchSizeFinder import find_max_batch_size_simple

import torch.multiprocessing

metrics = [
    torchmetrics.Accuracy(num_classes=2, mdmc_reduce='samplewise', average='none', multiclass=False),
    torchmetrics.F1Score(num_classes=2, mdmc_reduce='samplewise', average='none', multiclass=False),
    torchmetrics.Precision(num_classes=2, mdmc_reduce='samplewise', average='none', multiclass=False),
    torchmetrics.Recall(num_classes=2, mdmc_reduce='samplewise', average='none', multiclass=False),
    ]

# specify model hyper-parameters
loss_fn = torch.nn.CrossEntropyLoss()
crop_factor = 0.688
head = "unetplusplus"
backbone = "resnet34"
strategy = "no_freeze"
learning_rate = 0.09
momentum = 0.88
optimizer = "sgd"
blur_kernel_size = 1
size = 512
p_color_jitter = 0.0
pretrained = False if strategy == "train" else True

# fix data augmentation
transform = set_input_transform_options(head=head,
                                        size=size,
                                        crop_factor=crop_factor,
                                        blur_kernel_size=blur_kernel_size,
                                        p_color_jitter=p_color_jitter)

# find maximum batch size (current set-up)
batch_size = find_max_batch_size_simple(
    backbone=backbone,
    head=head,
    size=size,
    crop_factor=crop_factor,
    max_batch_size=83,
)

datamodule = SemanticSegmentationData.from_folders(
    # train_folder="data/SegVeg/train/images",
    # train_target_folder="data/SegVeg/train/masks",
    # val_folder="data/SegVeg/validation/images",
    # val_target_folder="data/SegVeg/validation/masks",
    train_folder="/projects/EWS_herbifly/data3/Training/images",
    train_target_folder="/projects/EWS_herbifly/data3/Training/masks",
    val_folder="/projects/EWS_herbifly/data3/Validation/images",
    val_target_folder="/projects/EWS_herbifly/data3/Validation/masks",
    train_transform=transform,
    val_transform=transform,
    test_transform=transform,
    predict_transform=transform,
    num_classes=2,
    batch_size=batch_size,
    num_workers=8,
    sampler=RandomSampler,
)

# Build the task
model = SemanticSegmentation(
    pretrained=pretrained,
    backbone=backbone,
    head=head,
    num_classes=datamodule.num_classes,
    metrics=metrics,
    loss_fn=loss_fn,
    optimizer=(optimizer, {"momentum": momentum}),
    learning_rate=learning_rate,
)

early_stopping = EarlyStopping(monitor='val_f1score', mode='max', patience=10, min_delta=0)
lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)
checkpointing = ModelCheckpoint(
    dirpath="/projects/EWS_herbifly",
    save_top_k=1,
    monitor="val_f1score",
    mode="max",
    filename="VegAnn_herbifly_best-{epoch:02d}-{step:.2f}",
    save_weights_only=True,
    # save_on_train_epoch_end=False,
)

logger = TensorBoardLogger(save_dir="./loggers",
                           name="segveg_vegann")

logger.log_hyperparams({"head": head,
                        "backbone": backbone,
                        "strategy": strategy,
                        "blur_kernel_size": blur_kernel_size,
                        "optimizer": optimizer,
                        "learning_rate": learning_rate,
                        "momentum": momentum,
                        "size": size,
                        "p_color_jitter": p_color_jitter,
                        "batch_size": datamodule.batch_size})

trainer = Trainer(max_epochs=150,
                  move_metrics_to_cpu=True,
                  gpus=[3],
                  precision=16,
                  logger=logger,
                  callbacks=[early_stopping, lr_monitor],
                  # reload_dataloaders_every_n_epochs=1,
                  enable_checkpointing=True,
                  # log_every_n_steps=10,
                  )

# train or finetune
if strategy == "train":
    trainer.fit(model, datamodule=datamodule)
else:
    trainer.finetune(model, datamodule=datamodule, strategy=strategy)

# 5. Save the model!
trainer.save_checkpoint("./vegAnn_herbifly.pt")

# extract highest score and log this value
event_acc = EventAccumulator(path=logger.log_dir)
event_acc.Reload()
tags = event_acc.Tags()['scalars']
v = event_acc.Scalars('val_f1score')
v = [v[i].value for i in range(len(v))]
value = max(v)
logger.log_metrics({"hp_metric": value})

