import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from models.orbit.config import OrbitConfig
from models.orbit.model import Classifier

from data_module import HelocDataModule
from module import SupervisedTuningModule

# set the precision for matmul
torch.set_float32_matmul_precision("high")
print("number of gpus", torch.cuda.device_count())
print("number of cpus", os.cpu_count())

# prepare the data module
config = OrbitConfig()
data_module = HelocDataModule(config)

# prepare the dataloaders
train_dataloader = data_module.get_dataloader(
    indices=data_module.train_indices,
    ranked_indices=data_module.train_ranked_indices,
    similarity_scores=data_module.train_similarity_scores,
    batch_size=config.batch_size,
    top_k=config.top_k,
)
val_dataloader = data_module.get_dataloader(
    indices=data_module.val_indices,
    ranked_indices=data_module.val_ranked_indices,
    similarity_scores=data_module.val_similarity_scores,
    batch_size=1, # use batch size 1 for testing
    top_k=config.top_k,
)
test_dataloader = data_module.get_dataloader(
    indices=data_module.test_indices,
    ranked_indices=data_module.test_ranked_indices,
    similarity_scores=data_module.test_similarity_scores,
    batch_size=1, # use batch size 1 for testing
    top_k=config.top_k,
)

# load the tokenizer and the model for sequence classification
model = Classifier(config=config)
module = SupervisedTuningModule(
    model=model, 
    config=config,
)
module.model.train()

# initialize the trainer
save_dir = "results"
name = "orbit"
checkpoint_callback = ModelCheckpoint(
    filename="model-{epoch:02d}-{total_loss_val:.3f}", # name of the save_top_k model
    monitor="total_loss_val", # metric to monitor
    mode="min", # minimize the metric
    save_top_k=1, # save the top k models with filename.ckpt
    save_last=True, # save the last model with last.ckpt
)
logger = TensorBoardLogger(
    save_dir=save_dir, # directory to save logs
    name=name # subdirectory to save versions
)
early_stopping = EarlyStopping(
    monitor="total_loss_val", # metric to monitor
    min_delta=0.00, # minimum change to qualify as an improvement
    patience=config.patience, # number of validation checks with no improvement to wait before stopping
    verbose=True,
    mode="min" # minimize the metric
)
trainer = Trainer(
    accelerator="auto",
    devices=1, # disable DDP
    max_epochs=config.max_epochs,
    logger=logger,
    callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback, early_stopping],
    enable_model_summary=False,
    log_every_n_steps=1,
    num_sanity_val_steps=0,
)

# start training
trainer.fit(
    model=module,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# load the best model for evaluation
checkpoint = torch.load(checkpoint_callback.best_model_path, weights_only=False)
module.on_load_checkpoint(checkpoint)
module.model.eval()

# create new trainer to avoid redundant event file
trainer = Trainer(
    accelerator="auto",
    devices=1, # disable DDP
    logger=False, # no logger for testing
    enable_checkpointing=False, # no checkpointing for testing
)
trainer.test(
    model=module,
    dataloaders=test_dataloader,
)
trainer.test(
    model=module,
    dataloaders=val_dataloader,
)
