import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

from data_module import HelocDataModule
from module import SFTModule

from models.cls.config import CLSConfig

# set the precision for matmul
torch.set_float32_matmul_precision("high")
print("number of gpus", torch.cuda.device_count())
print("number of cpus", os.cpu_count())

# prepare the data module
config = CLSConfig()
data_module = HelocDataModule()


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name, 
    padding_side="left",
)
base_model = AutoModelForSequenceClassification.from_pretrained(
    config.model_name,
    num_labels=config.num_labels,
    dtype="auto",
    device_map="auto",
)
base_model.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules="all-linear",
    modules_to_save=["score"],
)
config.peft_config = peft_config
model = get_peft_model(base_model, peft_config)
module = SFTModule(
    model=model, 
    tokenizer=tokenizer,
    config=config,
)

# prepare the dataloaders
train_dataloader = data_module.get_dataloader(
    indices=data_module.train_indices,
    tokenizer=tokenizer,
    question_template=config.question_template,
    answer_template=config.answer_template,
    batch_size=config.batch_size,
)
val_dataloader = data_module.get_dataloader(
    indices=data_module.val_indices,
    tokenizer=tokenizer,
    question_template=config.question_template,
    answer_template=config.answer_template,
    batch_size=1,
)
test_dataloader = data_module.get_dataloader(
    indices=data_module.train_indices,
    tokenizer=tokenizer,
    question_template=config.question_template,
    answer_template=config.answer_template,
    batch_size=1,
)


# initialize the trainer
save_dir = "results"
name = "lora"
checkpoint_callback = ModelCheckpoint(
    filename="model-{epoch:02d}-{total_loss:.3f}", # name of the save_top_k model
    monitor="total_loss", # metric to monitor
    mode="min", # minimize the metric
    save_top_k=1, # save the top k models with filename.ckpt
    save_last=True, # save the last model with last.ckpt
)
logger = TensorBoardLogger(
    save_dir=save_dir, # directory to save logs
    name=name # subdirectory to save versions
)
trainer = Trainer(
    precision="bf16-mixed", # no need to manually convert to half precision in training_step
    accelerator="auto",
    devices=1, # disable DDP
    max_epochs=config.max_epochs,
    logger=logger,
    callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback],
    enable_model_summary=False,
    log_every_n_steps=1,
    num_sanity_val_steps=0,
    accumulate_grad_batches=config.accumulate_grad_batches,
)

# start training
trainer.fit(
    model=module,
    train_dataloaders=train_dataloader,
)

# load the best model for evaluation
checkpoint = torch.load(checkpoint_callback.best_model_path, weights_only=False)
module.on_load_checkpoint(checkpoint)
module.model.eval()

# create new trainer to avoid redundant event file
trainer = Trainer(
    precision="bf16-mixed", # no need to manually convert to half precision in training_step
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
