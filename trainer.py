import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from methods.base.config import CLSConfig
from methods.snap.config import SNAPConfig
from methods.snap.model import SNAP

from data_module import HelocDataModule
from module import SFTModule

# set the precision for matmul
torch.set_float32_matmul_precision("high")
print("number of gpus", torch.cuda.device_count())
print("number of cpus", os.cpu_count())

CONFIG_MAPPINGS = {
    "snap": SNAPConfig,
    "calm": CLSConfig,
}

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run different sets of experiments")
    
    parser.add_argument(
        "--experiment_name", type=str,
        choices=[
            "calm",
            "snap",
        ],
        default="snap",
        help="The experiment to run.",
    )
    parser.add_argument(
        "--use_numerical_embedding", action="store_false",
        help="Whether to use numerical embeddings in the prompt encoder (snap).",
    )
    parser.add_argument(
        "--use_numerical_profiling", action="store_false",
        help="Whether to use numerical profiling in the prompt encoder (snap).",
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Load the configuration based on the experiment name

    config = CONFIG_MAPPINGS[args.experiment_name]()
    
    # Prepare the data module
    data_module = HelocDataModule()
    
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side="left",
        local_files_only=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    # Freeze the base model
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Prepare the LoRA configuration
        lora_config = LoraConfig(
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
        )
    
    # Initialize the model based on the experiment name
    torch.manual_seed(42)
    if args.experiment_name == "calm":
        config.lora_config = lora_config
        model = get_peft_model(base_model, lora_config)
    elif args.experiment_name == "snap":
        # Customize SNAP specific configurations for ablation study
        # Only one setting can be disabled at a time
        if not args.use_numerical_embedding:
            config.use_numerical_embedding = False
            args.experiment_name = "snap/without_numerical_embedding"
        elif not args.use_numerical_profiling:
            config.use_numerical_profiling = False
            args.experiment_name = "snap/without_numerical_profiling"
        else:
            args.experiment_name = "snap/full_model"
        
        # Set up the model
        config.lora_config = lora_config
        base_model = get_peft_model(base_model, lora_config)
        model = SNAP(
            config=config, 
            base_model=base_model,
        )
    
    # Initialize the SFT module
    module = SFTModule(
        model=model, 
        tokenizer=tokenizer,
        config=config,
        num_training_samples=len(data_module.train_indices),
    )
    
    # Prepare the dataloaders
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
        indices=data_module.test_indices,
        tokenizer=tokenizer,
        question_template=config.question_template,
        answer_template=config.answer_template,
        batch_size=1,
    )
    
    # Initialize the trainer
    save_dir = "results"
    checkpoint_callback = ModelCheckpoint(
        filename="model-{epoch:02d}-{total_loss:.3f}", # name of the save_top_k model
        monitor="total_loss", # metric to monitor
        mode="min", # minimize the metric
        save_top_k=1, # save the top k models with filename.ckpt
        save_last=True, # save the last model with last.ckpt
    )
    logger = TensorBoardLogger(
        save_dir=save_dir, # directory to save logs
        name=args.experiment_name # subdirectory to save versions
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
    )
    
    # Start training
    trainer.fit(
        model=module,
        train_dataloaders=train_dataloader,
    )
    
    # Load the best model for evaluation
    checkpoint = torch.load(checkpoint_callback.best_model_path, weights_only=False)
    module.on_load_checkpoint(checkpoint)
    module.model.eval()
    
    # Create new trainer to avoid redundant event file
    trainer = Trainer(
        precision="bf16-mixed", # no need to manually convert to half precision in training_step
        accelerator="auto",
        devices=1, # disable DDP
        logger=False, # no logger for testing
        enable_checkpointing=False, # no checkpointing for testing
    )
    trainer.test(
        model=module,
        dataloaders=val_dataloader,
    )
    trainer.test(
        model=module,
        dataloaders=test_dataloader,
    )
