import os

import torch
from lightning.pytorch import LightningModule

from utils.json_utils import update_json
from utils.plot_utils import parse_events_and_save_plot

class SFTModule(LightningModule):
    """
    Supervised fine-tuning module for binary classification.
    
    Args:
        model: The model to be trained.
        config: The configuration object containing hyperparameters.
    """
    def __init__(
        self,
        model,
        tokenizer,
        config,
        **kwargs,
    ):
        super().__init__()
        # Save the passed hyperparameters
        self.save_hyperparameters(ignore=["model"])
        
        # Model settings
        self.model = model
        self.choice_ids = tokenizer(["Good", "Bad"], add_special_tokens=False).input_ids
        
        # Enable manual optimization
        self.automatic_optimization = True
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            numeric_features=batch["numeric_features"],
            labels=batch["labels"],
        )
        
        # Loss is computed inside the model when labels are provided (default is cross entropy loss)
        self.log("total_loss", outputs["loss"], prog_bar=True, on_step=True, on_epoch=False)
        
        return outputs["loss"]
    
    def validation_step(self, batch, batch_idx):        
        # Forward pass (batch size is 1 during validation for stability)
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            numeric_features=batch["numeric_features"],
            labels=batch["labels"],
        )
        
        # Logging (during evaluation, normally save by epoch rather than step)
        self.log("total_loss_val", outputs["loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
    
    def test_step(self, batch, batch_idx):
        # Forward pass (batch size is 1 during testing for stability)
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            numeric_features=batch["numeric_features"],
        )
        
        # classification head
        logits = outputs.logits[0]
        y_pred = torch.argmax(logits).item()
        y_proba = torch.softmax(logits, dim=0)[1].item() # probability of "Bad"
        
        update_json(f"{self.log_dir}/predictions.json", batch["indices"][0], {"y_pred": y_pred, "y_proba": y_proba, "logits": logits.tolist()})
    
    def on_test_end(self):
        # Visualize the TensorBoard logs
        if not os.path.exists(f"{self.log_dir}/pictures.png"):
            for filename in os.listdir(self.log_dir):
                if filename.startswith("events.out.tfevents"):
                    parse_events_and_save_plot(
                        event_file_path=f"{self.log_dir}/{filename}",
                        output_image_path=f"{self.log_dir}/pictures.png",
                    )
    
    def configure_optimizers(self):
        # Set the log directory for saving predictions
        self.log_dir = self.trainer.log_dir
        # Optimizer settings
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.hparams.config.lr)
        return [optimizer], []
    
    def on_save_checkpoint(self, checkpoint):
        # Get the default state_dict
        state_dict = self.state_dict()
        # Remove base_model from the state_dict as peft don't change the base_model
        keys_to_remove = [key for key in state_dict if "lora" not in key and "prompt_encoder" not in key and "score" not in key]
        for key in keys_to_remove:
            del state_dict[key]
        # Remove 'model.' prefix in state_dict keys before saving
        checkpoint["state_dict"] = {
            k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")
        }
    
    def on_load_checkpoint(self, checkpoint):
        # Manually load the state_dict into self.model
        state_dict = checkpoint["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
