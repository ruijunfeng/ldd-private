import os

import torch
from lightning.pytorch import LightningModule

from utils.json_utils import update_json
from utils.plot_utils import parse_events_and_save_plot

from models.orbit.config import OrbitConfig
from models.orbit.model import Classifier

class SupervisedTuningModule(LightningModule):
    """
    Supervised tuning module for binary classification.
    
    Args:
        model: The model to be trained.
        config: The configuration object containing hyperparameters.
    """
    def __init__(
        self,
        model: Classifier,
        config: OrbitConfig,
        **kwargs,
    ):
        super().__init__()
        # Save the passed hyperparameters
        self.save_hyperparameters(ignore=["model"])
        
        # Model settings
        self.model = model
        
        # Enable manual optimization
        self.automatic_optimization = True
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.forward(
            profile_embeddings=batch["profile_embeddings"],
            context_embeddings=batch["context_embeddings"],
            topk_scores=batch["topk_scores"],
            labels=batch["labels"],
        )
        
        # Loss is computed inside the model when labels are provided (default is cross entropy loss)
        self.log("total_loss", outputs["loss"], prog_bar=True, on_step=True, on_epoch=False)
        
        return outputs["loss"]
    
    def validation_step(self, batch, batch_idx):        
        # Forward pass (batch size is 1 during validation for stability)
        outputs = self.forward(
            profile_embeddings=batch["profile_embeddings"],
            context_embeddings=batch["context_embeddings"],
            topk_scores=batch["topk_scores"],
            labels=batch["labels"],
        )
        
        # Logging (during evaluation, normally save by epoch rather than step)
        self.log("total_loss_val", outputs["loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
    
    def test_step(self, batch, batch_idx):
        # Forward pass (batch size is 1 during testing for stability)
        outputs = self.forward(
            profile_embeddings=batch["profile_embeddings"],
            context_embeddings=batch["context_embeddings"],
            topk_scores=batch["topk_scores"],
        )
        logits = outputs.logits[0]
        y_proba = torch.sigmoid(logits)
        y_pred = (y_proba >= 0.5).float()
        update_json(f"{self.log_dir}/predictions.json", batch["indices"][0].item(), {"y_pred": y_pred.tolist(), "y_proba": y_proba.tolist(), "logits": logits.tolist()})
    
    def on_test_end(self):
        # Visualize the TensorBoard logs
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
        # Remove 'model.' prefix in state_dict keys before saving
        checkpoint["state_dict"] = {
            k[len("model."):]: v for k, v in self.state_dict().items() if k.startswith("model.")
        }
    
    def on_load_checkpoint(self, checkpoint):
        # Manually load the state_dict into self.model
        state_dict = checkpoint["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
