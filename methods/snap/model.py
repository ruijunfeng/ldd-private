from typing import Optional

import torch
from torch import nn

from methods.snap.prompt_encoder import NumericalPromptEncoder

class SNAP(nn.Module):
    """
    Similar to PeftModelForCausalLM, this class is a wrapper for SNAP. 
    The difference is that the prompt encoder here takes a numerical features as input.
    
    Args:
        config: The configuration of the prompt encoder.
        base_model: The base model to be used.
    """
    def __init__(self, config, base_model):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.word_embeddings = self.base_model.get_input_embeddings()
        self.prompt_encoder = NumericalPromptEncoder(
            use_numerical_embedding=config.use_numerical_embedding,
            use_multi_head_self_attn=config.use_multi_head_self_attn,
            num_features=config.num_features,
            embed_dim=self.word_embeddings.weight.shape[1],
            head_dim=config.head_dim,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        numeric_features: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Initialize the input_embeds
        batch_size = input_ids.shape[0]
        input_embeds = self.word_embeddings(input_ids)
        
        # Generate soft prompts
        soft_prompts = self.prompt_encoder(numeric_features)
        total_virtual_tokens = soft_prompts.size(1)
        
        # Concat soft_prompts with input_embeds
        input_embeds = torch.cat((soft_prompts, input_embeds), dim=1)
        # Concat attention mask with prefix
        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, total_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        # Concat labels with prefix
        if labels is not None:
            prefix_labels = torch.full((batch_size, total_virtual_tokens), -100).to(labels.device) # prefix the labels with -100 (ignore index)
            labels = torch.cat((prefix_labels, labels), dim=1)
        
        # Forward pass with the base_model
        outputs = self.base_model(
            inputs_embeds=input_embeds, 
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        # Adds the soft_prompts to the outputs
        outputs["soft_prompts"] = soft_prompts
        return outputs
    
    def generate(
        self, 
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        numeric_features: torch.Tensor = None,
        **kwargs,
    ):
        # Generate soft prompts
        batch_size, seq_len = input_ids.shape
        input_embeds = self.word_embeddings(input_ids)
        soft_prompts = self.prompt_encoder(numeric_features)
        total_virtual_tokens = soft_prompts.size(1)
        
        # Concat soft prompts
        input_embeds = torch.cat((soft_prompts, input_embeds), dim=1)
        # Concat attention mask with prefix
        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, total_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        # Generate using the base_model
        outputs = self.base_model.generate(
            inputs_embeds=input_embeds, 
            attention_mask=attention_mask,
            **kwargs,
        )
        
        # Adds the soft_prompts to the outputs
        outputs["soft_prompts"] = soft_prompts
        
        return outputs
    
    def print_trainable_parameters(self):
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters())
        # Calculate trainable percentage
        trainable_percent = trainable_params / total_params * 100 if total_params > 0 else 0
        # Print the information in the desired format
        print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_percent:.4f}")
