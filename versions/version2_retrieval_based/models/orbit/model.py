import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

class Classifier(nn.Module):
    def __init__(
        self, 
        config, 
    ):
        super(Classifier, self).__init__()
        
        # The best result is using profile embeddings only
        self.classifier = nn.Linear(config.hidden_dim, 1)
    
    def forward(
        self, 
        profile_embeddings: torch.Tensor, 
        context_embeddings: torch.Tensor, 
        topk_scores: torch.Tensor,
        labels: torch.Tensor=None,
    ):
        """ Forward pass of classifier.
        
        Args:
            profile_embeddings: Tensor of shape (batch_size, hidden_dim)
            context_embeddings: Tensor of shape (batch_size, hidden_dim)
            topk_scores: Tensor of shape (batch_size, top_k)
            labels: Tensor of shape (batch_size,) (optional)
        
        Returns:
            logits: Tensor of shape (batch_size, 1)
        """
        # Forward pass through the network
        logits = self.classifier(profile_embeddings)  # (batch_size, 1)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1),
                labels.float(),
            )
        
        # Return logits and loss
        return ModelOutput(
            logits=logits,
            loss=loss,
        )


class MultiClassifier(nn.Module):
    def __init__(
        self, 
        config, 
    ):
        super(MultiClassifier, self).__init__()
        
        self.classifier_1 = nn.Linear(config.hidden_dim, 1)
        self.classifier_2 = nn.Linear(config.hidden_dim, 1)
        self.classifier_3 = nn.Linear(config.top_k, 1)
    
    def forward(
        self, 
        profile_embeddings: torch.Tensor, 
        context_embeddings: torch.Tensor, 
        topk_scores: torch.Tensor,
        labels: torch.Tensor=None,
    ):
        """ Forward pass of classifier.
        
        Args:
            profile_embeddings: Tensor of shape (batch_size, hidden_dim)
            context_embeddings: Tensor of shape (batch_size, hidden_dim)
            topk_scores: Tensor of shape (batch_size, top_k)
            labels: Tensor of shape (batch_size,) (optional)
        
        Returns:
            logits: Tensor of shape (batch_size, 1)
        """
        # Forward pass through the network
        logits_1 = self.classifier_1(profile_embeddings)  # (batch_size, 1)
        logits_2 = self.classifier_2(context_embeddings)  # (batch_size, 1)
        logits_3 = self.classifier_3(topk_scores)  # (batch_size, 1)
        logits = (logits_1 + logits_2 + logits_3) / 3 # (batch_size, 1)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1),
                labels.float(),
            )
            # Don't calculate per-classifier BCE loss, the performance is not good.
        
        # Return logits and loss
        return ModelOutput(
            logits=logits,
            loss=loss,
        )


class ContextClassifier(nn.Module):
    def __init__(
        self, 
        config, 
    ):
        super(ContextClassifier, self).__init__()
        
        # A simple feedforward neural network as classifier
        self.output_layer = nn.Linear(config.hidden_dim * 2, 1)
    
    def forward(
        self, 
        profile_embeddings: torch.Tensor, 
        context_embeddings: torch.Tensor, 
        labels: torch.Tensor=None,
    ):
        """ Forward pass of classifier.
        
        Args:
            profile_embeddings: Tensor of shape (batch_size, hidden_dim)
            context_embeddings: Tensor of shape (batch_size, hidden_dim)
            labels: Tensor of shape (batch_size,) (optional)
        
        Returns:
            logits: Tensor of shape (batch_size, 1)
        """
        # Concat with profile_embeddings and feed into classifier
        combined_embeddings = torch.cat([profile_embeddings, context_embeddings], dim=-1) # (batch_size, hidden_dim * 2)
        
        # Forward pass through the network
        logits = self.output_layer(combined_embeddings)  # (batch_size, 1)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1),
                labels.float(),
            )
        
        # Return logits and loss
        return ModelOutput(
            logits=logits,
            loss=loss,
        )


class CrossAttentionClassifier(nn.Module):
    def __init__(
        self, 
        config,
    ):
        super(CrossAttentionClassifier, self).__init__()
        
        # A simple feedforward neural network as classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim), # Input layer
            nn.ReLU(),                             # Activation function
            nn.Dropout(config.dropout_rate),              # Dropout for regularization
            nn.Linear(config.hidden_dim, 1)               # Output layer
        )
    
    def forward(
        self, 
        profile_embeddings: torch.Tensor, 
        topk_embeddings: torch.Tensor, 
        labels: torch.Tensor=None,
    ):
        """ Forward pass of the cross attention classifier.
        
        Args:
            profile_embeddings: Tensor of shape (batch_size, hidden_dim)
            topk_embeddings: Tensor of shape (batch_size, top_k, hidden_dim)
            labels: Tensor of shape (batch_size,) (optional)
        
        Returns:
            logits: Tensor of shape (batch_size, 1)
        """
        # Cross attention, use profile_embeddings as query and topk_embeddings as key-value pairs
        attention_scores = torch.matmul(
            profile_embeddings.unsqueeze(1), # (batch_size, 1, hidden_dim)
            topk_embeddings.transpose(1, 2) # (batch_size, hidden_dim, top_k)
        ) / (profile_embeddings.size(-1) ** 0.5) # Divide by sqrt(hidden_dim) for scaling
        attention_weights = F.softmax(attention_scores, dim=-1) # (batch_size, 1, top_k)
        fused_embeddings = torch.matmul(attention_weights, topk_embeddings).squeeze(1) # (batch_size, hidden_dim)
        
        # Concat with profile_embeddings and feed into classifier
        combined_embeddings = torch.cat([profile_embeddings, fused_embeddings], dim=-1) # (batch_size, hidden_dim * 2)
        logits = self.classifier(combined_embeddings) # (batch_size, 1)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1),
                labels.float(),
            )
        
        # Return logits and loss
        return ModelOutput(
            logits=logits,
            loss=loss,
        )
