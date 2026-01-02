import os
from tqdm import tqdm
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(
    task_description: str, 
    query: str
) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def get_profile_embeddings(
    profiles: List[str], 
    model: AutoTokenizer, 
    tokenizer: AutoModel,
):
    """
    Generate embeddings for a list of profiles.
    
    Args:
        profiles: List of profile strings.
        model: Pretrained model for embedding generation.
        tokenizer: Tokenizer corresponding to the model.
    
    Returns:
        profile_embeddings: Tensor of shape (num_profiles, embedding_dim) containing the embeddings for each profile.
    """
    profile_embeddings = []
    
    for profile in tqdm(profiles, desc="Processing profiles"):
        # Tokenize the input text
        model_inputs = tokenizer(profile, truncation=False, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**model_inputs)
        
        # Pool the last hidden states to get the profile embedding (don't normalize it to preserve magnitude information for classifier)
        profile_embeddings.append(last_token_pool(outputs.last_hidden_state, model_inputs["attention_mask"]).to(torch.float32).clone())
        
        # Clear GPU memory
        del model_inputs
        del outputs
        torch.cuda.empty_cache()
    
    # Concatenate all profile embeddings
    profile_embeddings = torch.cat(profile_embeddings, dim=0)
    
    return profile_embeddings


def get_ranked_indices(
    profiles: List[str], 
    profile_embeddings: torch.Tensor, 
    model: AutoModel, 
    tokenizer: AutoTokenizer, 
    instruction: str, 
    exclude_self: bool,
):
    """
    Compute the top-k most similar profiles for a given set of profiles.
    
    Args:
        profiles: List of profile strings for the current split (train/valid/test).
        profile_embeddings: Embeddings from the training set used for similarity calculation.
        model: Pretrained model for embedding generation.
        tokenizer: Tokenizer corresponding to the model.
        instruction: Task-specific instruction string.
        exclude_self: Whether to exclude the self-similarity (used for train set).
    
    Returns:
        ranked_indices: List of tensors containing the ranked indices of similar profiles for each profile. 
        similarity_scores: List of tensors containing the similarity scores for each profile.
    """
    ranked_indices = []
    similarity_scores = []
    
    # Normalize profile embeddings for cosine similarity calculation
    profile_embeddings = F.normalize(profile_embeddings, p=2, dim=1)
    
    for i, profile in enumerate(tqdm(profiles, desc="Processing profiles for similarity")):
        # Tokenize the input text combined with the task instruction
        model_inputs = tokenizer(get_detailed_instruct(instruction, profile), truncation=False, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**model_inputs)
        
        # Pool the last hidden states to get the query embedding
        query_embedding = F.normalize(
            last_token_pool(outputs.last_hidden_state, model_inputs["attention_mask"]).to(torch.float32),
            p=2, dim=1
        )
        
        # Compute cosine similarity between this query and all profile embeddings
        cosine_similarity = (query_embedding @ profile_embeddings.T)[0]
        
        # Exclude self-similarity if we are processing the train set
        if exclude_self:
            cosine_similarity[i] = -float("inf")  # Exclude itself to the last
        
        # Get ranked indices based on similarity scores (in train set, self will be at the end)
        indices = torch.argsort(cosine_similarity, descending=True, stable=True)
        
        # Save the ranked indices
        ranked_indices.append(indices.cpu())
        similarity_scores.append(cosine_similarity.cpu())
        
        # Clear GPU memory
        del model_inputs
        del outputs
        torch.cuda.empty_cache()
    
    # Concatenate all ranked indices
    ranked_indices = torch.stack(ranked_indices, dim=0)
    similarity_scores = torch.stack(similarity_scores, dim=0)
    
    return ranked_indices, similarity_scores


if __name__ == "__main__":
    from data_module import HelocDataModule
    
    # Load the data module
    data_module = HelocDataModule()
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-Embedding-8B", 
        padding_side="left"
    )
    model = AutoModel.from_pretrained(
        "Qwen/Qwen3-Embedding-8B", 
        dtype="auto", 
        device_map="auto",
    )
    
    # ---- Step 1: Generate Profile Embeddings for the Entire Dataset ----
    profiles, targets = data_module.get_profile_dataset(data_module.indices)
    profile_embeddings = get_profile_embeddings(profiles, model, tokenizer)
    
    # ---- Step 2: Get Top-K Similar Profiles (Train, Valid, Test) ----
    instruction = "Given an applicant's credit profile, retrieve other applicant credit profiles that are similar or related to this one."

    # ---- Step 3: Process Training Set ----
    profiles, targets = data_module.get_profile_dataset(data_module.train_indices)
    train_ranked_indices, train_similarity_scores = get_ranked_indices(profiles, profile_embeddings[data_module.train_indices], model, tokenizer, instruction, exclude_self=True)
    
    # ---- Step 4: Process Train and Validation Set ----
    profiles, targets = data_module.get_profile_dataset(data_module.train_val_indices)
    train_val_ranked_indices, train_val_similarity_scores = get_ranked_indices(profiles, profile_embeddings[data_module.train_val_indices], model, tokenizer, instruction, exclude_self=True)
    
    # ---- Step 5: Process Validation Set ----
    profiles, targets = data_module.get_profile_dataset(data_module.valid_indices)
    val_ranked_indices, val_similarity_scores = get_ranked_indices(profiles, profile_embeddings[data_module.train_indices], model, tokenizer, instruction, exclude_self=False)

    # ---- Step 6: Process Test Set ----
    profiles, targets = data_module.get_profile_dataset(data_module.test_indices)
    test_ranked_indices, test_similarity_scores = get_ranked_indices(profiles, profile_embeddings[data_module.train_val_indices], model, tokenizer, instruction, exclude_self=False)
    
    # ---- Step 7: Save Results ----
    os.makedirs(f"datasets/heloc/data_splits/", exist_ok=True)
    torch.save(profile_embeddings, f"datasets/heloc/data_splits/profile_embeddings.pt")
    torch.save({
        "train_ranked_indices": train_ranked_indices,
        "val_ranked_indices": val_ranked_indices,
        "train_val_ranked_indices": train_val_ranked_indices,
        "test_ranked_indices": test_ranked_indices,
    }, f"datasets/data_splits/ranked_indices.pt")
    torch.save({
        "train_similarity_scores": train_similarity_scores,
        "val_similarity_scores": val_similarity_scores,
        "test_similarity_scores": test_similarity_scores,
        "train_val_similarity_scores": train_val_similarity_scores,
    }, f"datasets/heloc/data_splits/similarity_scores.pt")
