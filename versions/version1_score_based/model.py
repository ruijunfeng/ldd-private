import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from data_module import HelocDataModule


def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


data_module = HelocDataModule()
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-Embedding-8B", 
    padding_side="left"
)
model = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-8B", 
    dtype="auto", 
    device_map="auto",
)


profiles, labels = data_module.get_profile_dataset(data_module.train_indices)
profile_embeddings = []

for profile in tqdm(profiles):
    # Tokenize the input texts
    model_inputs = tokenizer(
        profile,
        truncation=False,
        return_tensors="pt",
    ).to(model.device)
    
    # inference
    with torch.no_grad():
        outputs = model(**model_inputs)
    
    # pool the last hidden states as the profile embedding
    profile_embedding = last_token_pool(outputs.last_hidden_state, model_inputs["attention_mask"])
    
    # normalize profile embedding
    profile_embeddings.append(F.normalize(profile_embedding, p=2, dim=1))
    
    # clear GPU memory
    del model_inputs
    del outputs
    torch.cuda.empty_cache()

profile_embeddings = torch.cat(profile_embeddings, dim=0)
scores = torch.tensor(labels, device=model.device).float()
scores = torch.where(scores == 0, torch.tensor(-1), scores) # convert 0 into -1


# Each query must come with a one-sentence instruction that describes the task
instruction = "Given an applicant's credit profile, retrieve other applicant credit profiles that are similar or related to this one."
profiles, labels = data_module.get_profile_dataset(data_module.val_indices)
scores = torch.tensor(scores, device=model.device)
y_prob = []

for profile in tqdm(profiles):
    # Tokenize the input texts
    model_inputs = tokenizer(
        get_detailed_instruct(instruction, profile),
        truncation=False,
        return_tensors="pt",
    ).to(model.device)
    
    # inference
    with torch.no_grad():
        outputs = model(**model_inputs)
    
    # pool the last hidden states as the query embedding
    query_embedding = F.normalize(last_token_pool(outputs.last_hidden_state, model_inputs["attention_mask"]), p=2, dim=1)
    
    # calculate cosine similarity
    cosine_similarity = (query_embedding @ profile_embeddings.T)[0]
    
    # calculate local default density as y_prob
    indices = torch.argsort(cosine_similarity, descending=True, stable=True)[:40]
    y_prob.append((cosine_similarity[indices] * scores[indices]).mean().item())

from sklearn.metrics import classification_report, accuracy_score
print(accuracy_score(torch.tensor(labels), (torch.tensor(y_prob) > 0).int()))
print(classification_report(torch.tensor(labels), (torch.tensor(y_prob) > 0).int()))
