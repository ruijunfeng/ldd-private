
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from utils.meta_utils import load_metadata
from utils.profile_utils import generate_profile
from utils.json_utils import load_json, save_json
from utils.emb_utils import get_profile_embeddings, get_ranked_indices

from models.orbit.config import OrbitConfig


class HelocBaseDataModule:
    def __init__(
        self,
    ):
        # Paths
        excel_path = "datasets/heloc/raw/heloc_data_dictionary-2.xlsx"
        csv_path = "datasets/heloc/raw/heloc_dataset_v1.csv"
        self.data_split_path = "datasets/heloc/data_splits/"
        os.makedirs(self.data_split_path, exist_ok=True)
        
        # Load HELOC metadata and dataset
        data_dict, max_delq_dict, special_vals = load_metadata(excel_path)
        self.df = pd.read_csv(csv_path)
        
        # Generate applicant profiles
        self.df["ApplicantProfile"] = self.df.apply(
            lambda row: generate_profile(
                row,
                max_delq_dict=max_delq_dict,
                special_values=special_vals,
            ),
            axis=1,
        )
        
        # Generate indices
        self.setup_splits()
    
    def setup_splits(self):
        self.indices = np.arange(len(self.df))
        if not os.path.exists(os.path.join(self.data_split_path, "valid_indices.json")):
            # Remove samples where all values are -9 and samples contain -7 or -8
            feature_cols = self.df.columns.drop(["RiskPerformance", "ApplicantProfile"])
            valid_mask = ~(self.df[feature_cols].eq(-9).all(axis=1) | self.df[feature_cols].isin([-7, -8]).any(axis=1))
            self.valid_indices = self.indices[valid_mask].tolist()
            
            # Split into training and test indices
            self.train_val_indices, self.test_indices = train_test_split(self.valid_indices, test_size=0.2, random_state=42)
            
            # From the training set, create a validation set
            self.train_indices, self.val_indices = train_test_split(self.train_val_indices, test_size=0.2, random_state=42)
            
            # Save indices
            save_json(os.path.join(self.data_split_path, "valid_indices.json"), self.valid_indices)
            save_json(os.path.join(self.data_split_path, "train_indices.json"), self.train_indices)
            save_json(os.path.join(self.data_split_path, "val_indices.json"), self.val_indices)
            save_json(os.path.join(self.data_split_path, "train_val_indices.json"), self.train_val_indices)
            save_json(os.path.join(self.data_split_path, "test_indices.json"), self.test_indices)
        else:
            self.valid_indices = load_json(os.path.join(self.data_split_path, "valid_indices.json"))
            self.train_indices = load_json(os.path.join(self.data_split_path, "train_indices.json"))
            self.val_indices = load_json(os.path.join(self.data_split_path, "val_indices.json"))
            self.train_val_indices = load_json(os.path.join(self.data_split_path, "train_val_indices.json"))
            self.test_indices = load_json(os.path.join(self.data_split_path, "test_indices.json"))
    
    def get_feature_dataset(
        self,
        indices: list,
    ):
        X = self.df.iloc[indices].drop(columns=["RiskPerformance", "ApplicantProfile"]).values
        y = self.df.iloc[indices]["RiskPerformance"].map({"Good": 0, "Bad": 1}).values
        return X, y
    
    def get_profile_dataset(
        self,
        indices: list,
    ):
        profiles = self.df.iloc[indices]["ApplicantProfile"].values
        labels = self.df.iloc[indices]["RiskPerformance"].map({"Good": 0, "Bad": 1}).values
        return profiles, labels


class HelocDataModule(HelocBaseDataModule):
    def __init__(
        self,
        config: OrbitConfig,
    ):
        super().__init__()
        self.config = config
        self.setup_embeddings()
        
    def setup_embeddings(self):
        if not os.path.exists(os.path.join(self.data_split_path, "profile_embeddings.pt")):
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, 
                padding_side="left"
            )
            model = AutoModel.from_pretrained(
                self.config.model_name, 
                dtype="auto", 
                device_map="auto",
            )
            
            # ---- Step 1: Generate Profile Embeddings for the Entire Dataset ----
            profiles, labels = self.get_profile_dataset(self.indices)
            self.profile_embeddings = get_profile_embeddings(profiles, model, tokenizer)
            
            # ---- Step 2: Process Training Set ----
            # For training set, self-exclusion is needed
            profiles, labels = self.get_profile_dataset(self.train_indices)
            self.train_ranked_indices, self.train_similarity_scores = get_ranked_indices(profiles, self.profile_embeddings[self.train_indices], model, tokenizer, self.config.instruction, exclude_self=True)
            
            # ---- Step 3: Process Validation Set ----
            # For validation set, only the training set is used as reference, self-exclusion is not needed as validation samples are not in training set
            profiles, labels = self.get_profile_dataset(self.val_indices)
            self.val_ranked_indices, self.val_similarity_scores = get_ranked_indices(profiles, self.profile_embeddings[self.train_indices], model, tokenizer, self.config.instruction, exclude_self=False)
            
            # ---- Step 4: Process Train and Validation Set ----
            # For train + validation set combined, self-exclusion is needed
            profiles, labels = self.get_profile_dataset(self.train_val_indices)
            self.train_val_ranked_indices, self.train_val_similarity_scores = get_ranked_indices(profiles, self.profile_embeddings[self.train_val_indices], model, tokenizer, self.config.instruction, exclude_self=True)
            
            # ---- Step 5: Process Test Set ----
            # For test set, both training and validation sets are used as reference, self-exclusion is not needed as test samples are not in training/validation set
            profiles, labels = self.get_profile_dataset(self.test_indices)
            self.test_ranked_indices, self.test_similarity_scores = get_ranked_indices(profiles, self.profile_embeddings[self.train_val_indices], model, tokenizer, self.config.instruction, exclude_self=False)
            
            # ---- Step 6: Save Results ----
            torch.save(self.profile_embeddings.cpu(), f"{self.data_split_path}/profile_embeddings.pt")
            torch.save({
                "train_ranked_indices": self.train_ranked_indices,
                "val_ranked_indices": self.val_ranked_indices,
                "train_val_ranked_indices": self.train_val_ranked_indices,
                "test_ranked_indices": self.test_ranked_indices,
            }, f"{self.data_split_path}/ranked_indices.pt")
            torch.save({
                "train_similarity_scores": self.train_similarity_scores,
                "val_similarity_scores": self.val_similarity_scores,
                "test_similarity_scores": self.test_similarity_scores,
                "train_val_similarity_scores": self.train_val_similarity_scores,
            }, f"{self.data_split_path}/similarity_scores.pt")
        else:
            self.profile_embeddings = torch.load(f"{self.data_split_path}/profile_embeddings.pt")
            ranked_indices = torch.load(f"{self.data_split_path}/ranked_indices.pt")
            self.train_ranked_indices = ranked_indices["train_ranked_indices"]
            self.val_ranked_indices = ranked_indices["val_ranked_indices"]
            self.train_val_ranked_indices = ranked_indices["train_val_ranked_indices"]
            self.test_ranked_indices = ranked_indices["test_ranked_indices"]
            similarity_scores = torch.load(f"{self.data_split_path}/similarity_scores.pt")
            self.train_similarity_scores = similarity_scores["train_similarity_scores"]
            self.val_similarity_scores = similarity_scores["val_similarity_scores"]
            self.test_similarity_scores = similarity_scores["test_similarity_scores"]
            self.train_val_similarity_scores = similarity_scores["train_val_similarity_scores"]
    
    def get_enhanced_feature_dataset(
        self,
        indices: list,
        ranked_indices: list,
        similarity_scores: torch.Tensor,
        top_k: int,
    ):
        # Get profile embeddings based on indices as only features
        # profiles = self.df.iloc[indices].drop(columns=["RiskPerformance", "ApplicantProfile"]).values
        
        # Get top-k scores based on the labels of the top-k neighbours as additional features
        # topk_similarity_scores = torch.gather(similarity_scores, 1, ranked_indices[:, :top_k]) # (num_samples, top_k)
        # topk_labels = torch.tensor(self.df["RiskPerformance"].map({"Good": -1, "Bad": 1}).values)[ranked_indices[:, :top_k]] # (num_samples, top_k)
        # topk_scores = topk_similarity_scores * topk_labels # (num_samples, top_k)
        # topk_scores = torch.mean(topk_scores, dim=1, keepdim=True)  # (num_samples, 1)
        # profiles = np.hstack((self.df.iloc[indices].drop(columns=["RiskPerformance", "ApplicantProfile"]).values, topk_scores.to(torch.float32).numpy()))
        
        # Get profile embeddings as additional features
        # profiles = np.hstack((self.df.iloc[indices].drop(columns=["RiskPerformance", "ApplicantProfile"]).values, self.profile_embeddings[indices][:, :32].to(torch.float32).numpy()))
        
        top1_similarity_scores = torch.gather(similarity_scores, 1, ranked_indices[:, 0:1]) # (num_samples, 1)
        top1_labels = torch.tensor(self.df["RiskPerformance"].map({"Good": 0, "Bad": 1}).values)[ranked_indices[:, 0:1]] # (num_samples, 1)
        profiles = np.hstack((self.df.iloc[indices].drop(columns=["RiskPerformance", "ApplicantProfile"]).values, top1_labels.to(torch.float32).numpy(), top1_similarity_scores.to(torch.float32).numpy()))
        
        labels = self.df.iloc[indices]["RiskPerformance"].map({"Good": 0, "Bad": 1}).values
        return profiles, labels
    
    def get_dataloader(
        self,
        indices: list,
        ranked_indices: list,
        similarity_scores: torch.Tensor,
        top_k: int,
        batch_size: int,
    ):
        # Get profile_embeddings based on indices
        profile_embeddings = self.profile_embeddings[indices] # (num_samples, hidden_dim)
        # Get top-k embeddings and cosine similarity scores
        topk_embeddings = self.profile_embeddings[ranked_indices[:, :top_k]] # (num_samples, top_k, hidden_dim)
        topk_similarity_scores = torch.gather(similarity_scores, 1, ranked_indices[:, :top_k]) # (num_samples, top_k)
        # Get top-k scores based on the labels of the top-k neighbours
        topk_labels = torch.tensor(self.df["RiskPerformance"].map({"Good": -1, "Bad": 1}).values)[ranked_indices[:, :top_k]] # (num_samples, top_k)
        topk_scores = topk_similarity_scores * topk_labels # (num_samples, top_k)
        # Get context embeddings based on top-k neighours profile_embeddings
        topk_similarity_scores = F.softmax(topk_similarity_scores, dim=1).unsqueeze(1) # (num_samples, 1, top_k)
        context_embeddings = torch.bmm(topk_similarity_scores, topk_embeddings).squeeze(1) # (num_samples, hidden_dim)
        # Get labels
        labels = torch.tensor(self.df.iloc[indices]["RiskPerformance"].map({"Good": 0, "Bad": 1}).values)
        # Convert into DataLoader
        dataset = HelocDataset(profile_embeddings.to(torch.float32), context_embeddings.to(torch.float32), topk_scores.to(torch.float32), labels, indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader


class HelocDataset(Dataset):
    def __init__(
        self, 
        profile_embeddings: torch.Tensor, 
        context_embeddings: torch.Tensor, 
        topk_scores: torch.Tensor, 
        labels: torch.Tensor, 
        indices: torch.Tensor, 
    ):
        self.profile_embeddings = profile_embeddings
        self.context_embeddings = context_embeddings
        self.topk_scores = topk_scores
        self.labels = labels
        self.indices = indices
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "profile_embeddings": self.profile_embeddings[idx],
            "context_embeddings": self.context_embeddings[idx],
            "topk_scores": self.topk_scores[idx],
            "labels": self.labels[idx],
            "indices": self.indices[idx],
        }
