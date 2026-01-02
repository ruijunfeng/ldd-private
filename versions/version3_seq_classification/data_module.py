
import os
import copy
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from utils.meta_utils import load_metadata
from utils.profile_utils import generate_profile
from utils.json_utils import load_json, save_json


class HelocDataModule:
    def __init__(
        self,
    ):
        self.label_map = {"Good": 0, "Bad": 1}
        
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
        indices: List,
    ):
        X = self.df.iloc[indices].drop(columns=["RiskPerformance", "ApplicantProfile"]).values
        y = self.df.iloc[indices]["RiskPerformance"].map(self.label_map).values
        return X, y
    
    def get_profile_dataset(
        self,
        indices: list,
    ):
        dataset = []
        for index in indices:
            dataset.append({
                "indices": index,
                "numeric_features": self.df.iloc[index].drop(labels=["RiskPerformance", "ApplicantProfile"]).values,
                "profiles": self.df.iloc[index]["ApplicantProfile"],
                "labels": self.label_map[self.df.iloc[index]["RiskPerformance"]],
            })
        return dataset
    

    def get_seq_class_dataloader(
        self,
        indices: List,
        tokenizer: AutoTokenizer,
        question_template: str,
        batch_size: int,
    ):
        """Returns a DataLoader in Sequence Classfication Format based on the provided indices.
        
        Args:
            indices (List): List of indices to select from the dataset.
            tokenizer (AutoTokenizer): The tokenizer to use.
            question_template (str): The template for the question prompt.
            batch_size (int): The batch size for the DataLoader.
        
        Returns:
            dataloader: A DataLoader object with dynamic padding in Sequence Classification Format.
        """
        def tokenize_fn(example, tokenizer, max_length=None):
            """Tokenize a single example into input_ids and attention_mask.
            In sequence classification, we only need to tokenize the question prompt.
            There is no need to use the answer_template here.
            """
            prompt = question_template.format(
                profile=example["profiles"],
            )
            messages = [
                {"role": "user", "content": prompt}
            ]
            model_inputs = tokenizer.apply_chat_template(
                messages, # no need to wrap with answer_template
                tokenize=True,
                add_generation_prompt=True,
                truncation=False,
                padding=False,
                max_length=max_length,
            )
            
            return {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "labels": example["labels"],
                "numeric_features": example["numeric_features"],
                "indices": example["indices"],
            }
        
        def collate_fn(features):
            # Separate indices and numeric features from collate_fn
            indices = [f["indices"] for f in features]
            numeric_features = [f["numeric_features"] for f in features]
            labels = [f["labels"] for f in features]
            # Remove from dict so collator doesn't complain
            for f in features:
                f.pop("indices")
                f.pop("numeric_features")
                f.pop("labels")
            # Use the data collator to get input_ids and labels
            batch = data_collator(features)
            # Add back indices and numeric features
            batch["indices"] = indices
            batch["numeric_features"] = torch.tensor(numeric_features, dtype=torch.float32)
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
            return batch
        
        # create a dataset
        numeric_features = []
        profiles = []
        labels = []
        for index in indices:
            numeric_features.append(self.df.iloc[index].drop(labels=["RiskPerformance", "ApplicantProfile"]).values.astype(int).tolist())
            profiles.append(self.df.iloc[index]["ApplicantProfile"])
            labels.append(self.label_map[self.df.iloc[index]["RiskPerformance"]])
        dataset = Dataset.from_dict({"indices": indices, "numeric_features": numeric_features, "profiles": profiles, "labels": labels})
        dataset = dataset.map(
            tokenize_fn,
            fn_kwargs={"tokenizer": tokenizer, "max_length": None},
            batched=False, # make tokenize_fn to process sample by sample
            remove_columns=["profiles"],
        )
        
        # create a dynamic padding collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
        
        # dataLoader with dynamic padding
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False, # no shuffle as train_test_split already shuffles
            collate_fn=collate_fn,
        )
        
        return dataloader
